import torch
import numpy as np
from .eim_class import eim_vectorial
from .utils import unravel_index


class Eim(eim_vectorial):
    """

    Summary :

        this class implemented the EIM algorithm:

    How to use it:

        >>> Z = torch.zeros(5,5,1):
        >>> ev = eim_vectorial_optim(Z)
        >>> ev.reach_precision(plot=False,silent=True)
        >>> ev.save_model("data/model/test.mdl")
        >>> ev_load  = eim_vectorial_optim(None,load=True):
        >>> ev_load.load_model("data/model/test.mdl")


    Class attribute:

            x_magics

            mu_magics

            j_magics

            Q_tab

    """

    def __init__(self, Z, from_numpy=False, gpu=True, max_shape="max", load=False):
        """

        init method of class eim_vectorial_optim

        Keyword arguments:


            * Z  : Tensor : should be a tensor of shape : (M,N,D)

            * from_numpy : default values is False, if True it will convet the Z entry from a numpy ndarray to the equivalent torch Tensor

            * gpu : Boolean : if avalaible the computation will be done on the GPU

            * max_shape : Integer :default value "max", should be an integer otherwise, it will preallocated memory you wont be able to do more than max_shape iteration

            * load : Boolean : default value False, will skip init if the purpose of the object is to load a pretrained model


        """

        if torch.cuda.is_available() and gpu:
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        if not (load):

            if len(Z.shape) == 2:
                Z = Z.reshape(Z.shape[0], Z.shape[1], 1)
            elif len(Z.shape) != 3:
                raise ValueError(" Z shloud be a tensor of shape : (M,N,D)")

            if from_numpy:
                self.Z = torch.from_numpy(Z)
            else:
                self.Z = Z

            if max_shape == "max":
                max_shape = min(self.Z.shape[0], self.Z.shape[1])

            self.x_magics = torch.zeros(max_shape, dtype=torch.long)

            self.mu_magics = torch.zeros(max_shape, dtype=torch.long)

            self.j_magics = torch.zeros(max_shape, dtype=torch.long)

            self.Q_tab = torch.zeros(
                (max_shape, self.Z.shape[1], self.Z.shape[2]), dtype=self.Z.dtype
            )

            self.m = 0
            ### to gpu

            self.Z = self.Z.to(self.device)
            self.Q_tab = self.Q_tab.to(self.device)

    def norme_infini(self, X):
        return torch.abs(X).max()

    def _quadratic_error(self):
        return ((self.Z - self.I) ** 2).mean().to("cpu")

    def _compute_Q_at_magic_points(self, m):
        return (self.Q_tab[0:m, self.x_magics[0:m], self.j_magics[0:m]].T).to(
            self.device
        )

    def compute_alpha(self, m, Z=None, z_at_magic_points=None):
        """
        Summary

            this method will calculate alphas : projection coef in the eim basis for new data

        Return value

            alphas : torch Tensor (m,Z.shape[0])

        keyword arguments

            m : integer : it is the eim basis that you want to use you would usualy wanted to use self.m
            Z: Torch Tensor (N,M,D) the input datas that you want to project in the m-eim basis

        """
        with torch.no_grad():

            mat = self._compute_Q_at_magic_points(m)

            if z_at_magic_points is None:
                z_at_magic_points = Z[:, self.x_magics[0:m], self.j_magics[0:m]]
            else:
                if Z is not None:
                    raise ValueError(
                        " if z_at_magic_points is passed qs args Z should be None "
                    )

            z_at_magic_points = z_at_magic_points.T.view(m, z_at_magic_points.shape[0])
            z_at_magic_points = z_at_magic_points.to(self.device)

            alpha, lu = torch.solve(z_at_magic_points, mat)

            return alpha

    def project_with_alpha(self, alpha):
        """
        Summary :

            given alphas coeficients it will compute the eim interpolation in the eim basis

        Return value:

            I_values : torch tensor

        Keyword argument :

            alpha : torch tensor ( m,M)


        """
        with torch.no_grad():

            m = alpha.shape[0]
            alpha.to(self.device)
            I_values = alpha.T @ (
                self.Q_tab[0:m, :].view((m, -1))
            )  # we need here to reshape the (m,N,ND) tensor to a (m,N*ND) matrix
            I_values = I_values.to(self.device)

            return I_values.view(
                alpha.shape[1], self.Q_tab.shape[1], self.Q_tab.shape[2]
            )  # and we reshape it to a (1,N,ND)

    def _interpol_mu(self, m, Z):

        # m-ème iteration
        with torch.no_grad():
            if m > 0:

                alpha = self.compute_alpha(m, Z)
                return self.project_with_alpha(alpha)

            else:
                return torch.zeros((Z.shape[1], Z.shape[2])).to(
                    self.device
                )  # first case

    def _iteration_n(self, epsilon=1e-4):
        ## base algo

        with torch.no_grad():
            self.I = self._interpol_mu(self.m, self.Z)

            self.I.to(self.device)

            a = torch.argmax(torch.abs(self.Z - self.I))
            [arg_mu_0, arg_x0, arg_j0] = unravel_index(a, self.Z.shape)

            bas = (self.Z - self.I)[arg_mu_0, arg_x0, arg_j0]
            dist_max = torch.abs(bas)

            ## update class variable
            if dist_max <= epsilon:
                return dist_max.to("cpu")

            q = (self.Z - self.I)[arg_mu_0] / bas

            self.Q_tab[self.m] = q
            self.mu_magics[self.m] = arg_mu_0
            self.x_magics[self.m] = arg_x0
            self.j_magics[self.m] = arg_j0

            self.m = self.m + 1

            return dist_max.to("cpu")

    def _reset_(self):
        return self.__init__(self.Z, from_numpy=False)

    def compress_model(self):
        """
        Summary :

            it will shrink the data and delete useless preallocated zeros be careful you wont be able to train the model further more

        """
        self.x_magics = self.x_magics[0 : self.m]

        self.mu_magics = self.mu_magics[0 : self.m]

        self.j_magics = self.j_magics[0 : self.m]

        self.Q_tab = self.Q_tab[0 : self.m]

    def save_model(self, file, compress=True):
        """
        Summary :

            will save model parameters

        Keyword argument:

            file : string : path to the file were you want to save the model parameters
            compress : Boolen : default value True : if set to True will call the compress_model method it will shrink the data and delete useless preallocated zeros


        """

        if compress:
            self.compress_model()
        torch.save(
            [self.Q_tab, self.mu_magics, self.x_magics, self.j_magics, self.m], file
        )

    def load_model(self, file):
        """
        Summary :

            it will load a pretrain model from a file

        Keyword argument:

            file : string : path to the file were models parameters are stored


        """
        [self.Q_tab, self.mu_magics, self.x_magics, self.j_magics, self.m] = torch.load(
            file
        )


class eim_vectorial_optim(Eim):
    pass
