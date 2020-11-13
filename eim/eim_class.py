import time
import matplotlib.pyplot as plt
import numpy as np
import itertools
from tqdm import tqdm


class eim_vectorial:

    def __init__(self,Z):
        '''
        Z shloud have a  spahe : (M,N,D) where N is the number of point and M the number of mu, and D the dimension of the final space
        '''
        self.Z = Z

        self.x_magics = np.zeros(Z.shape[0]).astype(int)

        self.mu_magics = np.zeros(Z.shape[0]).astype(int)

        self.j_magics = np.zeros(Z.shape[0]).astype(int)

        self.Q_tab=np.zeros((Z.shape[0],self.Z.shape[1],self.Z.shape[2]))

        self.m = 0



    def norme_infini(self,X):
        return np.absolute(X).max()

    def _interpol_mu(self,mu_i):


        m = self.m # m-ème iteration

        if m>0:

            self.mat = self.Q_tab[0:self.m,self.x_magics[0:self.m],self.j_magics[0:self.m]].T

            b = self.Z[mu_i,self.x_magics[0:self.m],self.j_magics[0:self.m]]

            alpha = np.linalg.solve(self.mat,b)
            I_values= np.array([ (alpha.T)@self.Q_tab[0:self.m,x_i] for x_i in range(self.Z.shape[1]) ])

            return I_values

        else:
            self.mat = np.zeros((0,0))
            return np.zeros((self.Z.shape[1],self.Z.shape[2])) # first case



    def _iteration_n(self,epsilon= 1e-4):

        self.I = np.array([self._interpol_mu(i)  for i in range(len(self.Z))])
        self.diff_mat = self.Z - self.I

        self.dist_mat = np.absolute(self.diff_mat)
        self.dist = np.array([self.norme_infini(d) for d in self.dist_mat])

        arg_mu_0 = self.dist.argmax()# arg of the mu 0 ( magic point)

        if np.abs(self.dist[arg_mu_0]) <= epsilon:
            return np.abs(self.dist[arg_mu_0])

        # the lines below calculate the index (that represent the mu) that maximize the error of the interpolator with respect to infinite norme
        x0_list,j0_list = np.where(self.dist_mat[arg_mu_0]==self.dist_mat[arg_mu_0].max())
        x0 = x0_list[0]
        j0 = j0_list[0]
        # then we find in wich point ( x0 E R**n) the error is maximal we will then try to reduce the error at that point


        bas = self.diff_mat[arg_mu_0,x0,j0]
        q = self.diff_mat[arg_mu_0]/bas
        self.Q_tab[self.m] = q


        self.mu_magics[self.m] = arg_mu_0
        self.x_magics[self.m] = x0
        self.j_magics[self.m] = j0

        self.m = self.m + 1


        return self.dist[arg_mu_0]


    def _reset_(self):
        self.__init__(self.Z)

    def _quadratic_error(self):
        self.quad_diff = self.dist_mat**2
        return self.quad_diff.mean()


    def reach_precision(self,epsilon=1e-2,nb_iter=float('inf'),reset = True,silent = False,plot=True):
        
        '''
        Summary:
        
        This method will compute eim steps to achieve the wanted precision. Be careful with the epsilon value you should keep epislon not to short oterhwise eim loose his interest and you should used gauss pivot for same result and better perfomrmance.

        Keyword Arguments : 

        * epsilon : float : precision wanted
        
        * nb_iter : integer : default is python infinite, it set the number of iteration before the algorithme stop. Be sure to set plot to False otherwise you would have a NotImplementedError 
        
        * reset : Boolean : default is True : if you want to reset value and restart the algo or if you to continue a previous computation

        * silent : Boolean :  default False, True value will stop tqdm and any message to go to stdout
        
        * plot : Boolean : default is True , if nb_iter = inf it will plot the error curve , if nb_iter is an integer it will plot the worst approximation of eim interpolation at each step ( be sure to have implemented a correct plotinng method : _plot for your datas

        
        Exemple :

        >>> Z = torch.zeros((5,5,1))
        >>> ev = eim_vectorial_optim(Z)

        >>> ev.reach_precision(epsilon=1e-1)

        >>> we have a 1-kolmogorov-width with a final error of 0.0, dimensionality reduction : 0.0
        '''

        self.epsilon = epsilon

        if reset:
            self._reset_()

        t0 = time.time()


        self.error= { "infinite":np.array([self._iteration_n(epsilon)])}
        #self.error["quadratic"] = np.array([self._quadratic_error()])

        if(not(silent)):
            pbar = tqdm(total = min(nb_iter,self.Q_tab.shape[0]) )

        if plot and not(nb_iter == float('inf'))  :
                self._plot()

        while self.error["infinite"][-1]>epsilon and len(self.error["infinite"])<nb_iter:
            t1= time.time()
            delta =t1-t0
            t0=t1

            n_k = len(self.error["infinite"])-1
            error = self.error["infinite"][-1]

            if(not(silent)):
                pbar.update(1)
                pbar.set_description((f"{error} error in {delta} s "))


            self.error["infinite"] = np.append(self.error["infinite"],self._iteration_n(epsilon))


            if plot and not(nb_iter == float('inf'))  :
                self._plot()



        n_k = len(self.error["infinite"])-1
        error = self.error["infinite"][-1]


        self.kol_width =  n_k
        self.reduction = 1 - self.kol_width/self.Z.shape[0]
        if(not(silent)):
            print(f"========================= \n we have a {n_k}-kolmogorov-width with a final error of {error}, dimensionality reduction : {self.reduction}  \n=========================")
            pbar.close()


        if plot:
            if nb_iter == float('inf'):
                self._plot_error(self.error)
            else:
                self._plot()



    def _plot_error(self,errors):

        for key in errors:
            plt.plot(range(1,len(errors[key])+1),errors[key],label=key)
            plt.plot(range(1,len(errors[key])+1),np.repeat(self.epsilon,len(errors[key])), label="goal" )

        plt.legend()
        plt.show()

    def _plot(self):
        raise NotImplementedError("according to your datas you should heritate this class and implement the plot function")



class eim(eim_vectorial):

    '''
    x_magics   : is a np array which represent the n+1-magic point of the iteration n

    mu_magics  : is a np array which represent the mu with the worst approximation by iterator a each iteration
    '''

    def __create_Z_mu(self,M,X,f):

        Z = np.zeros((M.shape[0],X.shape[0],1))

        for i in range(0,len(M)):
            Z[i,:,0]=f(X,M[i])

        return Z


    def __init__(self,f,x_values,M):
        self.f =f

        self.x_values = x_values
        self.M = M
        self.Z = self.__create_Z_mu(self.M,self.x_values,self.f)
        super().__init__(self.Z)

    def _reset_(self):
        self.__init__(self.f,self.x_values,self.M)

    def _plot(self):
        plt.plot(self.x_values,self.Z[self.mu_magics[-1]],label=f"f for worst mu={self.mu_magics[-1]}")
        plt.plot(self.x_values,self.I[self.mu_magics[-1]],label=f"Interpolation of f for worst mu={self.mu_magics[-1]}")
        plt.show()
        plt.legend()

class eim_prestored_data(eim_vectorial):

    def __init__(self,Z):
        '''
        Z shloud have a (M,N) where N is the number of point and M the number of mu
        '''
        self.Z = Z.reshape( Z.shape+tuple([1])) #expand one dim
        super().__init__(self.Z)

    def _reset_(self):
        self.__init__(self.Z)

    def _plot(self):
        raise NotImplementedError("according to your datas you should heritate this class and implement the plot function")
