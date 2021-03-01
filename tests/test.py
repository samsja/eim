
from eim import Eim


import matplotlib.pyplot as plt
import numpy as np
import torch


def f(x,mu):
    return np.array([np.cos(mu*x),np.sin(x/mu),np.exp(x/mu)]).T
N=200
x_values = np.linspace(1,10,num=N)
Nmu=300
M = np.linspace(1,10,Nmu)

Z = np.zeros((M.shape[0],x_values.shape[0],3))

for i in range(M.shape[0]):
    Z[i] = np.array(f(x_values,M[i]))



ev_new = Eim(Z,from_numpy=True)

ev_new.reach_precision(epsilon=1e-1,plot=False,silent=True)
#ev_new.save_model("test.model")
ev_new.compress_model()

ev_test = Eim(None,load=True)
ev_test.load_model("tests/test.model")



def test_q_tab():

    assert  (ev_test.Q_tab == ev_new.Q_tab).all() 

def test_x_magics():

    assert  (ev_test.x_magics == ev_new.x_magics).all() 


def test_j_magics():

    assert  (ev_test.j_magics == ev_new.j_magics).all() 

def test_mu_magics():

    assert  (ev_test.mu_magics == ev_new.mu_magics).all() 


def test_alphas():
    
    assert  (ev_test.compute_alpha(ev_test.m,ev_new.Z) == ev_new.compute_alpha(ev_new.m,ev_new.Z)).all() 


def test_interpolation():
    
    alphas = ev_new.compute_alpha(ev_new.m,ev_new.Z)
    
    assert  (ev_test.project_with_alpha(alphas)== ev_new.project_with_alpha(alphas)).all()


