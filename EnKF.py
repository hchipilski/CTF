#!/usr/bin/env python
# -*- coding: utf-8 -*-

"Abstract: Solvers for the Ensemble Kalman Filter (EnKF)."

## Packages
import numpy as np
from scipy.stats import norm as univariate_normal
from math_funcs import *


## Functions
def EnKF_scalarOb(Xf,Yf,y,H,Ne):
    """ 
    Analysis update with a scalar observation according to the 
    stochastic Ensemble Kalman Filter (EnKF).

    Parameters
    ----------
    Xf : ndarray [Nx,Ne]
        Prior ensemble.
    Yf : ndarray [Ny,Ne]
        Pertubed observation ensemble.
    y : float
        Observation.
    H : ndarray [1,Nx]
        Observation operator.
    Ne : integer
        Ensemble size.
    
    Returns
    -------
    Xa : ndarray [Nx,Ne]
        Posterior ensemble.
    """
    
    ## Kalman gain
    HXf = H.dot(Xf)
    SHt = calc_norm_pert_ens(Xf).dot(calc_norm_pert_ens(HXf).T)
    HSHt_R = calc_norm_pert_ens(Yf).dot(calc_norm_pert_ens(Yf).T)
    K = SHt.dot(inv_2by2(HSHt_R))

    ## Apply the stochastic EnKF update
    y_arr = np.array([y]*Ne); y_arr.shape = (1,Ne)
    Xa = Xf + K.dot(y_arr-Yf)    
    return Xa


def EnKF_scalarOb_explicitR(Xf,Y,y,H,Ne):
    """ 
    Analysis update with a scalar observation according to the 
    stochastic Ensemble Kalman Filter (EnKF). Unlike EnKF_scalarOb,
    R matrix here is explicitly calculated from the likelihood ensemble.

    Parameters
    ----------
    Xf : ndarray [Nx,Ne]
        Prior ensemble.
    Y : ndarray [Ny,Ne]
        Likelihood sample.
    y : float
        Observation.
    H : ndarray [1,Nx]
        Observation operator.
    Ne : integer
        Ensemble size.
    
    Returns
    -------
    Xa : ndarray [Nx,Ne]
        Posterior ensemble.
    """
    
    ## Kalman gain
    HXf = H.dot(Xf)
    SHt = calc_norm_pert_ens(Xf).dot(calc_norm_pert_ens(HXf).T)
    HSHt = calc_norm_pert_ens(HXf).dot(calc_norm_pert_ens(HXf).T)
    R = np.array([[var_bessel(Y)]])
    K = SHt.dot(inv_2by2(HSHt+R))

    ## Sample from the observation noise with covariance R (estimated above)
    distr_obErr = univariate_normal(loc=0.0,scale=np.sqrt(R[0][0]))
    obErr_ens = distr_obErr.rvs(size=Ne); obErr_ens.shape = (1,Ne)
    
    ## Apply the stochastic EnKF update
    y_arr = np.array([y]*Ne); y_arr.shape = (1,Ne)
    Xa = Xf + K.dot(y_arr-HXf-obErr_ens)    
    return Xa

def calc_norm_pert_ens(W):
    """  
    Calculate a normalized perturbed ensemble. 
    W [N,Ne] : Ensemble matrix (Ne is the ensemble size).
    """
    W_mean = W.mean(axis=1)
    W_mean2 = np.tile(W_mean,(W.shape[1],1)).T
    W_pert_norm = (W-W_mean2)/np.sqrt(W.shape[1]-1.0)
    return W_pert_norm
