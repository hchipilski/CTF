#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Packages
import numpy as np
from scipy.stats import norm as univariate_normal
from math_funcs import *

## Functions
def ECTF_scalarOb(XfG,YfG,yG,H,Ne,exp_name):
    """ 
    Analysis update with a scalar observation according to the 
    stochastic Ensemble Conjugate Transform Filter (ECTF).

    Parameters
    ----------
    XfG : ndarray [Nx,Ne]
        Prior ensemble in the latent space.
    YfG : ndarray [Ny,Ne]
        Perturbed observation ensemble in the latent space.
    yG : float
        Scalar observation in the latent space.
    R : ndarray [1,1]
        Variance of the observation noise in the latent space.
    H : ndarray [1,Nx]
        Observation operator.
    Ne : integer
        Ensemble size.
    exp_name : string
        Name of experiment (e.g., loglogitPrior_biasedOb).
    
    Returns
    -------
    Xa : ndarray [Nx,Ne]
        Posterior ensemble.
    """
    
    ## Kalman gain in the latent space
    HXfG = H.dot(XfG)
    SHt = calc_norm_pert_ens(XfG).dot(calc_norm_pert_ens(HXfG).T)
    HSHt_R = calc_norm_pert_ens(YfG).dot(calc_norm_pert_ens(YfG).T)
    K = SHt.dot(inv_2by2(HSHt_R))

    ## Analysis ensemble in the latent space
    yG_arr = np.array([yG]*Ne); yG_arr.shape = (1,Ne)
    XaG = XfG + K.dot(yG_arr-YfG)
    
    ## Map to physical space
    if (exp_name == 'loglogitPrior_logOb') or \
       (exp_name == 'loglogitPrior_biasedOb') or \
       (exp_name == 'loglogitPrior_adaptiveOb'):
        Xa = np.array([np.exp(XaG[0,:]),logistic(XaG[1,:])])
    return Xa


def ECTF_scalarOb_old(XfG,yG,R,H,Ne,exp_name):
    """ 
    Analysis update with a scalar observation according to the 
    stochastic Ensemble Conjugate Transform Filter (ECTF). 

    The difference with the newer version 'ECTF_scalarOb' is that the 
    observation noise is explicitly simulated here. The newer version
    uses the perturbed observation ensemble in the latent space and 
    does not require the R matrix as input.

    Parameters
    ----------
    XfG : ndarray [Nx,Ne]
        Prior ensemble in the latent space.
    yG : float
        Scalar observation in the latent space.
    R : ndarray [1,1]
        Variance of the observation noise in the latent space.
    H : ndarray [1,Nx]
        Observation operator.
    Ne : integer
        Ensemble size.
    exp_name : string
        Name of experiment (e.g., loglogitPrior_biasedOb).
    
    Returns
    -------
    Xa : ndarray [Nx,Ne]
        Posterior ensemble.
    """
    
    ## Create an ensemble of observation errors
    distr_obErrG = univariate_normal(loc=0.0,scale=np.sqrt(R[0][0]))
    obErr_ensG = distr_obErrG.rvs(size=Ne); obErr_ensG.shape = (1,Ne)

    ## Kalman gain in the latent space
    XfG_mean = XfG.mean(axis=1)
    XfG_mean2 = np.tile(XfG_mean,(Ne,1)).T
    XfG_pert = (XfG-XfG_mean2)/np.sqrt(Ne-1.0)
    S_XfG = XfG_pert.dot(XfG_pert.T)
    SHt_XfG = S_XfG.dot(H.T)
    HSHt_XfG = np.dot(H.dot(S_XfG),H.T)
    K_XfG = SHt_XfG.dot(inv_2by2(R+HSHt_XfG))

    ## Analysis ensemble in the latent space
    yG_arr = np.array([yG]*Ne); yG_arr.shape = (1,Ne)
    XaG = XfG + K_XfG.dot(yG_arr+obErr_ensG-H.dot(XfG))
    
    ## Map to physical space
    if (exp_name == 'loglogitPrior_logOb') or \
       (exp_name == 'loglogitPrior_biasedOb') or \
       (exp_name == 'loglogitPrior_adaptiveOb'):
        Xa = np.array([np.exp(XaG[0,:]),logistic(XaG[1,:])])
    return Xa

def calc_norm_pert_ens(W):
    """  Calculate a normalized perturbed ensemble. 
    W [N,Ne] : Ensemble matrix (Ne is the ensemble size).
    """
    W_mean = W.mean(axis=1)
    W_mean2 = np.tile(W_mean,(W.shape[1],1)).T
    W_pert_norm = (W-W_mean2)/np.sqrt(W.shape[1]-1.0)
    return W_pert_norm
