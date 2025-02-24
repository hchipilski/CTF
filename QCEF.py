#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Packages
import numpy as np
from distributions import *

## Functions
def QCEF_scalarOb(Xf,MUp,Sp,MUu,Su,H,Ne,exp_name):
    """ Analysis update with a scalar observation according to the 
    Quantile Conserving Ensemble Filter (QCEF) of Anderson (2022).

    Parameters
    ----------
    Xf : ndarray [Nx,Ne]
        Prior ensemble.
    MUp : ndarray [Nx,1]
        Prior Mu parameter.
    Sp : ndarray [Nx,Nx]
        Prior Sigma parameter.
    MUu : ndarray [Nx,1]
        Posterior Mu parameter.
    Su : ndarray [Nx,Nx]
        Posterior Sigma parameter.
    H : ndarray [1,Nx]
        Observation operator.
    Ne : int
        Ensemble size.
    exp_name : string
        Name of experiment (e.g., loglogitPrior_biasedOb).
    
    Returns
    -------
    Xa : ndarray [Nx,Ne]
        Posterior ensemble.
    """

    ## Analysis in observation space
    if (exp_name == 'loglogitPrior_logOb') or \
       (exp_name == 'loglogitPrior_biasedOb') or \
       (exp_name == 'loglogitPrior_adaptiveOb'):
        Yf = H.dot(Xf).ravel()
        Ya = univariate_ppf(univariate_cdf(Yf,\
                            MUp[0,0],np.sqrt(Sp[0,0]),\
                            'lognormal'),\
                            MUu[0,0],np.sqrt(Su[0,0]),\
                            'lognormal')
    
    ## Analysis increments in observation space
    dY = Ya-Yf

    ## Ensemble covariance between observed and unobserved state
    Yf_mean = Yf.mean()
    Yf_prime = Yf-Yf_mean
    Xf2 = Xf[1,:]
    Xf2_mean = Xf2.mean()
    Xf2_prime = Xf2-Xf2_mean
    prod_XY_primes = Yf_prime*Xf2_prime
    covar_XY = prod_XY_primes.sum()/(Ne-1.0)

    ## Ensemble variance of observed state
    prod_YY_primes = Yf_prime*Yf_prime
    var_Y = prod_YY_primes.sum()/(Ne-1.0)
    
    ## Transfer analysis increment to unobserved state variables
    Xa2 = Xf2 + covar_XY/var_Y*dY

    ## Form QCEF analysis vector
    Xa = np.array([Ya,Xa2])
    return Xa
