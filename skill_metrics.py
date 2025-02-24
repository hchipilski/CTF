#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Packages
import numpy as np
from math_funcs import *

## Functions
def kullback_leibler_divergence(P,Q):
    """ 
    Calculates the Kullback-Leibler (KL) divergence of 
    distribution P from distribution Q. Note that KL is not 
    a symmetric measure, i.e. KL(P,Q) != KL(Q,P).

    Parameters
    ----------
    P : ndarray [x2_gridSize,x1_gridSize]
        First distribution in discretized form.
    Q : ndarray [x2_gridSize,x1_gridSize]
        Second distribution in discretized form.
    
    Returns
    -------
    kl_div : float
        Kullback-Leibler divergence.
    """
    eps = 1e-12 # make sure the PDFs != 0
    P2 = P+eps
    Q2 = Q+eps
    kl_div = np.sum( P2*np.log(P2/Q2) )
    return kl_div

def jensen_shannon_divergence(P,Q):
    """ 
    Calculates the Jensen-Shannon (JS) divergence of 
    distribution P from distribution Q. Unlike the KL
    divergence, the JS divergence is a symmetric measure, 
    i.e. JS(P,Q) = JS(Q,P).

    Parameters
    ----------
    P : ndarray [Nx,Nx]
        First distribution in discretized form.
    Q : ndarray [Nx,Nx]
        Second distribution in discretized form.
    
    Returns
    -------
    js_div : float
        Jensen-Shannon divergence.
    """
    M = 0.5*(P+Q)
    js_div = 0.5*kullback_leibler_divergence(P,M) + \
             0.5*kullback_leibler_divergence(Q,M)
    return js_div

def mean_errors(Xa,EXu,VarXu,xt):
    """ 
    Mean errors of the analysis ensemble with
    respect to the posterior expectation and standard 
    deviation (std) as well as the randomly drawn 
    true state.

    Parameters
    ----------
    Xa : ndarray [Nx,Ne]
        Posterior ensemble.
    EXu : ndarray [Nx]
        Posterior expectation.
    VarXu : ndarray [Nx]
        Posterior variance.
    xt : ndarray [Nx]
        Randomly drawn truth.
    
    Returns
    -------
    Xa_me_EXu : float
        Mean error of the posterior (analysis) ensemble mean 
        with respect to the posterior expectation.
    Xa_me_stdXu : float
        Mean error of the posterior (analysis) ensemble std 
        with respect to the posterior std (std=standard deviation).
    Xa_me_xt : float
        Mean error of the posterior (analysis) ensemble mean 
        with respect to the randomly drawn truth.
    """
    # error with respect to posterior mean
    dEXu = Xa.mean(axis=1)-EXu
    Xa_me_EXu = np.mean(dEXu)
    # error with respect to posterior std
    Xa1_std = np.sqrt(var_bessel(Xa[0,:])); Xu1_std = np.sqrt(VarXu[0])
    Xa2_std = np.sqrt(var_bessel(Xa[1,:])); Xu2_std = np.sqrt(VarXu[1])
    dXa_std = np.array([Xa1_std-Xu1_std,Xa2_std-Xu2_std])
    Xa_me_stdXu = np.mean(dXa_std)
    # error with respect to truth
    dxt = Xa.mean(axis=1)-xt
    Xa_me_xt = np.mean(dxt)
    return Xa_me_EXu,Xa_me_stdXu,Xa_me_xt

def num_mems_outside_bounds(Xa,exp_name):
    """ 
    Number of posterior ensemble members outside of physical bounds.

    Parameters
    ----------
    Xa : ndarray [Nx,Ne]
        Posterior ensemble.
    exp_name : string
        Experiment name (e.g., loglogitPrior_biasedOb)
    
    Returns
    -------
    Xa_num_outside : float
        Number of ensemble members outside of physical bounds.
    """
    
    if (exp_name == 'loglogitPrior_logOb') or \
       (exp_name == 'loglogitPrior_biasedOb') or \
       (exp_name == 'loglogitPrior_adaptiveOb'):
        x1_min,x1_max = 0.0,1e12 # 1e12 -> +inf
        x2_min,x2_max = 0.0,1.0
        idx_x1 = np.where((Xa[0,:] < x1_min)\
                        | (Xa[0,:] > x1_max))[0]
        idx_x2 = np.where((Xa[1,:] < x2_min)\
                        | (Xa[1,:] > x2_max))[0]
        Xa_num_outside = idx_x1.size+idx_x2.size
    return Xa_num_outside


def spread_rmse_ratio(Xa,xt):
    """ 
    Evaluate accuracy of posterior ensemble spread.

    Parameters
    ----------
    Xa : ndarray [Nx,Ne]
        Posterior ensemble.
    xt : ndarray [Nx]
        Truth.
    
    Returns
    -------
    Xa_spread : ndarray [Nx]
        Posterior ensemble spread for each component of the 
        state vector. Spread=sample variance here.
    Xa_rmse : ndarray [Nx]
        Mean-squared-error of posterior ensemble mean with 
        respect to the truth for each component of the 
        state vector.
    Xa_ratio : ndarray [Nx]
        Posterior ensemble consistency ratio 
        Xa_ratio := Xa_spread/Xa_rmse.
    """
    Ne = Xa.shape[1]
    Xa_ens_mean = Xa.mean(axis=1)
    Xa_ens_mean2 = np.tile(Xa_ens_mean,(Ne,1)).T
    Xa_prime = Xa-Xa_ens_mean2
    Xa_spread_tmp = (Xa_prime**2.0).sum(axis=1)/(Ne-1)
    Xa_spread = np.sqrt(Xa_spread_tmp.mean())
    Xa_rmse_tmp = (Xa_ens_mean-xt)**2.0
    Xa_rmse = np.sqrt(Xa_rmse_tmp.mean())
    Xa_ratio = Xa_spread/Xa_rmse
    return Xa_spread,Xa_rmse,Xa_ratio
