#!/usr/bin/env python
# -*- coding: utf-8 -*-

"Abstract: A collection of useful mathematical functions."

## Packages
import numpy as np


## Functions
def var_bessel(x):
    """ 
    Unbiased (Bessel) variance estimator.
    
    Parameters
    ----------
    x : ndarray
        Sample.
    
    Returns
    -------
    var_unbiased : float
        Unbiased (Bessel) variance.
    """
    N = len(x) # sample size
    mean = x.mean()
    arr_1 = x-mean
    arr_2 = arr_1**2.0
    sum_ = np.sum(arr_2)
    var_unbiased = sum_/(N-1.0)
    return var_unbiased

def det_2by2(A):
    """ 
    Determinant of a 2x2 matrix A.
    
    Parameters
    ----------
    A : [2,2] ndarray
        Matrix whose determinant is to be found.
    
    Returns
    -------
    det_A : float
        Determinant of A.
    """
    det_A = A[0,0]*A[1,1]-A[1,0]*A[0,1] 
    return det_A 

def inv_2by2(A):
    """ 
    Inverse of a 1x1 (scalar) or 2x2 matrix A.
    
    Parameters
    ----------
    A : float or [2,2] ndarray
        Matrix to be inverted.
    
    Returns
    -------
    A_inv : float or [2,2] ndarray
        Inverse of A.
    """
    if A.size>1:
        A_inv = 1.0/det_2by2(A)*np.array([[A[1,1],-A[0,1]],[-A[1,0],A[0,0]]])
        return A_inv
    else:
        A_inv = 1.0/A[0][0]
        return A_inv

def logistic(x):
    """ 
    Standard logistic function.
    
    Parameters
    ----------
    x : float or ndarray
        Input.
    
    Returns
    -------
    out : float or ndarray
        Function output.
    """
    out = 1.0/(1.0+np.exp(-x))
    return out

def inv_logistic(x):
    """ 
    Inverse of the standard logistic function, 
    i.e. the logit function.
    
    Parameters
    ----------
    x : float or ndarray
        Input.
    
    Returns
    -------
    out : float or ndarray
        Function output.
    """
    out = np.log(x / (1.0 - x))
    return out
