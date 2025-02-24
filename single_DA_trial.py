#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Single trial DA experiments with EnKF, QCEF and ECTF.
"""

## Packages
import os
import sys
current_dir = os.getcwd()
main_dir = os.path.dirname(current_dir)
sys.path.append(main_dir)
import numpy as np
from scipy.integrate import simps
from scipy.stats import norm as univariate_normal
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import time
from math_funcs import *
from distributions import *
from skill_metrics import *
from EnKF import *
from QCEF import *
from ECTF import *
from plotting_tools import *


def main():

    start_time_total = time.ctime()
    print('\n')
    print('PROGRAM STARTED: {}'.format(start_time_total))
    print('\n')

    ## User input
    seed_num = 25 # gives a good illustration of filtering differences
    print('\n=== DA TRIAL WITH seed_num={} ==='.format(seed_num))
    np.random.seed(seed=seed_num)
    start_time_trial = time.time()

    ## Settings
    yG = np.log(0.5)
    RHOp = 0.99 # correlation coefficient for latent prior PDF
    R = np.array([[0.05]]) # covariance parameter of latent observation noise
    Nx,Ny = 2,1 # state and observation dimensions
    Ne = int(1e6) # ensemble size; try 100, 1000 or 1e6
    x1_min,x2_min = 1e-15,1e-15 # min values for 2D grid (needed in PDF calculations)
    x1_max,x2_max = 100.0,1.0-1e-15 # max values for 2D grid
    x1_gridSize,x2_gridSize = 10000,500 # number of grid points discretizing x1 and x2 vars
    ymin = 1.0 # min observation threshold (for the 'loglogitPrior_biasedOb' experiment)
    H = np.array([1,0]); H.shape = (Ny,Nx) # observation operator
    MUp_low,MUp_high = -1.0,1.0 # prior MU parameter range to be uniformly sampled from 
    Sp_low,Sp_high = 0.05,2.0 # prior Sigma parameter range to be uniformly sampled from
    exp_name = 'loglogitPrior_logOb' # select loglogitPrior_logOb, loglogitPrior_biasedOb,loglogitPrior_adaptiveOb
    outdir_results = '/glade/derecho/scratch/hristoc/statistical_experiments/y_variations' # where filtering results are written
    outdir_figures = '{}/figures'.format(current_dir) # where figures go
    do_plotting = 1

    ## Check if outdirs exist; if not, create them
    if not os.path.exists(outdir_results):
        os.makedirs(outdir_results)
    if not os.path.exists(outdir_figures):
        os.makedirs(outdir_figures)

    ## Discretize 2D state domain based on its distribution
    if (exp_name == 'loglogitPrior_logOb') or \
       (exp_name == 'loglogitPrior_biasedOb') or \
       (exp_name == 'loglogitPrior_adaptiveOb'):
        x1_arr = np.linspace(x1_min,x1_max,x1_gridSize)
        x2_arr = np.linspace(x2_min,x2_max,x2_gridSize)
    else:
        print("FATAL ERROR: Invalid experiment entered. Exiting ...")
        exit()


    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #                 Data generation
    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=  

    ## Define prior PDF by randomly sampling its parameters
    MUp_x1 = np.random.uniform(low=MUp_low,high=MUp_high)
    MUp_x2 = np.random.uniform(low=MUp_low,high=MUp_high)
    MUp = np.array([MUp_x1,MUp_x2]); MUp.shape = (Nx,1)
    Sp_x1 = np.random.uniform(low=Sp_low,high=Sp_high)
    Sp_x2 = np.random.uniform(low=Sp_low,high=Sp_high)
    Sp = np.array([[Sp_x1,RHOp*np.sqrt(Sp_x1)*np.sqrt(Sp_x2)],\
                   [RHOp*np.sqrt(Sp_x1)*np.sqrt(Sp_x2),Sp_x2]])

    ## Randomly generate prior ensemble, true state and observations
    distr_pG = multivariate_normal(mean=MUp.ravel(),cov=Sp) # prior distribution in the latent space
    XfG = distr_pG.rvs(size=Ne).T # prior ensemble in the latent space
    xtG = distr_pG.rvs(size=1) # truth in the latent space
    distr_obErrG = univariate_normal(loc=0.0,scale=np.sqrt(R[0][0]))
    YfG = H.dot(XfG)+distr_obErrG.rvs(size=Ne) # perturbed observation ensemble in the latent space
    YG = H.dot(xtG)+distr_obErrG.rvs(size=Ne) # likelihood sample in the latent space
    if exp_name == 'loglogitPrior_logOb':
        Xf = np.array([np.exp(XfG[0,:]),logistic(XfG[1,:])])
        xt = np.array([np.exp(xtG[0]),logistic(xtG[1])])
        y = np.exp(yG)
        Yf = np.exp(YfG)
        Y = np.exp(YG)
    if exp_name == 'loglogitPrior_biasedOb':
        Xf = np.array([np.exp(XfG[0,:]),logistic(XfG[1,:])])
        xt = np.array([np.exp(xtG[0]),logistic(xtG[1])])
        y = np.exp(yG)+ymin
        Yf = np.exp(YfG)+ymin
        Y = np.exp(YG)+ymin
    if exp_name == 'loglogitPrior_adaptiveOb':
        Xf = np.array([np.exp(XfG[0,:]),logistic(XfG[1,:])])
        xt = np.array([np.exp(xtG[0]),logistic(xtG[1])])
        if yG <= 0.0: # lognormal likelihood
            y = np.exp(yG)
            Yf = np.exp(YfG)
            Y = np.exp(YG)
        else: # Gaussian likelihood
            y = yG
            Yf = YfG
            Y = YG

    ## Prior mean in x1 and x2
    Xf_mean_x1 = Xf.mean(axis=1)[0]
    Xf_mean_x2 = Xf.mean(axis=1)[1]


    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #                 PDF calculations
    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=  
        
    ## Bayesian update of prior PDF parameters
    MUu,Su = bayesian_update_2D(MUp,Sp,H,yG,R)
    MUu_x1,MUu_x2 = MUu[0,0],MUu[1,0]
    Su_x1,Su_x2 = Su[0,0],Su[1,1]

    ## Create a meshgrid for (x1,x2) values
    x1_grid,x2_grid = np.meshgrid(x1_arr,x2_arr)
    '''
    * Dimensions of x1_grid, x2_grid: [x2_gridSize,x1_gridSize]; [100,3000] is the default.
    * Values in x1_grid increase by increasing the 2nd index keeping the 1st index fixed.
    * Values in x2_grid increase by increasing the 1st index keeping the 2nd index fixed.
    '''    

    ## Evaluate PDFs for this problem
    PDFp = loglogit_normal_PDF(x1_grid,x2_grid,MUp,Sp) # dim=[x2_gridSize,x1_gridSize]
    PDFu_theory = loglogit_normal_PDF(x1_grid,x2_grid,MUu,Su) # same dim as PDFp
    if exp_name == 'loglogitPrior_logOb':
        likelihood = likelihood_logScalarOb_LLprior(y,R[0][0],H,x1_grid,x2_grid) # same dim as PDFp
    elif exp_name == 'loglogitPrior_biasedOb':
        likelihood = likelihood_biasedScalarOb_LLprior(y,ymin,R[0][0],H,x1_grid,x2_grid) # -||-
    elif exp_name == 'loglogitPrior_adaptiveOb':
        likelihood = likelihood_adaptiveScalarOb_LLprior(y,yG,R[0][0],H,x1_grid,x2_grid) # -||-
    prod_PDF = PDFp * likelihood
    prod_EX1u = x1_grid * prod_PDF
    prod_EX2u = x2_grid * prod_PDF
    prod_EX1u2 = (x1_grid ** 2.0) * prod_PDF
    prod_EX2u2 = (x2_grid ** 2.0) * prod_PDF
    # explicitly computed posterior PDF
    PDFu_explicit = prod_PDF/simps([simps(k,x1_arr) for k in prod_PDF],x2_arr)
    # expected value of posterior X
    EX1u = simps([simps(k,x1_arr) for k in prod_EX1u],x2_arr)/\
           simps([simps(k,x1_arr) for k in prod_PDF],x2_arr)
    EX2u = simps([simps(k,x1_arr) for k in prod_EX2u],x2_arr)/\
           simps([simps(k,x1_arr) for k in prod_PDF],x2_arr)
    EXu = np.array([EX1u,EX2u])
    # expected value of posterior X^2
    EX1u2 = simps([simps(k,x1_arr) for k in prod_EX1u2],x2_arr)/\
            simps([simps(k,x1_arr) for k in prod_PDF],x2_arr)
    EX2u2 = simps([simps(k,x1_arr) for k in prod_EX2u2],x2_arr)/\
            simps([simps(k,x1_arr) for k in prod_PDF],x2_arr)
    EXu2 = np.array([EX1u2,EX2u2])
    # variance of posterior X: Var[X] = E[X^2] - E[X]^2
    VarXu = EXu2-EXu**2.0

    ## Marginal PDFs
    PDFp_x1 = np.empty_like(x1_arr)
    PDFp_x2 = np.empty_like(x2_arr)
    PDFu_x1 = np.empty_like(x1_arr)
    PDFu_x2 = np.empty_like(x2_arr)
    for i,val in enumerate(x1_arr):
        PDFp_x1[i] = lognormal_scalar_PDF(val,MUp[0,0],np.sqrt(Sp[0,0]))
        PDFu_x1[i] = lognormal_scalar_PDF(val,MUu[0,0],np.sqrt(Su[0,0]))
    for i,val in enumerate(x2_arr):
        PDFp_x2[i] = logit_normal_scalar_PDF(val,MUp[1,0],np.sqrt(Sp[1,1]))
        PDFu_x2[i] = logit_normal_scalar_PDF(val,MUu[1,0],np.sqrt(Su[1,1]))
            
    ## Print some diagnostics
    print('\n* Parameters for this DA trial')
    print('RHOp: ', RHOp)
    print('R: ',R[0][0])
    print('Observation of X1 (latent and physical): ',y,yG)
    print('MUp: ',MUp[0,0],MUp[1,0])
    print('Sp: ',Sp[0,0],Sp[1,1])
    print('Prior ensemble mean: ',Xf.mean(axis=1))
    print('Truth: ',xt)
    #
    PDFp_int2 = simps([simps(k,x1_arr) for k in PDFp],x2_arr)
    PDFu_theory_int2 = simps([simps(k,x1_arr) for k in PDFu_theory],x2_arr)
    PDFu_explicit_int2 = simps([simps(k,x1_arr) for k in PDFu_explicit],x2_arr) 
    maxDiff_PDFu = abs(PDFu_theory-PDFu_explicit).max()
    print('\n* PDF diagnostics')
    print('Prior PDF integrates to',PDFp_int2)
    print('Theoretical posterior PDF integrates to',PDFu_theory_int2)
    print('Explicit posterior PDF integrates to',PDFu_explicit_int2)
    print('Max difference between theoretical and explicit posterior PDFs is',maxDiff_PDFu)
    print('Prior p(x1) integrates to',simps(PDFp_x1,x=x1_arr,even='avg'))
    print('Posterior p(x1) integrates to',simps(PDFu_x1,x=x1_arr,even='avg'))
    print('Prior p(x2) integrates to',simps(PDFp_x2,x=x2_arr,even='avg'))
    print('Posterior p(x2) integrates to',simps(PDFu_x2,x=x2_arr,even='avg'))


    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    #             Ensemble filtering results
    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=  

    ## Define 2D ensemble histogram edges to be later used for diagnostics and plotting
    offset_x1 = (x1_arr[1]-x1_arr[0])/2.0
    offset_x2 = (x2_arr[1]-x2_arr[0])/2.0
    edges_x1_tmp = x1_arr-offset_x1
    edges_x1 = np.append(edges_x1_tmp, x1_arr[-1]+offset_x1)
    edges_x2_tmp = x2_arr-offset_x2
    edges_x2 = np.append(edges_x2_tmp, x2_arr[-1]+offset_x2)


    ########## EnKF ##########
    Xa_EnKF = EnKF_scalarOb(Xf,Yf,y,H,Ne)
    Xa_density2d_tmp,xedges,yedges = \
            np.histogram2d(Xa_EnKF[0,:],Xa_EnKF[1,:],\
            bins=(edges_x1,edges_x2),density=True)
    Xa_density2d_EnKF = Xa_density2d_tmp.T
    Xa_JSdiv_EnKF = jensen_shannon_divergence(Xa_density2d_EnKF,PDFu_explicit)
    Xa_me_EX_EnKF,Xa_me_stdX_EnKF,Xa_me_xt_EnKF = mean_errors(Xa_EnKF,EXu,VarXu,xt)
    Xa_num_outside_EnKF = num_mems_outside_bounds(Xa_EnKF,exp_name)
    Xa_perc_outside_EnKF = 100.0*Xa_num_outside_EnKF/Ne
    Xa_spread_EnKF,Xa_rmse_EnKF,Xa_ratio_EnKF = spread_rmse_ratio(Xa_EnKF,xt)
    Xa_KLdiv_EnKF = kullback_leibler_divergence(Xa_density2d_EnKF,PDFu_explicit)

    # print diagnostics
    print('\n* EnKF')
    print('JS divergence = ', Xa_JSdiv_EnKF)
    print('ME wrt posterior expectation = ', Xa_me_EX_EnKF)
    print('ME wrt posterior std = ', Xa_me_stdX_EnKF)
    print('Percent analysis members outside of bounds = ', Xa_perc_outside_EnKF)
    print('ME wrt truth = ', Xa_me_xt_EnKF)
    print('Spread = ', Xa_spread_EnKF)
    print('RMSE = ', Xa_rmse_EnKF)
    print('Consistency ratio = ', Xa_ratio_EnKF)
    print('KL divergence = ', Xa_KLdiv_EnKF)

    # write results to file
    output_EnKF_tmp = [seed_num,RHOp,R[0][0],y,Xf_mean_x1,Xf_mean_x2,\
                       Xa_JSdiv_EnKF,Xa_me_EX_EnKF,Xa_me_stdX_EnKF,Xa_perc_outside_EnKF,\
                       MUp_x1,MUp_x2,Sp_x1,Sp_x2,MUu_x1,MUu_x2,Su_x1,Su_x2,\
                       PDFp_int2,PDFu_theory_int2,PDFu_explicit_int2,maxDiff_PDFu]
    output_EnKF = format_output(output_EnKF_tmp)
    fileout_EnKF = '{}/EnKF_RHOp={}_R={}_y={:.1f}.txt'.\
                    format(outdir_results,RHOp,R[0][0],y)
    mode = 'w' if not os.path.exists(fileout_EnKF) else 'a'
    print('Writing output ...')
    with open(fileout_EnKF,mode) as ifile:
        ifile.write(' '.join(output_EnKF)+'\n')

    # optionally plot
    if do_plotting == 1:
        print('Plotting results ...')
        plot_multivariate_update(Xa_EnKF,Xa_density2d_EnKF,Xa_JSdiv_EnKF,Xa_me_EX_EnKF,\
                                 Xa_me_stdX_EnKF,Xa_num_outside_EnKF,\
                                 Xf,Ne,xt,y,EX1u,EX2u,PDFp,likelihood,PDFu_explicit,\
                                 x1_arr,x2_arr,edges_x1,edges_x2,'EnKF',RHOp,R[0][0],\
                                 outdir_figures,seed_num,PDFu_x1,PDFu_x2)


    ########## EnKF (explicit R calculation) ##########
#    Xa_EnKF_R = EnKF_scalarOb_explicitR(Xf,Y,y,H,Ne)
#    Xa_density2d_tmp,xedges,yedges = \
#            np.histogram2d(Xa_EnKF_R[0,:],Xa_EnKF_R[1,:],\
#            bins=(edges_x1,edges_x2),density=True)
#    Xa_density2d_EnKF_R = Xa_density2d_tmp.T
#    Xa_JSdiv_EnKF_R = jensen_shannon_divergence(Xa_density2d_EnKF_R,PDFu_explicit)
#    Xa_me_EX_EnKF_R,Xa_me_stdX_EnKF_R,Xa_me_xt_EnKF_R = mean_errors(Xa_EnKF_R,EXu,VarXu,xt)
#    Xa_num_outside_EnKF_R = num_mems_outside_bounds(Xa_EnKF_R,exp_name)
#    Xa_perc_outside_EnKF_R = 100.0*Xa_num_outside_EnKF_R/Ne
#    Xa_spread_EnKF_R,Xa_rmse_EnKF_R,Xa_ratio_EnKF_R = spread_rmse_ratio(Xa_EnKF_R,xt)
#    Xa_KLdiv_EnKF_R = kullback_leibler_divergence(Xa_density2d_EnKF_R,PDFu_explicit)
#    print('\n* EnKF (explicit R)')
#    print('JS divergence = ',Xa_JSdiv_EnKF_R)
#    print('ME wrt posterior expectation = ',Xa_me_EX_EnKF_R)
#    print('ME wrt posterior std = ',Xa_me_stdX_EnKF_R)
#    print('Percent analysis members outside of bounds = ',Xa_perc_outside_EnKF_R)
#    print('ME wrt truth = ', Xa_me_xt_EnKF_R)
#    print('Spread = ', Xa_spread_EnKF_R)
#    print('RMSE = ', Xa_rmse_EnKF_R)
#    print('Consistency ratio = ', Xa_ratio_EnKF_R)
#    print('KL divergence = ', Xa_KLdiv_EnKF_R)
#
#    # write results to file
#    output_EnKF_R_tmp = [seed_num,RHOp,R[0][0],y,Xf_mean_x1,Xf_mean_x2,\
#                         Xa_JSdiv_EnKF_R,Xa_me_EX_EnKF_R,Xa_me_stdX_EnKF_R,Xa_perc_outside_EnKF_R,\
#                         MUp_x1,MUp_x2,Sp_x1,Sp_x2,MUu_x1,MUu_x2,Su_x1,Su_x2,\
#                         PDFp_int2,PDFu_theory_int2,PDFu_explicit_int2,maxDiff_PDFu]
#    output_EnKF_R = format_output(output_EnKF_R_tmp)
#    fileout_EnKF_R = '{}/EnKF_R_RHOp={}_R={}_y={:.1f}.txt'.\
#                      format(outdir_results,RHOp,R[0][0],y)
#    mode = 'w' if not os.path.exists(fileout_EnKF_R) else 'a'
#    print('Writing output ...')
#    with open(fileout_EnKF_R,mode) as ifile:
#        ifile.write(' '.join(output_EnKF_R)+'\n')
#
#    # optionally plot
#    if do_plotting == 1:
#        print('Plotting results ...')
#        plot_multivariate_update(Xa_EnKF_R,Xa_density2d_EnKF_R,Xa_JSdiv_EnKF_R,Xa_me_EX_EnKF_R,\
#                                 Xa_me_stdX_EnKF_R,Xa_num_outside_EnKF_R,\
#                                 Xf,Ne,xt,y,EX1u,EX2u,PDFp,likelihood,PDFu_explicit,\
#                                 x1_arr,x2_arr,edges_x1,edges_x2,'EnKF_R',RHOp,R[0][0],\
#                                 outdir_figures,seed_num,PDFu_x1,PDFu_x2)


    ########## QCEF ##########
    Xa_QCEF = QCEF_scalarOb(Xf,MUp,Sp,MUu,Su,H,Ne,exp_name)
    Xa_density2d_tmp,xedges,yedges = \
        np.histogram2d(Xa_QCEF[0,:],Xa_QCEF[1,:],\
        bins=(edges_x1,edges_x2),density=True)
    Xa_density2d_QCEF = Xa_density2d_tmp.T
    Xa_JSdiv_QCEF = jensen_shannon_divergence(Xa_density2d_QCEF,PDFu_explicit)
    Xa_me_EX_QCEF,Xa_me_stdX_QCEF,Xa_me_xt_QCEF = mean_errors(Xa_QCEF,EXu,VarXu,xt)
    Xa_num_outside_QCEF = num_mems_outside_bounds(Xa_QCEF,exp_name)
    Xa_perc_outside_QCEF = 100.0*Xa_num_outside_QCEF/Ne
    Xa_spread_QCEF,Xa_rmse_QCEF,Xa_ratio_QCEF = spread_rmse_ratio(Xa_QCEF,xt)
    Xa_KLdiv_QCEF = kullback_leibler_divergence(Xa_density2d_QCEF,PDFu_explicit)
    print('\n* QCEF')
    print('JS divergence = ', Xa_JSdiv_QCEF)
    print('ME wrt posterior expectation = ', Xa_me_EX_QCEF)
    print('ME wrt posterior std = ', Xa_me_stdX_QCEF)
    print('Percent analysis members outside of bounds = ', Xa_perc_outside_QCEF)
    print('ME wrt truth = ', Xa_me_xt_QCEF)
    print('Spread = ', Xa_spread_QCEF)
    print('RMSE = ', Xa_rmse_QCEF)
    print('Consistency ratio = ', Xa_ratio_QCEF)
    print('KL divergence = ', Xa_KLdiv_QCEF)

    # write results to file
    output_QCEF_tmp = [seed_num,RHOp,R[0][0],y,Xf_mean_x1,Xf_mean_x2,\
                       Xa_JSdiv_QCEF,Xa_me_EX_QCEF,Xa_me_stdX_QCEF,Xa_perc_outside_QCEF,\
                       MUp_x1,MUp_x2,Sp_x1,Sp_x2,MUu_x1,MUu_x2,Su_x1,Su_x2,\
                       PDFp_int2,PDFu_theory_int2,PDFu_explicit_int2,maxDiff_PDFu]
    output_QCEF = format_output(output_QCEF_tmp)
    fileout_QCEF = '{}/QCEF_RHOp={}_R={}_y={:.1f}.txt'.\
                   format(outdir_results,RHOp,R[0][0],y)
    mode = 'w' if not os.path.exists(fileout_QCEF) else 'a'
    print('Writing output ...')
    with open(fileout_QCEF,mode) as ifile:
        ifile.write(' '.join(output_QCEF)+'\n')

    # optionally plot
    if do_plotting == 1:
        print('Plotting results ...')
        plot_multivariate_update(Xa_QCEF,Xa_density2d_QCEF,Xa_JSdiv_QCEF,Xa_me_EX_QCEF,\
                                 Xa_me_stdX_QCEF,Xa_num_outside_QCEF,\
                                 Xf,Ne,xt,y,EX1u,EX2u,PDFp,likelihood,PDFu_explicit,\
                                 x1_arr,x2_arr,edges_x1,edges_x2,'QCEF-LR',RHOp,R[0][0],\
                                 outdir_figures,seed_num,PDFu_x1,PDFu_x2)


    ########## ECTF ##########
    Xa_ECTF = ECTF_scalarOb(XfG,YfG,yG,H,Ne,exp_name)
    Xa_density2d_tmp,xedges,yedges = \
            np.histogram2d(Xa_ECTF[0,:],Xa_ECTF[1,:],\
            bins=(edges_x1,edges_x2),density=True)
    Xa_density2d_ECTF = Xa_density2d_tmp.T
    Xa_JSdiv_ECTF = jensen_shannon_divergence(Xa_density2d_ECTF,PDFu_explicit)
    Xa_me_EX_ECTF,Xa_me_stdX_ECTF,Xa_me_xt_ECTF = mean_errors(Xa_ECTF,EXu,VarXu,xt)
    Xa_num_outside_ECTF = num_mems_outside_bounds(Xa_ECTF,exp_name)
    Xa_perc_outside_ECTF = 100.0*Xa_num_outside_ECTF/Ne
    Xa_spread_ECTF,Xa_rmse_ECTF,Xa_ratio_ECTF = spread_rmse_ratio(Xa_ECTF,xt)
    Xa_KLdiv_ECTF = kullback_leibler_divergence(Xa_density2d_ECTF,PDFu_explicit)
    print('\n* ECTF')
    print('JS divergence = ', Xa_JSdiv_ECTF)
    print('ME wrt posterior expectation = ', Xa_me_EX_ECTF)
    print('ME wrt posterior std = ', Xa_me_stdX_ECTF)
    print('Percent analysis members outside of bounds = ', Xa_perc_outside_ECTF)
    print('ME wrt truth = ', Xa_me_xt_ECTF)
    print('Spread = ', Xa_spread_ECTF)
    print('RMSE = ', Xa_rmse_ECTF)
    print('Consistency ratio = ', Xa_ratio_ECTF)
    print('KL divergence = ', Xa_KLdiv_ECTF)

    # write results to file
    output_ECTF_tmp = [seed_num,RHOp,R[0][0],y,Xf_mean_x1,Xf_mean_x2,\
                       Xa_JSdiv_ECTF,Xa_me_EX_ECTF,Xa_me_stdX_ECTF,Xa_perc_outside_ECTF,\
                       MUp_x1,MUp_x2,Sp_x1,Sp_x2,MUu_x1,MUu_x2,Su_x1,Su_x2,\
                       PDFp_int2,PDFu_theory_int2,PDFu_explicit_int2,maxDiff_PDFu]
    output_ECTF = format_output(output_ECTF_tmp)
    fileout_ECTF = '{}/ECTF_RHOp={}_R={}_y={:.1f}.txt'.\
                   format(outdir_results,RHOp,R[0][0],y)
    mode = 'w' if not os.path.exists(fileout_ECTF) else 'a'
    print('Writing output ...')
    with open(fileout_ECTF,mode) as ifile:
        ifile.write(' '.join(output_ECTF)+'\n')

    # optionally plot
    if do_plotting == 1:
        print('Plotting results ...')
        plot_multivariate_update(Xa_ECTF,Xa_density2d_ECTF,Xa_JSdiv_ECTF,Xa_me_EX_ECTF,\
                                 Xa_me_stdX_ECTF,Xa_num_outside_ECTF,\
                                 Xf,Ne,xt,y,EX1u,EX2u,PDFp,likelihood,PDFu_explicit,\
                                 x1_arr,x2_arr,edges_x1,edges_x2,'ECTF',RHOp,R[0][0],\
                                 outdir_figures,seed_num,PDFu_x1,PDFu_x2)

    ## Time and go to the next trial
    end_time_trial = time.time()
    print('\n')
    print('Trial runtime: {:.2f}s'.format(end_time_trial-start_time_trial))
    print('\n')


    ## Finalize
    end_time_total = time.ctime()
    print('\n')
    print('PROGRAM ENDED: {}'.format(end_time_total))
    print('\n')
    return None


def format_output(output_tmp):
    """
    Format the output produced by an ensemble filter.
    output_tmp : list
    """
    output = []
    output.append('{}'.format(output_tmp[0])) # seed num
    output.append('{:.2f}'.format(output_tmp[1])) # RHOp
    output.append('{:.2f}'.format(output_tmp[2])) # R
    output.append('{:.2f}'.format(output_tmp[3])) # y
    output.append('{:.2f}'.format(output_tmp[4])) # prior mean in x1
    output.append('{:.2f}'.format(output_tmp[5])) # prior mean in x2
    output.append('{:.1f}'.format(output_tmp[6])) # JS div
    output.append('{:.3f}'.format(output_tmp[7])) # ME in E[Xa]
    output.append('{:.3f}'.format(output_tmp[8])) # ME in Var[Xa]
    output.append('{:.1f}'.format(output_tmp[9])) # perc outside
    output.append('{:.2f}'.format(output_tmp[10])) # MUp_x1
    output.append('{:.2f}'.format(output_tmp[11])) # MUp_x2
    output.append('{:.2f}'.format(output_tmp[12])) # Sp_x1
    output.append('{:.2f}'.format(output_tmp[13])) # Sp_x2
    output.append('{:.2f}'.format(output_tmp[14])) # MUu_x1
    output.append('{:.2f}'.format(output_tmp[15])) # MUu_x2
    output.append('{:.2f}'.format(output_tmp[16])) # Su_x1
    output.append('{:.2f}'.format(output_tmp[17])) # Su_x2
    output.append('{:.3f}'.format(output_tmp[18])) # PDFp_int2
    output.append('{:.3f}'.format(output_tmp[19])) # PDFu_theory_int2
    output.append('{:.3f}'.format(output_tmp[20])) # PDFu_explicit_int2
    output.append('{:.3f}'.format(output_tmp[21])) # maxDiff_PDFu
    return output 

if __name__ == "__main__":
    main()
