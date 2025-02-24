#!/usr/bin/env python
# -*- coding: utf-8 -*-

"Abstract: Plotting tools to visualize the ensemble filter updates."

## Packages
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import inspect
import numpy as np

## Functions
def plot_multivariate_update(Xa,Xa_density2d,Xa_JSdiv,Xa_me_EX,Xa_me_stdX,Xa_num_outside,\
                             Xf,Ne,xt,y,EX1u,EX2u,PDFp,likelihood,PDFu_theory,\
                             x1_arr,x2_arr,xedges,yedges,filter_title,RHOp,R,\
                             outdir_figures,seed_num,PDFu_x1,PDFu_x2):
    
    """ 
    Assess the multivariate ensemble filter performance: plot a 2D histogram
    of the analysis ensemble against the true posterior PDF. The histogram and 
    the PDF have consistent 'resolutions' in the sense that the PDF is always 
    evaluated in the middle of each histogram bin.

    Parameters
    ----------
    Xa : ndarray [Nx,Ne]
        Analysis ensemble.
    Xa_density2d : 2D ndarray
        2D histogram of the analysis ensemble.
    Xa_JSdiv : float
        Jensen-Shannon divergence of the analysis ensemble.
    Xa_me_EX : float 
        Mean error of the analysis ensemble mean 
        with respect to the posterior expectation vector.
    Xa_me_stdX : float 
        Mean error of the posterior ensemble std 
        with respect to the posterior std (std=standard deviation).
    Xa_num_outside : int
        Number of analysis members outside of physical bounds.
    Xf : ndarray [Nx,Ne]
        Forecast ensemble.
    Ne : int
        Ensemble size.
    xt : ndarray [Nx]
        True state.
    y : float
        Observation.
    EX1u,EX1u : floats
        Expectation values associated with the posterior PDF.
    PDFp,likelihood,PDFu_theory : 2D ndarrays
        Prior PDF, likelihood function and posterior PDF.
    x1_arr,x2_arr : 1D ndarrays
        Arrays for the 2 state components.
    xedges,yedges : 1D ndarrays
        Arrays defining the location of the 2D histogram bins.
    filter_title : str
        Name of the ensemble filter to be used in the title.
    RHOp : float
        Correlation coefficient of latent prior covariance.
    R : float
        Covariance parameter of latent observation noise.
    outdir_figures : string
        Path to directory where figures are saved.
    seed_num : int
        Seed number corresponding to a particular DA trial.
    PDFu_x1,PDFu_x2 : 1D ndarrays [same dim as x1_arr and x2_arr]
        Marginal posterior PDFs.
 
    Returns
    -------
        None.
    """
   
    ## Settings
    x1_plot_min,x1_plot_max = 0.01,3.6+0.01 # x-axis range
    x2_plot_min,x2_plot_max = 0.01,0.67 # y-axis range
    xticks = np.arange(0.0,x1_plot_max+0.01,0.5)
    yticks = np.arange(0.1,0.61,0.1)
 
    ## Set up plotting environment
    fontsize = 45
    plt.rcParams.update({
        'font.size': fontsize,
    })

    ## Definitions for the axes
    left,width = 0.1,0.99
    bottom,height = 0.1,0.99
    spacing = 0.01
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    ## Add axes to figure
    fig = plt.figure(figsize=(12,12),facecolor='white')
    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx,sharex=ax)
    ax_histy = fig.add_axes(rect_histy,sharey=ax)

    ## Remove labels from marginal histogram plots
    ax_histx.tick_params(axis="x",labelbottom=False)
    ax_histy.tick_params(axis="y",labelleft=False)

    ## 2D histograms
    prior_PDF_plot = ax.contour(x1_arr,x2_arr,PDFp,2,colors='white',\
                                linewidths=2.5,zorder=101,linestyles='--')
    prior_PDF_plot.collections[0].set_label('prior')
    likelihood_plot = ax.plot(x1_arr,0.04*likelihood[0,:],'-',color='cyan',\
                              linewidth=2.5,label='likelihood',zorder=100)
    ob_plot = ax.plot(y,0.025,marker=(8,2,0),markersize=25,color='cyan',\
                      mew=3.0,linestyle='None',label='observation')
    posterior_PDF_theory_plot = ax.contour(x1_arr,x2_arr,PDFu_theory,3,colors='white',\
                                           linewidths=2.5,zorder=102)
    posterior_PDF_theory_plot.collections[0].set_label('posterior')
    X,Y = np.meshgrid(xedges,yedges)
    ax.pcolormesh(X,Y,Xa_density2d,cmap='inferno')
    ax.plot(EX1u,EX2u,marker='o',markersize=25.0,markerfacecolor='white',\
            markeredgecolor='black',markeredgewidth=1.0,\
            linestyle='None',label='true posterior mean',zorder=111)
    ax.plot(Xf.mean(axis=1)[0],Xf.mean(axis=1)[1],marker='x',markersize=19,\
            color='white',mew=5.0,linestyle='None',label='prior ensemble mean',zorder=110)
    ax.plot(Xa.mean(axis=1)[0],Xa.mean(axis=1)[1],marker='x',markersize=19,\
            color='maroon',mew=5.0,linestyle='None',label='posterior ensemble mean',zorder=111)
    #ax.plot(xt[0],xt[1],marker='s',markersize=10,\
    #        color='cyan',mew=5.0,linestyle='None',label='truth',zorder=110)
    ax.set_xlabel(r'$z_1$',labelpad=15,fontsize=fontsize)
    ax.set_ylabel(r'$z_2$',labelpad=60,rotation=0,fontsize=fontsize)
    ax.set_xlim(x1_plot_min,x1_plot_max)
    ax.set_xticks(xticks)
    ax.set_ylim(x2_plot_min,x2_plot_max)
    ax.set_yticks(yticks)
#    leg=ax.legend(loc='lower center',prop={'size':22},bbox_to_anchor=(0.5,-0.65))
#    leg.get_frame().set_alpha(1.0)
#    leg.get_frame().set_edgecolor('black')
#    leg.get_frame().set_facecolor('lightgray')

    ## 1D histograms
    ax_histx.hist(Xa[0,:],color='dimgray',bins='auto',density='True',alpha=0.7)
    ax_histx.plot(x1_arr,PDFu_x1,linestyle='-',color='black',linewidth=2.5)
    ax_histx.set_ylim([0.0,4.1])
    ax_histx.set_yticks([1.0,3.0])
    ax_histx.tick_params(axis='x',which='both',length=0)
    ax_histy.hist(Xa[1,:],color='dimgray',bins='auto',density='True',alpha=0.7,\
                  orientation='horizontal')
    ax_histy.plot(PDFu_x2,x2_arr,linestyle='-',color='black',linewidth=2.5)
    ax_histy.set_xlim([0.0,13.0])
    ax_histy.set_xticks([2.0,5.0,8.0,12.0])
    ax_histy.tick_params(axis='y',which='both',length=0)
    
    ## Title
#    ax_histx.set_title(r'$\mathbf{%s\ analysis}: \rho=%.2f, R=%.2f$'%(filter_title,RHOp,R)+\
#                 '\nJS div $={:.1f}$, '.format(Xa_JSdiv)+\
#                 r'ME in E[X$^a$]=%.2f, '%(Xa_me_EX)+\
#                 r'ME in Var[X$^a$]=%.2f, '%(Xa_me_stdX)+'\n'+\
#                 'analysis mems outside bounds $={:.2f}\%$'.format(100.0*Xa_num_outside/Ne),\
#                 fontsize=26.0,y=1.1)
    ax_histx.set_title('%s'%(filter_title),fontsize=fontsize+3,y=1.08,fontweight='bold')
    


    ## Save figure
    filename = '{}_rho={:.2f}_R={:.2f}_seedNum={}'.format(filter_title,RHOp,R,seed_num)
    plt.savefig('{}/{}.png'.format(outdir_figures,filename),bbox_inches='tight',dpi=300)
    return None 
