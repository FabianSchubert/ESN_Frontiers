#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

from matplotlib.lines import Line2D

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from matplotlib import cm
viridis = cm.get_cmap('viridis')

plt.rc('text.latex', preamble=r'''
\usepackage{dejavu}
\renewcommand*\familydefault{\sfdefault}
\usepackage[T1]{fontenc}''')

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--hide_plot",action='store_true')

args = parser.parse_args()

from stdParams import *
import os

files = ["sim_data_no_renorm.npz",
        "sim_data_with_renorm.npz"]

line_width = 1.5
err_alpha = .3

fig, ax = plt.subplots(1,2,figsize=(TEXT_WIDTH,TEXT_WIDTH*0.5))        

for k in range(2):
    
    dat = np.load(os.path.dirname(__file__)+"/data/"+files[k])
    
    T_conv = dat["T_conv"]
    sigma_ext = dat["sigma_ext"]
    R_t = dat["R_t"]
    R_a_est = dat["R_a_est"]
    R_t_arr = dat["R_t_arr"]
    R_a_thresh = dat["R_a_thresh"]
    
    T_conv_mask = np.ma.masked_array(T_conv,mask = np.abs(R_a_est-R_t_arr)>R_a_thresh)     
    #import pdb
    #pdb.set_trace()
    
    n_samples = T_conv.shape[0]
    n_R_t = T_conv.shape[1]
    n_sigm_ext = T_conv.shape[2]
    
    T_conv_mean = T_conv_mask.mean(axis=0)
    T_conv_err = T_conv_mask.std(axis=0)/n_samples**.5
    
    for l in range(n_R_t):
        ax[k].plot(sigma_ext[1:], T_conv_mean[l,1:],c=viridis(l/n_R_t),lw=line_width)
        ax[k].fill_between(sigma_ext[1:],
        T_conv_mean[l,1:] - T_conv_err[l,1:],
        T_conv_mean[l,1:] + T_conv_err[l,1:],
        color=viridis(l/n_R_t),
        alpha=err_alpha)
        #ax[k].plot(sigma_ext[1:],R_a_est.mean(axis=0)[l,1:],'--',c=viridis(l/n_R_t))
        
    ax[k].set_xlim(left=0.)
    ax[k].set_ylim([0.,15000.])
    
    ax[k].set_xlabel(r'$\sigma_{\rm ext}$')
    ax[k].set_ylabel(r'$T_{\rm conv}$')

custom_lines = [Line2D([0], [0], color=viridis(0.), lw=2),
                    Line2D([0], [0], color=viridis(.5), lw=2),
                    Line2D([0], [0], color=viridis(1.), lw=2)]

ax[1].legend(custom_lines, [('%.2f' % sigma_ext[1]),
                            ('%.2f' % sigma_ext[int(n_sigm_ext/2.)]),
                            ('%.2f' % sigma_ext[n_sigm_ext-1])],
                            title=r'$R_{\rm t}$')

fig.tight_layout(pad=0.1,h_pad=0.1,w_pad=0.3)

ax0_title = '\\makebox['+str(ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf A} \\hfill \\normalfont without renormalization}'
ax1_title = '\\makebox['+str(ax[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf B} \\hfill \\normalfont with renormalization}'

ax[0].set_title(ax0_title,loc='left',usetex=True)
ax[1].set_title(ax1_title,loc='left',usetex=True)

fig.tight_layout(pad=0.1,h_pad=0.1,w_pad=0.3)

fig.savefig(os.path.dirname(__file__)+'/Fig7.pdf')
fig.savefig(os.path.dirname(__file__)+'/Fig7.png',dpi=1000)

if not(args.hide_plot):
   plt.show()