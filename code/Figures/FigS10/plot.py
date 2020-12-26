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

files = ["sim_data_binary.npz",
        "sim_data_gauss.npz"]

cmap_vals = [0.,0.4,0.8]

line_width = 1.5
err_alpha = 0.3

fig,ax = plt.subplots(1,2,figsize=(TEXT_WIDTH,TEXT_WIDTH*0.5))

for k in range(2):
    
    dat = np.load(os.path.dirname(__file__)+"/data/"+files[k])
    
    C = dat["C"]
    sigma_ext = dat["sigma_ext"]
    R_a = dat["R_a"]
    
    n_samples = C.shape[0]
    
    C_mean = C.mean(axis=0)
    C_err = C.std(axis=0)/n_samples**.5
    
    for l in range(sigma_ext.shape[0]):
        col = viridis(cmap_vals[l])
        lab = '$\\sigma_{\\rm ext} = ' + str(sigma_ext[l]) + '$'
        ax[k].plot(R_a,C_mean[l],lw=line_width,c=col,label=lab)
        ax[k].fill_between(R_a,C_mean[l]-C_err[l],C_mean[l]+C_err[l],color=col,alpha=err_alpha)    
    
    ax[k].set_yscale("log")
    
    ax[k].legend()
    
    ax[k].set_xlabel(r'$R_{\rm a}$')
    ax[k].set_ylabel(r'$\overline{C}$')
    
fig.tight_layout(pad=0.1,h_pad=0.1,w_pad=0.3)

ax0_title = '\\makebox['+str(ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf A} \\hfill \\normalfont heterogeneous binary}'
ax1_title = '\\makebox['+str(ax[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf B} \\hfill \\normalfont heterogeneous gauss}'

ax[0].set_title(ax0_title,loc='left',usetex=True)
ax[1].set_title(ax1_title,loc='left',usetex=True)

fig.tight_layout(pad=0.1,h_pad=0.1,w_pad=0.3)
   
fig.savefig(os.path.dirname(__file__)+'/FigS10.pdf')
fig.savefig(os.path.dirname(__file__)+'/FigS10.png',dpi=1000)

if not(args.hide_plot):
   plt.show()