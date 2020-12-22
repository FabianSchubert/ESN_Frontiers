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

fig, ax = plt.subplots(2,2,figsize=(TEXT_WIDTH,TEXT_WIDTH))

dat = np.load(os.path.dirname(__file__)+"/data/sim_data.npz")

y_squ = dat["sigm_y_squ"]
a = dat["a"]
range_a_r_init = dat["range_a_r_init"]
range_y_squ_init = dat["range_sigm_y_squ_init"]
params = dat["params"]
eps_a_r = dat["eps_a_r"]

n_samples = y_squ.shape[1]

a_mesh = np.linspace(range_a_r_init[0],range_a_r_init[1],200)
y_squ_mesh = np.linspace(range_y_squ_init[0],range_y_squ_init[1],200)

A, Y_SQU = np.meshgrid(a_mesh,y_squ_mesh)



for k in range(2):
    for l in range(2):
        m = int(k*2 + l)
        
        for n in range(n_samples):
            ax[k,l].plot(a[m,n],y_squ[m,n],c="tab:orange")
        
        ax[k,l].plot(a[m,:,-1],y_squ[m,:,-1],'.',c='k',markersize=5)
            
        ax[k,l].set_xlabel(r'$a$')
        ax[k,l].set_ylabel(r'$\sigma_{\rm y}^2$')
        
        R_t = params[m,1]
        sigm_ext = params[m,0]
        
        DA = eps_a_r*(R_t**2. - A**2.)*Y_SQU
        DY_SQU = 1.-Y_SQU-1./(1.+2.*A**2.*Y_SQU + 2.*sigm_ext**2.)**.5
        
        ax[k,l].streamplot(a_mesh,y_squ_mesh,DA,DY_SQU)

fig.tight_layout(pad=0.1,h_pad=0.1,w_pad=0.3)

for k in range(2):
    for l in range(2):
        m = int(k*2 + l)
        
        form_string = '$\\sigma_{\\rm ext} = %.2f, \; R_{\\rm t} = %.2f$' % (params[m,0],params[m,1])
        title_idx = '{\\bf %s}' % (["A","B","C","D"][m])
        ax_title = '\\makebox['+str(ax[k,l].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ '+title_idx+' \\hfill \\normalfont '+form_string+'}'

        ax[k,l].set_title(ax_title,loc='left',usetex=True)

fig.tight_layout(pad=0.1,h_pad=0.1,w_pad=0.3)
         
fig.savefig(os.path.dirname(__file__)+'/Fig8.pdf')
fig.savefig(os.path.dirname(__file__)+'/Fig8.png',dpi=1000)

if not(args.hide_plot):
   plt.show()