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

dot_size = 5

T_it = 1000
N_sample = 10000

fig, ax = plt.subplots(2,2,figsize=(TEXT_WIDTH,TEXT_WIDTH))

dat = np.load(os.path.dirname(__file__)+"/data/sim_data.npz")

sigm_y = dat["sigm_y"]
sigm_y_analytic = dat["sigm_y_analytic"]
sigm_ext= dat["sigm_ext"]
R_a = dat["R_a"]
input_types = dat["input_types"]

for k in range(2):
    for l in range(2):
        m = int(k*2 + l)
        for n in range(sigm_ext.shape[0]):
            col = viridis(n/sigm_ext.shape[0])
            ax[k,l].plot(sigm_y[m,n],R_a,'^',markersize=dot_size,c=col)
            ax[k,l].plot(sigm_y_analytic[n],R_a,c=col)            
                  
        
        ax[k,l].set_xlabel(r'$\sigma_{\rm y}$')
        ax[k,l].set_ylabel(r'$R_{\rm a}$')
        
fig.tight_layout(pad=0.1,h_pad=0.1,w_pad=0.3)

for k in range(2):
    for l in range(2):
        m = int(k*2 + l)
        
        title_idx = '{\\bf %s}' % (["A","B","C","D"][m])
        ax_title = '\\makebox['+str(ax[k,l].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ '+title_idx+' \\hfill \\normalfont '+input_types[m]+'}'

        ax[k,l].set_title(ax_title,loc='left',usetex=True)

fig.tight_layout(pad=0.1,h_pad=0.1,w_pad=0.3)     

fig.savefig(os.path.dirname(__file__)+'/Fig9.pdf')
fig.savefig(os.path.dirname(__file__)+'/Fig9.png',dpi=1000)

if not(args.hide_plot):
   plt.show()