#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

from stdParams import *
import os

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--hide_plot",action='store_true')

args = parser.parse_args()
    
def R(sigm_y,sigm_e):
    v_y = sigm_y**2.
    return ((1.-(1.+2.*sigm_e**2.)*(1-v_y)**2.)/(2.*v_y*(1.-v_y)**2.))**.5


s_y = np.linspace(1e-3,0.6,10000)

fig, ax = plt.subplots(1,1,figsize=(TEXT_WIDTH,0.4*TEXT_WIDTH))

ax.plot(R(s_y,0.1),s_y,label=r'$\sigma_{\rm ext} = 0.1$')
ax.plot(R(s_y,0.01),s_y,label=r'$\sigma_{\rm ext} = 0.01$')
ax.plot(R(s_y,0.001),s_y,label=r'$\sigma_{\rm ext} = 0.001$')


ax.plot([1.,1.],[-.5,1.],'--',c='k')

ax.set_xlim([0.8,1.2])
ax.set_ylim([-0.02,0.6])

ax.set_xlabel(r'$R_{\rm a}$')
ax.set_ylabel(r'$\sigma_{\rm y}$')

ax.legend()

fig.tight_layout(pad=0.1,rect=(0.2,0.0,0.8,1.))

fig.savefig(os.path.dirname(__file__)+'/Fig10.pdf')
fig.savefig(os.path.dirname(__file__)+'/Fig10.png',dpi=1000)

if not(args.hide_plot):
   plt.show()

plt.show()
