#!/usr/bin/env python3
import numpy as np
from stdParams import *
import os
import sys
import glob
from pathlib import Path
from tqdm import tqdm

from src.analysis_tools import get_simfile_prop

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from tqdm import tqdm

import sys

import argparse

def plot(ax,path,col=colors[0],skip_steps=1):

    dat = np.load(path)

    a_rec = dat['a']
    W = dat['W']
    skip_steps_rec = dat['skip_steps']
    
    n_steps = a_rec.shape[0]
    
    r_a = a_rec[::skip_steps,:]**2. * (W**2.).sum(axis=1)
    
    t = (np.arange(n_steps)*skip_steps_rec)[::skip_steps]
    
    n_steps_plot = t.shape[0]
    
    R_a = np.ndarray((n_steps_plot))
    
    print("Calculating spectral radii...")
    for k in tqdm(range(n_steps_plot)):
        R_a[k] = np.abs(np.linalg.eigvals(a_rec[k*skip_steps]*W.T)).max()
    
    ax.plot(t,r_a[:,0],c=col,alpha=0.25,label='$R^2_{{\\rm a},i}$')
    ax.plot(t,r_a[:,1:100],c=col,alpha=0.25)

    ax.plot(t,R_a**2.,'--',c='k',label='$R^2_{\\rm a}$',lw=2)

    ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0),useMathText=True)

    leg = ax.legend()
    for lh in leg.legendHandles:
        lh.set_alpha(1)

    ax.set_xlabel('time steps')
    ax.set_ylabel('$R^2_{{\\rm a},i}$')