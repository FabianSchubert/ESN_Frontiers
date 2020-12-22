#!/usr/bin/env python3
import numpy as np
from stdParams import *
import os
import sys
import glob
from pathlib import Path

from src.analysis_tools import get_simfile_prop

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

import matplotlib.patches as mpatches

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from tqdm import tqdm

import argparse

def plot(ax,path,label_str,col=colors[0]):

    dat = np.load(path)

    a_rec = dat['a']
    W = dat['W']

    l_start = np.linalg.eigvals((W.T * a_rec[0,:]).T)
    l_end = np.linalg.eigvals((W.T * a_rec[-1,:]).T)

    #ax.plot(l_start.real,l_start.imag,'.',markersize=5,label='$t=0$')
    #sc_not_fact = a_rec.shape[1]/10**sc_not_exp
    #sc_not_exp = int(np.log10(a_rec.shape[1]))
    #ax.plot(l_end.real,l_end.imag,'.',markersize=5,label='$t='+str(sc_not_fact)+'\\times 10^'+str(sc_not_exp)+'$')
    ax.plot(l_end.real,l_end.imag,'.',markersize=4,alpha=0.8,c=col,label=label_str)
    circle = plt.Circle((0,0),np.abs(l_end).max(),facecolor=(0,0,0,0),edgecolor=col,lw=1.5,linestyle='--')
    ax.add_artist(circle)

    ax.set_xlabel('$\\mathrm{Re}(\\lambda_i)$')
    ax.set_ylabel('$\\mathrm{Im}(\\lambda_i)$')

    ax.legend()

    ax.axis('equal')
