#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

from matplotlib.lines import Line2D

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

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

dat = np.load(os.path.dirname(__file__)+"/data/sim_data.npz")

N = dat["N"]

MAE = dat["MAE"]

n_samples = MAE.shape[1]
n_N = MAE.shape[2]

MAE_mean = MAE.mean(axis=1)
MAE_err = MAE.std(axis=1)/n_samples**.5

color_remap = [3,1,2,0]

input_types = ["heterogeneous binary",
                "homogeneous binary",
                "heterogeneous gaussian",
                "homogeneous gaussian"]

fig = plt.figure(figsize=(TEXT_WIDTH,TEXT_WIDTH*0.5))

ax = plt.subplot(111)

for k in range(4):
    ax.plot(N,MAE_mean[k],c=colors[color_remap[k]])
    ax.fill_between(N,MAE_mean[k]-MAE_err[k],MAE_mean[k]+MAE_err[k],color=colors[color_remap[k]],alpha=0.3)

ax.set_yscale("log")
ax.set_xscale("log")

custom_lines = [Line2D([0], [0], color=colors[color_remap[0]], lw=2),
                Line2D([0], [0], color=colors[color_remap[1]], lw=2),
                Line2D([0], [0], color=colors[color_remap[2]], lw=2),
                Line2D([0], [0], color=colors[color_remap[3]], lw=2)]

ax.legend(custom_lines,['heterogeneous binary',
                        'homomogeneous binary',
                        'heterogeneous gaussian',
                        'homogeneous gaussian'])


fig.tight_layout(rect=[0.1, 0, 0.9, 1],pad=0.1)

fig.savefig(os.path.dirname(__file__)+'/Fig5.pdf')
fig.savefig(os.path.dirname(__file__)+'/Fig5.png',dpi=1000)

if not(args.hide_plot):
   plt.show()
