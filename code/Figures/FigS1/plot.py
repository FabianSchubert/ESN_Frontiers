#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')

import plotting.r_a as r_a

#from matplotlib.lines import Line2D

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from stdParams import *

#from matplotlib import cm
#viridis = cm.get_cmap('viridis')

plt.rc('text.latex', preamble=r'''
\usepackage{dejavu}
\renewcommand*\familydefault{\sfdefault}
\usepackage[T1]{fontenc}''')

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--hide_plot",action='store_true')

args = parser.parse_args()

input_types = ["heterogeneous binary",
                "heterogeneous gaussian",
                "homogeneous binary",
                "homogeneous gaussian"]

files = ["sim_data_het_bin.npz",
        "sim_data_het_gauss.npz",
        "sim_data_hom_bin.npz",
        "sim_data_hom_gauss.npz"]

fig, ax = plt.subplots(2,2,figsize=(TEXT_WIDTH,TEXT_WIDTH))

for k in range(2):
    for l in range(2):
        m = k*2 + l

        r_a.plot(ax[k,l],os.path.dirname(__file__)+"/data/"+files[m],col=colors[0],skip_steps=5)

fig.tight_layout(pad=0.1,h_pad=0.1,w_pad=0.3)

for k in range(2):
    for l in range(2):
        m = k*2 + l

        ax_title = ('\\makebox['
        +str(ax[k,l].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+
        'in]{ {\\bf '+['A','B','C','D'][m]+'} \\hfill \\normalfont '
        +input_types[m] +'}')

        ax[k,l].set_title(ax_title,loc='left',usetex=True)

fig.tight_layout(pad=0.1,h_pad=0.1,w_pad=0.3)

fig.savefig(os.path.dirname(__file__)+'/FigS1.pdf')
fig.savefig(os.path.dirname(__file__)+'/FigS1.png',dpi=1000)

if not(args.hide_plot):
   plt.show()
