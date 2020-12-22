import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
mpl.style.use('matplotlibrc')
import os

plt.rc('text.latex', preamble=r'''
\usepackage{dejavu}
\renewcommand*\familydefault{\sfdefault}
\usepackage[T1]{fontenc}''')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from stdParams import *
import os

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--hide_plot",action='store_true')

args = parser.parse_args()

lw = 1.5
mask_th = 0.1

dat_binary = np.load(os.path.dirname(__file__)+"/data/sim_data_binary.npz")
dat_gauss = np.load(os.path.dirname(__file__)+"/data/sim_data_gauss.npz")

sigm_e = dat_binary["sigm_e"]
R_t = dat_binary["R_t"]

sigm_e_ax = np.append(sigm_e,2.*sigm_e[-1] - sigm_e[-2]) - 0.5*(sigm_e[1] - sigm_e[0])
R_t_ax = np.append(R_t,2.*R_t[-1] - R_t[-2]) - 0.5*(R_t[1] - R_t[0])

R_binary = dat_binary["R"]
MC_binary = dat_binary["MC"]
MC_abs_binary = MC_binary.sum(axis=2)

R_gauss = dat_gauss["R"]
MC_gauss = dat_gauss["MC"]
MC_abs_gauss = MC_gauss.sum(axis=2)

MC_binary_ma = np.ma.masked_array(MC_abs_binary, mask=MC_abs_binary <= mask_th)
MC_gauss_ma = np.ma.masked_array(MC_abs_gauss, mask=MC_abs_gauss <= mask_th)

idx_max_binary = np.argmax(MC_abs_binary,axis=1)
idx_max_gauss = np.argmax(MC_abs_gauss,axis=1)

fig, ax = plt.subplots(1,2,figsize=(TEXT_WIDTH,TEXT_WIDTH*0.5))

pc0 = ax[0].pcolormesh(R_t_ax,sigm_e_ax,MC_binary_ma,cmap="viridis")
pc1 = ax[1].pcolormesh(R_t_ax,sigm_e_ax,MC_gauss_ma,cmap="viridis")

plt.colorbar(mappable=pc0,ax=ax[0])
plt.colorbar(mappable=pc1,ax=ax[1])

ax[0].contour(R_binary,levels=[1.],colors="w",linestyles="dashed",linewidths=lw)
ax[1].contour(R_gauss,levels=[1.],colors="w",linestyles="dashed",linewidths=lw)

ax[0].plot(R_t[idx_max_binary],sigm_e,c=BRIGHT_YELLOW,linewidth=lw)
ax[1].plot(R_t[idx_max_gauss],sigm_e,c=BRIGHT_YELLOW,linewidth=lw)

ax[0].set_xlabel(r'$R_{\rm t}$')
ax[1].set_xlabel(r'$R_{\rm t}$')

ax[0].set_ylabel(r'$\sigma_{\rm ext}$')
ax[1].set_ylabel(r'$\sigma_{\rm ext}$')

ax[0].set_xlim([R_t_ax[0],R_t_ax[-1]])
ax[1].set_xlim([R_t_ax[0],R_t_ax[-1]])

ax[0].set_ylim([sigm_e_ax[0],sigm_e_ax[-1]])
ax[1].set_ylim([sigm_e_ax[0],sigm_e_ax[-1]])

fig.tight_layout(pad=0.1,h_pad=0.1,w_pad=0.3)

ax0_title = '\\makebox['+str(ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf A} \\hfill \\normalfont heterogeneous binary}'
ax1_title = '\\makebox['+str(ax[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width)+'in]{ {\\bf B} \\hfill \\normalfont heterogeneous gauss}'

ax[0].set_title(ax0_title,loc='left',usetex=True)
ax[1].set_title(ax1_title,loc='left',usetex=True)

fig.tight_layout(pad=0.1,h_pad=0.1,w_pad=0.3)

fig.savefig(os.path.dirname(__file__)+'/Fig3.pdf')
fig.savefig(os.path.dirname(__file__)+'/Fig3.png',dpi=1000)

#fig.savefig(os.path.join(PLOT_DIR,'alt_hom_regulation_composite_low_res.png'),dpi=300)
if not(args.hide_plot):
   plt.show()