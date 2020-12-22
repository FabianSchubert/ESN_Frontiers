import numpy as np

from src.rnn import RNN

from tqdm import tqdm

from stdParams import *
import os

import sys

T_run = int(2e4)

dR_a_thresh = 1e-3
R_a_thresh = 0.25
T_dr = 10

n_sweep_sigma_ext = 15
sigma_ext_list = np.linspace(0.,1.,n_sweep_sigma_ext)


n_sweep_R_t = 5
R_t_list = np.linspace(0.01,.3,n_sweep_R_t)

n_samples = 1

# no renorm
#######################

T_conv = np.ndarray((n_samples,n_sweep_R_t,n_sweep_sigma_ext))
R_a_est_end = np.ndarray((n_samples,n_sweep_R_t,n_sweep_sigma_ext))
R_t_arr = np.ndarray((n_samples,n_sweep_R_t,n_sweep_sigma_ext))

for n in tqdm(range(n_samples)):
    for k in tqdm(range(n_sweep_R_t),leave=False):
        for l in tqdm(range(n_sweep_sigma_ext),leave=False):
            
            rnn = RNN()
            rnn.W = (R_t_list[k] + .5) * rnn.W / np.abs(np.linalg.eigvals(rnn.W)).max()
            rnn.R_target = R_t_list[k]
            rnn.eps_a_r = 1e-3
            
            y, X_r, X_e, a_r, b, y_mean, y_std = rnn.run_hom_adapt("flow",
                    sigm_e=np.abs(np.random.normal(0.,1.,(rnn.N))*sigma_ext_list[l]),
                    T=T_run,
                    norm_flow=False,
                    show_progress=False)
            
            R_a_est = (a_r**2. * ((rnn.W**2.)).sum(axis=1)).mean(axis=1)**.5
            
            dR_a_est = (R_a_est[T_dr:] - R_a_est[:-T_dr])/T_dr
    
            T_conv[n,k,l] = np.argmax(((np.abs(dR_a_est)<dR_a_thresh)*(np.abs(R_a_est[T_dr:]-R_t_list[k])<R_a_thresh)))
            
            R_a_est_end[n,k,l] = R_a_est[-1]
            R_t_arr[n,k,l] = R_t_list[k]     

os.makedirs(os.path.dirname(__file__)+"/data/", exist_ok = True) 
np.savez(os.path.dirname(__file__)+"/data/sim_data_no_renorm.npz",
        T_conv=T_conv,
        R_a_est = R_a_est_end,
        R_t_arr = R_t_arr,
        sigma_ext = sigma_ext_list,
        R_t = R_t_list,
        dR_a_thresh = dR_a_thresh,
        R_a_thresh = R_a_thresh,
        T_dr = T_dr)
        
# with renorm
######################

T_conv = np.ndarray((n_samples,n_sweep_R_t,n_sweep_sigma_ext))
R_a_est_end = np.ndarray((n_samples,n_sweep_R_t,n_sweep_sigma_ext))
R_t_arr = np.ndarray((n_samples,n_sweep_R_t,n_sweep_sigma_ext))

for n in tqdm(range(n_samples)):
    for k in tqdm(range(n_sweep_R_t),leave=False):
        for l in tqdm(range(n_sweep_sigma_ext),leave=False):
            
            rnn = RNN()
            rnn.W = (R_t_list[k] + .5) * rnn.W / np.abs(np.linalg.eigvals(rnn.W)).max()
            rnn.R_target = R_t_list[k]
            rnn.eps_a_r = 1e-3
            
            y, X_r, X_e, a_r, b, y_mean, y_std = rnn.run_hom_adapt("flow",
                    sigm_e=np.abs(np.random.normal(0.,1.,(rnn.N))*sigma_ext_list[l]),
                    T=T_run,
                    norm_flow=True,
                    show_progress=False)
            
            R_a_est = (a_r**2. * ((rnn.W**2.)).sum(axis=1)).mean(axis=1)**.5
            
            dR_a_est = (R_a_est[T_dr:] - R_a_est[:-T_dr])/T_dr
    
            T_conv[n,k,l] = np.argmax(((np.abs(dR_a_est)<dR_a_thresh)*(np.abs(R_a_est[T_dr:]-R_t_list[k])<R_a_thresh)))
            
            R_a_est_end[n,k,l] = R_a_est[-1]
            R_t_arr[n,k,l] = R_t_list[k]
        
np.savez(os.path.dirname(__file__)+"/data/sim_data_with_renorm.npz",
        T_conv=T_conv,
        R_a_est = R_a_est_end,
        R_t_arr = R_t_arr,
        sigma_ext = sigma_ext_list,
        R_t = R_t_list,
        dR_a_thresh = dR_a_thresh,
        R_a_thresh = R_a_thresh,
        T_dr = T_dr)