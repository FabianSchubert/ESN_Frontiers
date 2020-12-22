import numpy as np

from src.rnn import RNN

from tqdm import tqdm

from stdParams import *
import os

import sys

T_run_sample = int(1e4)

sigma_ext_list = [0.,.5,1.5]
n_sweep_sigma_ext = len(sigma_ext_list)

n_sweep_R_a = 15
R_a_list = np.linspace(0.,2.,n_sweep_R_a)

n_samples = 5


# heterogeneous binary
################################

C = np.ndarray((n_samples,n_sweep_sigma_ext,n_sweep_R_a))

for n in tqdm(range(n_samples)):
    for k in tqdm(range(n_sweep_sigma_ext), leave=False):
        for l in tqdm(range(n_sweep_R_a), leave=False):
            
            rnn = RNN()
            
            rnn.W = R_a_list[l] * rnn.W/np.abs(np.linalg.eigvals(rnn.W)).max()
            
            rnn.w_in = np.random.normal(0.,1.,(rnn.N,1))*sigma_ext_list[k]
            
            u_in_sample = 2.*(np.random.rand(T_run_sample) <= 0.5) - 1.
            
            y, X_r, X_e = rnn.run_sample(u_in=u_in_sample,show_progress=False)
            
            corr = np.corrcoef(y.T)
            
            C[n,k,l] = (np.abs(corr).sum() - rnn.N)/(rnn.N*(rnn.N - 1.))
            
np.savez(os.path.dirname(__file__)+"/data/sim_data_binary.npz",
        C=C,
        sigma_ext = sigma_ext_list,
        R_a = R_a_list)

################################

# heterogeneous gauss
################################

C = np.ndarray((n_samples,n_sweep_sigma_ext,n_sweep_R_a))

for n in tqdm(range(n_samples)):
    for k in tqdm(range(n_sweep_sigma_ext), leave=False):
        for l in tqdm(range(n_sweep_R_a), leave=False):
            
            rnn = RNN()
            
            rnn.W = R_a_list[l] * rnn.W/np.abs(np.linalg.eigvals(rnn.W)).max()
            
            y, X_r, X_e = rnn.run_sample(T=T_run_sample,
                                    sigm_e=np.abs(np.random.normal(0.,1.,(rnn.N))*sigma_ext_list[k]),
                                    show_progress=False)
            
            corr = np.corrcoef(y.T)
            
            C[n,k,l] = (np.abs(corr).sum() - rnn.N)/(rnn.N*(rnn.N - 1.))
            
np.savez(os.path.dirname(__file__)+"/data/sim_data_gauss.npz",
        C=C,
        sigma_ext = sigma_ext_list,
        R_a = R_a_list)

################################