import numpy as np

from src.rnn import RNN

from tqdm import tqdm

from stdParams import *
import os

import sys

T_run = int(1e3)

sigm_ext = np.array([0.0,0.5,1.5])
n_sweep_sigm_ext = sigm_ext.shape[0]

n_sweep_R_a = 30
R_a = np.linspace(0.,5.,n_sweep_R_a)

sigm_y = np.ndarray((4,n_sweep_sigm_ext,n_sweep_R_a))
sigm_y_analytic = np.ndarray((n_sweep_sigm_ext,n_sweep_R_a))

input_types = ["homogeneous gaussian",
                "homogeneous binary",
                "heterogeneous gaussian",
                "heterogeneous binary"]

T_it_analytic = int(1e3)
N_samples = int(1e4)

for k in tqdm(range(4)):
    for l in tqdm(range(n_sweep_sigm_ext),leave=False):
        for m in tqdm(range(n_sweep_R_a),leave=False):
            
            rnn = RNN()
            #rnn.y_std_target = sigm_y_t[m]
            rnn.W = R_a[m] * rnn.W / np.abs(np.linalg.eigvals(rnn.W)).max()
            
            if k==0:
                #### hom gauss
                y, X_r, X_e = rnn.run_sample(T=T_run,
                sigm_e=sigm_ext[l],show_progress=False)
            elif k==1:
                ### hom binary
                rnn.w_in = np.ones((rnn.N,1))*sigm_ext[l]
                u_in = 2.*(np.random.rand(T_run) <= .5) - 1.
                y, X_r, X_e = rnn.run_sample(u_in=u_in,show_progress=False)
            elif k==2:
                ### het gauss
                y, X_r, X_e = rnn.run_sample(T=T_run,
                sigm_e=np.abs(np.random.normal(0.,1.,(rnn.N)))*sigm_ext[l],show_progress=False)
            else:
                ### het bin
                rnn.w_in = np.random.normal(0.,1.,(rnn.N,1))*sigm_ext[l]
                u_in = 2.*(np.random.rand(T_run) <= .5) - 1.
                y, X_r, X_e = rnn.run_sample(u_in=u_in,show_progress=False)   
            
            sigm_y[k,l,m] = y.std()
            
            if(k==0):
                y = np.tanh(R_a[m]*np.random.normal(0.,1.,(N_samples)) + 
                            np.random.normal(0.,1.,(N_samples))*sigm_ext[l])
                
                for t in range(T_it_analytic):
                    sigm_y_temp = y.std()
                    y = np.tanh(R_a[m]*np.random.normal(0.,1.,(N_samples)) * sigm_y_temp + 
                            np.random.normal(0.,1.,(N_samples))*sigm_ext[l])
                
                sigm_y_analytic[l,m] = sigm_y_temp

os.makedirs(os.path.dirname(__file__)+"/data/", exist_ok = True)            
np.savez(os.path.dirname(__file__)+"/data/sim_data.npz",
    sigm_y = sigm_y,
    sigm_y_analytic = sigm_y_analytic,
    sigm_ext= sigm_ext,
    R_a = R_a,
    input_types = input_types)
            