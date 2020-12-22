import numpy as np

from src.rnn import RNN

from tqdm import tqdm

from stdParams import *
import os

import sys

T_run = int(5e2)

eps_a_r = 1e-1

n_samples = 50

range_a_r_init = [0.,2.5]
range_sigm_y_squ_init = [0.,1.]

#### 4 pairs of (sigm_ext,R_t)
params = np.array([[0.0,1.],
        [0.05,0.7],
        [0.25,1.3],
        [0.5,1.]])

sigm_y_squ = np.ndarray((4,n_samples,T_run))
a = np.ndarray((4,n_samples,T_run))

for k in tqdm(range(4)):
    for l in tqdm(range(n_samples),leave=False):
        
        rnn = RNN()
        rnn.eps_a_r = eps_a_r
        
        rnn.W = rnn.W/np.abs(np.linalg.eigvals(rnn.W)).max()
        
        rnn.R_target = params[k,1]
        rnn.w_in = np.random.normal(0.,1.,(rnn.N,1)) * params[k,0]
        
        rnn.a_r[:] = np.random.rand()*(range_a_r_init[1] - range_a_r_init[0]) + range_a_r_init[0]
        
        u_in_adapt = 2.*(np.random.rand(T_run) <= .5) -1.
        
        y_init_squ = np.random.rand()*(range_sigm_y_squ_init[1] - range_sigm_y_squ_init[0]) + range_sigm_y_squ_init[0]
        
        y_init = np.random.rand(rnn.N)-.5
        y_init *= y_init_squ**.5 / (y_init**2.).mean()**.5
        
        y, X_r, X_e, a_r, b, y_mean, y_std = rnn.run_hom_adapt("flow",u_in=u_in_adapt,
                            y_init=y_init,adapt_mode="global",norm_flow=False,show_progress=False)
        
        sigm_y_squ[k,l] = (y**2.).mean(axis=1)
        a[k,l] = a_r[:,0]

os.makedirs(os.path.dirname(__file__)+"/data/", exist_ok = True) 
np.savez(os.path.dirname(__file__)+"/data/sim_data.npz",
        a=a,
        sigm_y_squ=sigm_y_squ,
        params=params,
        range_a_r_init=range_a_r_init,
        range_sigm_y_squ_init=range_sigm_y_squ_init,
        eps_a_r=eps_a_r)