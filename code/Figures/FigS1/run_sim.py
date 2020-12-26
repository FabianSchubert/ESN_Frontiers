import numpy as np
from src.rnn import RNN
import os

from tqdm import tqdm

T_run = int(1e4)
T_skip = 10
sigm_e = 0.5

input_types = ["heterogeneous binary",
                "heterogeneous gaussian",
                "homogeneous binary",
                "homogeneous gaussian"]

for type in tqdm(input_types):

    rnn = RNN()

    rnn.a_r[:] = 1.5
    
    if type=="heterogeneous binary":
    
        rnn.w_in = np.random.normal(0.,1.,(rnn.N,1)) * sigm_e
        
        u_in = 2.*(np.random.rand(T_run) <= 0.5) - 1.
            
        y, X_r, X_e, a_r, b, y_mean, y_std = rnn.run_hom_adapt("flow",
                                                u_in=u_in,
                                                adapt_mode="local",
                                                T_skip_rec=T_skip)

        os.makedirs(os.path.dirname(__file__)+"/data/", exist_ok = True) 

        np.savez(os.path.dirname(__file__)+"/data/sim_data_het_bin.npz",
                W=rnn.W,
                a=a_r,
                skip_steps=T_skip)
    
    if type=="heterogeneous gaussian":
        
        y, X_r, X_e, a_r, b, y_mean, y_std = rnn.run_hom_adapt("flow",
                                                T=T_run,
                                                adapt_mode="local",
                                                T_skip_rec=T_skip,
                                                sigm_e=np.abs(np.random.normal(0.,1.,(rnn.N))*sigm_e))

        os.makedirs(os.path.dirname(__file__)+"/data/", exist_ok = True) 

        np.savez(os.path.dirname(__file__)+"/data/sim_data_het_gauss.npz",
                W=rnn.W,
                a=a_r,
                skip_steps=T_skip)
                
    
    if type=="homogeneous binary":
    
        rnn.w_in = np.ones((rnn.N,1)) * sigm_e
        
        u_in = 2.*(np.random.rand(T_run) <= 0.5) - 1.
            
        y, X_r, X_e, a_r, b, y_mean, y_std = rnn.run_hom_adapt("flow",
                                                u_in=u_in,
                                                adapt_mode="local",
                                                T_skip_rec=T_skip)

        os.makedirs(os.path.dirname(__file__)+"/data/", exist_ok = True) 

        np.savez(os.path.dirname(__file__)+"/data/sim_data_hom_bin.npz",
                W=rnn.W,
                a=a_r,
                skip_steps=T_skip)
    
    if type=="homogeneous gaussian":
        
        y, X_r, X_e, a_r, b, y_mean, y_std = rnn.run_hom_adapt("flow",
                                                T=T_run,
                                                adapt_mode="local",
                                                T_skip_rec=T_skip,
                                                sigm_e=np.ones((rnn.N))*sigm_e)

        os.makedirs(os.path.dirname(__file__)+"/data/", exist_ok = True) 

        np.savez(os.path.dirname(__file__)+"/data/sim_data_hom_gauss.npz",
                W=rnn.W,
                a=a_r,
                skip_steps=T_skip)