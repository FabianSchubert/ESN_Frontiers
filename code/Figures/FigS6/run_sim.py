import numpy as np
from src.rnn import RNN
import os

from tqdm import tqdm

T_run = int(1e4)

n_sweep_sigm_e = 15
sigm_e = np.linspace(0.05,1.,n_sweep_sigm_e)

R_t = np.array([0.5,1.0,1.5])
n_sweep_R_t = R_t.shape[0]

n_samples = 1

R_a = np.ndarray((n_samples,n_sweep_R_t,n_sweep_sigm_e))

input_types = ["het_bin",
                "het_gauss",
                "hom_bin",
                "hom_gauss"]
                
for type in tqdm(input_types):
    for n in tqdm(range(n_samples),leave=False):
        for k in tqdm(range(n_sweep_R_t),leave=False):
            for l in tqdm(range(n_sweep_sigm_e),leave=False):
                
                rnn = RNN()
                rnn.R_target = R_t[k]
                
                if type=="het_bin":
                    rnn.w_in = np.random.normal(0.,1.,(rnn.N,1)) * sigm_e[l]
                    u_in = 2.*(np.random.rand(T_run) <= 0.5) - 1.
                    rnn.run_hom_adapt("variance",
                                    u_in=u_in,
                                    adapt_mode="local",
                                    show_progress=False)
                if type=="het_gauss":
                    #rnn.w_in = np.random.normal(0.,1.,(rnn.N,1)) * sigm_e[l]
                    sigm_e_temp = np.abs(np.random.normal(0.,1.,(rnn.N))*sigm_e[l])
                    rnn.run_hom_adapt("variance",
                                    T=T_run,
                                    sigm_e=sigm_e_temp,
                                    adapt_mode="local",
                                    show_progress=False)
                    
                if type=="hom_bin":
                    rnn.w_in = np.ones((rnn.N,1)) * sigm_e[l]
                    u_in = 2.*(np.random.rand(T_run) <= 0.5) - 1.
                    rnn.run_hom_adapt("variance",
                                    u_in=u_in,
                                    adapt_mode="local",
                                    show_progress=False)
                if type=="hom_gauss":
                    #rnn.w_in = np.random.normal(0.,1.,(rnn.N,1)) * sigm_e[l]
                    sigm_e_temp = np.ones((rnn.N))*sigm_e[l]
                    rnn.run_hom_adapt("variance",
                                    T=T_run,
                                    sigm_e=sigm_e_temp,
                                    adapt_mode="local",
                                    show_progress=False)
                
                R_a[n,k,l] = rnn.get_R_a()

    os.makedirs(os.path.dirname(__file__)+"/data/", exist_ok = True) 

    np.savez(os.path.dirname(__file__)+"/data/sim_data_%s.npz" % (type),
            R_a=R_a,
            sigm_e = sigm_e,
            R_t = R_t)