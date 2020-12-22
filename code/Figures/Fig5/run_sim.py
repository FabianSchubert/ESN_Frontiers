import numpy as np

from src.rnn import RNN

from tqdm import tqdm

from stdParams import *
import os

import sys

T_run_adapt = int(1e4)
T_prerun_sample = 100
T_run_sample = int(1e3)

sigm_e = 0.5

n_samples = 10

N_list = [100,200,300,400,500,700,1000]
n_N = len(N_list)


#####################################

# Mean Absolute Error
MAE_list = np.ndarray((4,n_samples,n_N))

####################################

input_types = ["heterogeneous binary",
                "homogeneous binary",
                "heterogeneous gaussian",
                "homogeneous gaussian"]

for input_type in tqdm(input_types):

    for k in tqdm(range(n_N),leave=False):

        N = N_list[k]

        for l in tqdm(range(n_samples),leave=False):

            rnn = RNN(N=N)
            
            rnn.W /= np.abs(np.linalg.eigvals(rnn.W)).max()
            
            rnn.eps_a_r = 0.

            ##################

            if input_type == "heterogeneous binary":

                rnn.w_in = np.random.normal(0.,1.,(N,1)) * sigm_e

                u_in_adapt = 2.*(np.random.rand(T_run_adapt) <= 0.5) - 1.

                adapt = rnn.run_hom_adapt_var_fix(u_in=u_in_adapt,T_skip_rec=1000,show_progress=False)

                #run test sample
                u_in_sample = 2.*(np.random.rand(T_run_sample + T_prerun_sample) <= 0.5) - 1.

                y_res,X_r_res,X_e_res = rnn.run_sample(u_in=u_in_sample,show_progress=False)


            elif input_type == "homogeneous binary":

                rnn.w_in = np.ones((N,1)) * sigm_e

                u_in_adapt = 2.*(np.random.rand(T_run_adapt) <= 0.5) - 1.

                adapt = rnn.run_hom_adapt_var_fix(u_in=u_in_adapt,T_skip_rec=1000,show_progress=False)

                #run test sample
                u_in_sample = 2.*(np.random.rand(T_run_sample + T_prerun_sample) <= 0.5) - 1.

                y_res,X_r_res,X_e_res = rnn.run_sample(u_in=u_in_sample,show_progress=False)
                
            elif input_type == "heterogeneous gaussian":

                #rnn.w_in = np.random.normal(0.,1.,(N,1))

                adapt = rnn.run_hom_adapt_var_fix(T=T_run_adapt,
                sigm_e=np.abs(np.random.normal(0.,sigm_e,(N))),
                T_skip_rec=1000,show_progress=False)

                y_res,X_r_res,X_e_res = rnn.run_sample(T=T_run_sample+T_prerun_sample,
                sigm_e=np.abs(np.random.normal(0.,sigm_e,(N))),show_progress=False)

            else:

                adapt = rnn.run_hom_adapt_var_fix(T=T_run_adapt,
                sigm_e=sigm_e,
                T_skip_rec=1000,show_progress=False)

                y_res,X_r_res,X_e_res = rnn.run_sample(T=T_run_sample+T_prerun_sample,
                sigm_e=sigm_e,show_progress=False)
                
            ####################################

            Var_X_r = X_r_res[T_prerun_sample:,:].var(axis=0)
            Var_y = y_res[T_prerun_sample:,:].var()
            Var_W = rnn.W.var(axis=1) * rnn.N
            
            MAE = np.abs(Var_X_r - Var_y*Var_W).mean()
            
            MAE_list[input_types.index(input_type),l,k] = MAE



################################

os.makedirs(os.path.dirname(__file__)+"/data/", exist_ok = True) 
np.savez(os.path.dirname(__file__)+"/data/sim_data.npz",
        MAE=MAE_list,
        N = N_list)