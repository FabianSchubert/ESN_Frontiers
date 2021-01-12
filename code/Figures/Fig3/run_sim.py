import numpy as np
from src.rnn import RNN
import os
from tqdm import tqdm

T_adapt = int(2e4)
T_learn = int(2e3)
T_prerun_learn = 100
T_test = int(2e3)


n_sweep_sigm_e = 30
n_sweep_R_t = 30

sigm_e = np.linspace(0.,1.,n_sweep_sigm_e)
R_t = np.linspace(0.,2.,n_sweep_R_t)

tau_max = 15

R = np.ndarray((n_sweep_sigm_e,n_sweep_R_t))
MC = np.ndarray((n_sweep_sigm_e,n_sweep_R_t,tau_max))

##### heterogeneous binary

for k in tqdm(range(n_sweep_sigm_e)):
    for l in tqdm(range(n_sweep_R_t),leave=False):

        rnn = RNN()

        rnn.R_target = R_t[l]
        rnn.w_in = np.random.normal(0.,sigm_e[k],(rnn.N,1))

        ###### adaptation
        u_in_adapt = (np.random.rand(T_adapt) <= 0.5) * 2. - 1.

        rnn.run_hom_adapt("flow",u_in=u_in_adapt,show_progress=False)

        R[k,l] = np.abs(np.linalg.eigvals(rnn.a_r * rnn.W.T)).max()

        ##### learning
        for tau in tqdm(range(tau_max),leave=False):
            ###### learn w_out
            u_in_learn = (np.random.rand(T_adapt) <= 0.5) * 2. - 1.

            if(tau > 0):
                u_out_learn = (u_in_learn[1:-tau] != u_in_learn[:-(tau+1)])*1.
            else:
                u_out_learn = (u_in_learn[1:] != u_in_learn[:-1])*1.
            u_in_learn = u_in_learn[1+tau:]

            rnn.learn_w_out_trial(u_in=u_in_learn,u_target=u_out_learn,T_prerun=T_prerun_learn)

            ##### test

            u_in_test = (np.random.rand(T_test) <= 0.5) * 2. - 1.

            if(tau > 0):
                u_out_test = (u_in_test[1:-tau] != u_in_test[:-(tau+1)])*1.
            else:
                u_out_test = (u_in_test[1:] != u_in_test[:-1])*1.

            u_in_test = u_in_test[1+tau:]

            u_out_pred = rnn.predict_data(u_in_test,show_progress=False)

            MC[k,l,tau] = np.corrcoef(u_out_test,u_out_pred)[1,0]**2.

os.makedirs(os.path.dirname(__file__)+"/data/", exist_ok = True)
np.savez(os.path.dirname(__file__)+"/data/sim_data_binary.npz",
        R=R,
        MC=MC,
        sigm_e=sigm_e,
        R_t=R_t)

##### heterogeneous binary

for k in tqdm(range(n_sweep_sigm_e)):
    for l in tqdm(range(n_sweep_R_t),leave=False):

        rnn = RNN()

        rnn.R_target = R_t[l]
        rnn.w_in = np.random.normal(0.,sigm_e[k],(rnn.N,1))

        ###### adaptation
        #u_in_adapt = (np.random.rand(T_adapt) <= 0.5) * 2. - 1.

        rnn.run_hom_adapt("flow",T=T_adapt,sigm_e=np.abs(rnn.w_in[:,0]),show_progress=False)

        R[k,l] = np.abs(np.linalg.eigvals(rnn.a_r * rnn.W.T)).max()

        ##### learning
        for tau in tqdm(range(tau_max),leave=False):
            ###### learn w_out
            u_in_learn = (np.random.rand(T_adapt) <= 0.5) * 2. - 1.

            if(tau > 0):
                u_out_learn = (u_in_learn[1:-tau] != u_in_learn[:-(tau+1)])*1.
            else:
                u_out_learn = (u_in_learn[1:] != u_in_learn[:-1])*1.
            u_in_learn = u_in_learn[1+tau:]

            rnn.learn_w_out_trial(u_in=u_in_learn,u_target=u_out_learn,T_prerun=T_prerun_learn)

            ##### test

            u_in_test = (np.random.rand(T_test) <= 0.5) * 2. - 1.

            if(tau > 0):
                u_out_test = (u_in_test[1:-tau] != u_in_test[:-(tau+1)])*1.
            else:
                u_out_test = (u_in_test[1:] != u_in_test[:-1])*1.

            u_in_test = u_in_test[1+tau:]

            u_out_pred = rnn.predict_data(u_in_test,show_progress=False)

            MC[k,l,tau] = np.corrcoef(u_out_test,u_out_pred)[1,0]**2.


np.savez(os.path.dirname(__file__)+"/data/sim_data_gauss.npz",
        R=R,
        MC=MC,
        sigm_e=sigm_e,
        R_t=R_t)
