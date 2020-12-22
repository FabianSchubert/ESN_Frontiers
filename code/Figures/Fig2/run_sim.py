import numpy as np
from src.rnn import RNN
import os

T_run = int(5e4)
T_skip = 10
sigm_e = 0.5

rnn = RNN()

rnn.a_r[:] = 1.5

y, X_r, X_e, a_r, b, y_mean, y_std = rnn.run_hom_adapt("variance",
                                        T=T_run,
                                        adapt_mode="local",
                                        T_skip_rec=T_skip,
                                        sigm_e = np.abs(np.random.normal(0.,sigm_e,(rnn.N))))

np.savez(os.path.dirname(__file__)+"/data/sim_data.npz",
        W=rnn.W,
        a=a_r,
        skip_steps=T_skip)