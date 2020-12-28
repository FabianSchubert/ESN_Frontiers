#!/usr/bin/env python3

import numpy as np

from tqdm import tqdm

from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt

class RNN():
    """
    A discrete time, rate-encoding RNN, implementing flow and variance control.
    
    Attributes
    ----------
    N : int
        Number of nodes in the network
    cf : float
        The recurrent connection fraction
    cf_in : float
        The external input connection fraction
    a_r : float numpy array of shape (N)
        Postsynaptic weight scaling factors for the 
        recurrent membrane potential
    b : float numpy array of shape (N)
        Biases
    mu_w : float
        Mean of recurrent weights
    mu_w_in : float
        Mean of external weights
    dim_in : int
        Dimension of the input layer. 
        I.e., the external input weight matrix has
        shape (N,dim_in).
    dim_out : int
        Dimension of the output layer.
        I.e., the output weight matrix has
        shape (dim_out,N + 1). The additional 
        dimension accounts for a potential bias 
        in the output.
    eps_a_r : float numpy array of shape (N)
        Local homeostatic adaptation rates for 
        weight scaling factors a_r.
    eps_b : float numpy array of shape (N)
        Local homeostatic adaptation rates for 
        biases b.
    eps_y_mean : float numpy array of shape (N)
        Local adaptation rates for trailing 
        averages of the mean activity y.
    eps_y_std : float numpy array of shape (N)
        Local adaptation rates for trailing 
        averages of the standard devation of the 
        activitiy y.
    eps_E_mean: float numpy array of shape (N)
        Local adaptation rates for trailing 
        averages of the external input.
    eps_E_std: float numpy array of shape (N)
        Local adaptation rates for trailing 
        averages of the standard devation of the 
        external input.
    y_mean_target : float numpy array of shape (N)
        Local homeostatic targets for the mean activity.
    y_std_target : float numpy array of shape (N)
        Local homeostatic targets for the standard 
        deviation of activity.
    R_target : float
        Homeostatic target for the spectral radius 
        of the effective weight matrix a_r_i * W_ij.
    W : float numpy array of shape (N,N)
        The bare recurrent weight matrix with a 
        connection probability cf and weights 
        drawn randomly from a Gaussian distribution 
        with mean mu_w and a standard deviation 
        of 1/sqrt(N*cf). The diagonal is always 
        set to zero (no autapses).
    w_in : float numpy array of shape (N,dim_in)
        External input weight matrix with a 
        connection probability cf_in. 
        Upon initialization, weights are drawn 
        from a Gaussian distribution with mean 
        mu_w_in and a standard devation of 1.
    w_out : float numpy array of shape (dim_out,N+1)
        Output weight matrix. Initialized with 
        uniform random weights. Note that the 
        additional dimension accounts for a 
        potential bias in the output.
    
    Methods
    -------
    f(x)
        The nonlinear activiation function.
    df_x(x)
        First derivative of f(x) as a function of x.
    df_y(y):
        First derivative of f(x), expressed as 
        a function of y = f(x).
    check_data_in_comp(data)
        Checks if the array data could serve as
        an input sequence, given dim_in
    check_data_out_comp(data)
        Checks if the array data is compatible
        with the dimensionality of the output
        layer dim_out.
    learn_w_out_trial(u_in,u_target,reg_fact)
        Least squares batch training of w_out 
        using an input sequence u_in and 
        a target output u_target. Thikonov 
        regularization is used with a regularization 
        factor reg_fact.
    predict_data(data)
        Create an output sequence for a given
        input sequence passed by data.
        Obviously, this function should be called
        after training w_out using the method
        learn_w_out_trial.
    run_hom_adapt_var_fix(u_in)
        Run homeostatic adaptation on a_r and b
        under an input sequence u_in, using the
        networks' homeostatic targets y_mean_target
        and y_std_target.
    run_hom_adapt(adapt_rule,u_in)
        Run either flow control (by passing "flow" via adapt_rule)
        or variance control (by passing "variance" via adapt_rule)
        under an input_sequence u_in, using the networks'
        homeostatic target R_target.
    run_sample(u_in)
        Return recordings of 
        neural activity y, the recurrent contribution 
        to the membrane potential X_r, and the external 
        contribution to the membrane potential X_e, 
        under an external input sequence u_in    
    """
    
    def __init__(self,**kwargs):

        self.N = kwargs.get('N',500)
        self.cf = kwargs.get('cf',.1)
        self.cf_in = kwargs.get('cf_in',1.)
        self.a_r = np.ones((self.N)) * kwargs.get('a_r',1.)
        self.b = np.ones((self.N)) * kwargs.get('b',0.)
        self.mu_w = kwargs.get('mu_w',0.)
        self.mu_w_in = kwargs.get('mu_w_in',0.)

        self.dim_in = kwargs.get('dim_in',1)
        self.dim_out = kwargs.get('dim_out',1)

        self.eps_a_r = np.ones((self.N)) * kwargs.get('eps_a_r',0.001)
        self.eps_b = np.ones((self.N)) * kwargs.get('eps_b',0.001)
        self.eps_y_mean = np.ones((self.N)) * kwargs.get('eps_y_mean',0.0001)
        self.eps_y_std = np.ones((self.N)) * kwargs.get('eps_y_std',0.001)
        self.eps_E_mean = np.ones((self.N)) * kwargs.get('eps_E_mean',0.0001)
        self.eps_E_std = np.ones((self.N)) * kwargs.get('eps_E_std',0.001)
        
        self.y_mean_target = np.ones((self.N)) * kwargs.get('y_mean_target',0.05)
        self.y_std_target = np.ones((self.N)) * kwargs.get('y_std_target',.5)
        
        self.R_target = kwargs.get('R_target',1.)
        
        self.W = np.random.normal(self.mu_w,1./(self.N*self.cf)**.5,(self.N,self.N))*(np.random.rand(self.N,self.N) <= self.cf)
        self.W[range(self.N),range(self.N)] = 0.

        self.w_in = np.random.normal(self.mu_w_in,1.,(self.N,self.dim_in))
        self.w_out = np.random.rand(self.dim_out,self.N+1)*0.01
        self.w_out[:,0] = 0.

    def f(self,x):
        """
        The neural transfer function as a function
        of the membrane potential x.
        Default is f(x) = tanh(x).
        """
        #return (np.tanh(2.*x)+1.)/2.
        return np.tanh(x)

    def df_x(self,x):
        """
        The derivative df(x)/dx as a function
        of the membrane potential x.
        Default is df(x)/dx = 1 - tanh(x)^2.
        Note that manually changing f(x) does 
        note automatically override this function.
        """
        f = self.f(x)
        #return 4.*f*(1.-f)
        return 1.-f**2.

    def df_y(self,y):
        """
        The derivative df(x)/dx as a function
        of the activity itself.
        Default is (df(x)/dx)(y) = 1 - y^2.
        Note that manually changing f(x) does 
        note automatically override this function.
        """
        #return 4.*y*(1.-y)
        return 1.-y**2.

    def check_data_in_comp(self,data):
        """
        Check if data has the appropriate
        shape to serve as an input sequence:
            1.  If dim_in = 1, data may either
                have shape (T) or (T,1), where
                T is the number of time steps
                in the sequence.
            2.  If dim_in > 1, data must have
                shape (T,dim_in).
        """
        if len(data.shape)==1:
            if self.dim_in != 1:
                print("input dimensions do not fit!")
                sys.exit()
            return np.array([data]).T

        elif (len(data.shape)>2) or (data.shape[1] != self.dim_in):
            print("input dimensions do not fit!")
            sys.exit()

        return data

    def check_data_out_comp(self,data):
        """
        Check if data has a
        shape that fits the dimension of
        the output layer:
            1.  If dim_out = 1, data may either
                have shape (T) or (T,1), where
                T is the number of time steps
                in the sequence.
            2.  If dim_out > 1, data must have
                shape (T,dim_out).
        """
        
        if len(data.shape)==1:
            if self.dim_out != 1:
                print("output dimensions do not fit!")
                #sys.exit()
            return np.array([data]).T

        elif (len(data.shape)>2) or (data.shape[1] != self.dim_out):
            print("output dimensions do not fit!")
            #sys.exit()

        return data

    def learn_w_out_trial(self,u_in,u_target,reg_fact=.01,show_progress=False,T_prerun=0):
        """
        This function does linear least squares regression using the
        neural activities acquired from running the network
        with an input sequence u_in, and an output target 
        sequence u_target. Thikonov regularization is used.
        
        Optional parameters:
            reg_fact : float, default = 0.01
                Set the regularization factor
                of the linear regression.
            show_progress : bool, default = False
                Specifies whether to show a progress
                bar when running the network simulation.
            T_prerun : int, default = 0
                Optionally, the first T_prerun time
                steps can be omitted for the linear
                regression.            
        """
        u_in = self.check_data_in_comp(u_in)
        u_target = self.check_data_out_comp(u_target)

        n_t = u_in.shape[0]


        y = np.ndarray((n_t,self.N+1))
        y[:,0] = 1.


        y[0,1:] = self.f(self.w_in @ u_in[0,:] - self.b)


        for t in tqdm(range(1,n_t),disable=not(show_progress)):

            y[t,1:] = self.f(self.a_r*(self.W.dot(y[t-1,1:])) + self.w_in @ u_in[t,:] - self.b)

        self.w_out[:,:] = (np.linalg.inv(y[T_prerun:,:].T @ y[T_prerun:,:] + reg_fact*np.eye(self.N+1)) @ y[T_prerun:,:].T @ u_target[T_prerun:,:]).T

    def predict_data(self,data,return_reservoir_rec=False,show_progress=True):
        """
        Generate an output sequence 
        
            u_out_i(t) = sum_j=1^N w_out_ij y_j(t)
        
        under a given input
        sequence given by data.
        
        Optional parameters:
            return_reservoir_rec : bool, default = False
                If True, the function returns
                (out,y), where out is the generated output
                sequence and y is the recorded neural activity.
                If False, only out is returned.
            show_progress : bool, default = True
                Specifies whether to show a progress
                bar when running the network simulation.             
        """
        data = self.check_data_in_comp(data)

        n_t = data.shape[0]

        u_in = data

        y = np.ndarray((n_t,self.N+1))
        y[:,0] = 1.

        y[0,1:] = self.f(self.w_in @ u_in[0,:] - self.b)

        for t in tqdm(range(1,n_t),disable=not(show_progress)):

            y[t,1:] = self.f(self.a_r*(self.W.dot(y[t-1,1:])) + self.w_in @ u_in[t,:] - self.b)

        out = (self.w_out @ y.T).T
        if self.dim_out == 1:
            out = out[:,0]

        if return_reservoir_rec:
            return (out,y)
        else:
            return out

    def run_hom_adapt_var_fix(self,u_in=None,sigm_e=1.,T_skip_rec = 1,T=None,show_progress=True):
        """
        This function does linear least squares regression using the
        neural activities acquired from running the network
        with an input sequence u_in, and an output target 
        sequence u_target. Thikonov regularization is used.
        
        Optional parameters:
            reg_fact : float, default = 0.01
                Set the regularization factor
                of the linear regression.
            show_progress : bool, default = False
                Specifies whether to show a progress
                bar when running the network simulation.
            T_prerun : int, default = 0
                Optionally, the first T_prerun time
                steps can be omitted for the linear
                regression.            
        """        
        
        if u_in is not None:
            mode = 'real_input'
            u_in = self.check_data_in_comp(u_in)
            T = u_in.shape[0]
        else:
            mode = 'gaussian_noise_input'
            if T == None:
                T = self.N*50

        T_rec = int(T/T_skip_rec)

        #### Recorders
        y_rec = np.ndarray((T_rec,self.N))
        X_r_rec = np.ndarray((T_rec,self.N))
        X_e_rec = np.ndarray((T_rec,self.N))
        a_r_rec = np.ndarray((T_rec,self.N))
        b_rec = np.ndarray((T_rec,self.N))
        y_mean_rec = np.ndarray((T_rec,self.N))
        y_std_rec = np.ndarray((T_rec,self.N))
        ####

        y = np.ndarray((self.N))
        y_mean = np.ndarray((self.N))
        y_var = np.ndarray((self.N))

        X_r = np.ndarray((self.N))
        X_e = np.ndarray((self.N))

        X_r[:] = np.random.normal(0.,1.,(self.N))
        #X_e[:] = self.w_in @ u_in[0,:]
        if mode == 'real_input':
            X_e = self.w_in @ u_in[0,:]
        else:
            X_e = np.random.normal(0.,sigm_e,(self.N))

        y = self.f(self.a_r * X_r + X_e - self.b)
        y_mean = y[:]
        y_var[:] = 0.

        delta_a = np.zeros((self.N))
        delta_b = np.zeros((self.N))

        #### Assign for t=0
        y_rec[0,:] = y
        X_r_rec[0,:] = X_r
        X_e_rec[0,:] = X_e
        a_r_rec[0,:] = self.a_r
        b_rec[0,:] = self.b
        y_mean_rec[0,:] = y_mean
        y_std_rec[0,:] = y_var**.5
        ####

        for t in tqdm(range(1,T),disable=not(show_progress)):

            X_r = self.W @ y
            if mode == 'real_input':
                X_e = self.w_in @ u_in[t,:]
            else:
                X_e = np.random.normal(0.,sigm_e,(self.N))

            y = self.f(self.a_r * X_r + X_e - self.b)

            y_mean += self.eps_y_mean * ( -y_mean + y)
            y_var += self.eps_y_std * ( -y_var + (y-y_mean)**2.)

            delta_a = (self.y_std_target**2. - (y-y_mean)**2.)
            delta_b = (y - self.y_mean_target)

            self.a_r += self.eps_a_r * delta_a

            self.a_r = np.maximum(0.001,self.a_r)

            self.b += self.eps_b * delta_b

            #### record
            if t%T_skip_rec == 0:

                t_rec = int(t/T_skip_rec)

                y_rec[t_rec,:] = y
                X_r_rec[t_rec,:] = X_r
                X_e_rec[t_rec,:] = X_e
                a_r_rec[t_rec,:] = self.a_r
                b_rec[t_rec,:] = self.b
                y_mean_rec[t_rec,:] = y_mean
                y_std_rec[t_rec,:] = y_var**.5

        return y_rec, X_r_rec, X_e_rec, a_r_rec, b_rec, y_mean_rec, y_std_rec
    
    
    def run_hom_adapt(self,adapt_rule,
                    u_in=None,
                    sigm_e=1.,
                    T_skip_rec = 1,
                    T=None,
                    adapt_mode="local",
                    norm_flow=True,
                    show_progress=True,
                    y_init=None):
                    
        if(not(adapt_rule in ("variance","flow"))):
            print("Error: adapt_rule must be either 'variance' or 'flow'.")
            return False
        
        if u_in is not None:
            mode = 'real_input'
            u_in = self.check_data_in_comp(u_in)
            T = u_in.shape[0]
        else:
            mode = 'gaussian_noise_input'
            if T == None:
                T = self.N*50

        T_rec = int((T-1)/T_skip_rec) + 1
        
        #check parameter
        if not(adapt_mode in ["local","global"]):
            raise ValueError("wrong adaptation mode parameter!")
        
        
        #### Recorders
        y_rec = np.ndarray((T_rec,self.N))
        X_r_rec = np.ndarray((T_rec,self.N))
        X_e_rec = np.ndarray((T_rec,self.N))
        a_r_rec = np.ndarray((T_rec,self.N))
        b_rec = np.ndarray((T_rec,self.N))
        y_mean_rec = np.ndarray((T_rec,self.N))
        y_std_rec = np.ndarray((T_rec,self.N))
        
        E_mean_rec = np.ndarray((T_rec,self.N))
        E_std_rec = np.ndarray((T_rec,self.N))
        
        var_t_rec = np.ndarray((T_rec,self.N))
        ####

        y = np.ndarray((self.N))
        y_prev = np.ndarray((self.N))
        y_mean = np.ndarray((self.N))
        y_var = np.ndarray((self.N))

        E_mean = np.ndarray((self.N))
        E_var = np.ndarray((self.N))
        
        var_t = np.ndarray((self.N))
        
        X_r = np.ndarray((self.N))
        X_e = np.ndarray((self.N))

        X_r[:] = np.random.normal(0.,1.,(self.N))
        
        if mode == 'real_input':
            X_e = self.w_in @ u_in[0,:]
        else:
            X_e = np.random.normal(0.,sigm_e,(self.N))
        
        if y_init is None:
            y = self.f(self.a_r * X_r + X_e - self.b)
        else:
            y = y_init
        
        y_prev[:] = y
        y_mean[:] = 0.
        y_var[:] = 0.25
        
        E_mean[:] = 0.
        E_var[:] = 0.25
        
        var_t[:] = 0.25

        delta_a = np.zeros((self.N))
        delta_b = np.zeros((self.N))

        #### Assign for t=0
        y_rec[0,:] = y
        X_r_rec[0,:] = X_r
        X_e_rec[0,:] = X_e
        a_r_rec[0,:] = self.a_r
        b_rec[0,:] = self.b
        y_mean_rec[0,:] = y_mean
        y_std_rec[0,:] = y_var**.5
        E_mean_rec[0] = E_mean
        E_std_rec[0] = E_var**.5
        var_t_rec[0,:] = var_t
        ####

        for t in tqdm(range(1,T),disable=not(show_progress)):

            X_r = self.a_r * (self.W @ y)
            if mode == 'real_input':
                X_e = self.w_in @ u_in[t,:]
            else:
                X_e = np.random.normal(0.,sigm_e,(self.N))

            y = self.f( X_r + X_e - self.b )

            y_mean += self.eps_y_mean * ( -y_mean + y)
            y_var += self.eps_y_std * ( -y_var + (y-y_mean)**2.)
            
            E_mean += self.eps_E_mean * ( -E_mean + X_e)
            E_var += self.eps_E_std * ( -E_var + (X_e - E_mean)**2.)
            
            if(adapt_rule == "variance"):
                    
                if adapt_mode == "local":
                    var_t = 1. - 1./(1. + 2. * self.R_target**2. * y_var +  2.*E_var)**.5
                else:
                    var_t = 1. - 1./(1. + 2. * self.R_target**2. * y_var.mean() + 2.*E_var)**.5
                
                delta_a = (var_t - (y-y_mean)**2.)
            
            else:
                if adapt_mode == "local":
                    delta_a = self.a_r * (self.R_target**2.*y_prev**2. - X_r**2.)
                else:
                    delta_a = self.a_r * (self.R_target**2.*(y_prev**2.).mean() - (X_r**2.).mean())

                if norm_flow:
                    delta_a /= (X_r**2.).mean()
                    
            y_prev[:] = y                
            
            delta_b = (y - self.y_mean_target)

            self.a_r += self.eps_a_r * delta_a

            self.a_r = np.maximum(0.001,self.a_r)

            self.b += self.eps_b * delta_b

            #### record
            if t%T_skip_rec == 0:

                t_rec = int(t/T_skip_rec)
                                
                y_rec[t_rec,:] = y
                X_r_rec[t_rec,:] = X_r
                X_e_rec[t_rec,:] = X_e
                a_r_rec[t_rec,:] = self.a_r
                b_rec[t_rec,:] = self.b
                y_mean_rec[t_rec,:] = y_mean
                y_std_rec[t_rec,:] = y_var**.5

        return y_rec, X_r_rec, X_e_rec, a_r_rec, b_rec, y_mean_rec, y_std_rec

    


    def run_sample(self,u_in=None,sigm_e=1.,X_r_init=None,T_skip_rec = 1,T=None,show_progress=True):

        if u_in is not None:
            mode = 'real_input'
            u_in = self.check_data_in_comp(u_in)
            T = u_in.shape[0]
        else:
            mode = 'gaussian_noise_input'
            if T == None:
                T = self.N*50

        T_rec = int(T/T_skip_rec)
        
        #### Recorders
        y_rec = np.ndarray((T_rec,self.N))
        X_r_rec = np.ndarray((T_rec,self.N))
        X_e_rec = np.ndarray((T_rec,self.N))
        ####

        y = np.ndarray((self.N))

        X_r = np.ndarray((self.N))
        X_e = np.ndarray((self.N))

        if X_r_init is not None:
            X_r[:] = X_r_init
        else:
            X_r[:] = np.random.normal(0.,1.,(self.N))
        #X_e[:] = self.w_in @ u_in[0,:]
        if mode == 'real_input':
            X_e = self.w_in @ u_in[0,:]
        else:
            X_e = np.random.normal(0.,sigm_e,(self.N))

        y = self.f(self.a_r * X_r + X_e - self.b)

        #### Assign for t=0
        y_rec[0,:] = y
        X_r_rec[0,:] = X_r
        X_e_rec[0,:] = X_e
        ####

        for t in tqdm(range(1,T),disable=not(show_progress)):

            X_r = self.W @ y
            if mode == 'real_input':
                X_e = self.w_in @ u_in[t,:]
            else:
                X_e = np.random.normal(0.,sigm_e,(self.N))

            y = self.f(self.a_r * X_r + X_e - self.b)


            #### record
            if t%T_skip_rec == 0:

                t_rec = int(t/T_skip_rec)

                y_rec[t_rec,:] = y
                X_r_rec[t_rec,:] = X_r
                X_e_rec[t_rec,:] = X_e


        return y_rec, X_r_rec, X_e_rec
    
    def get_R_a(self):
        
        return np.abs(np.linalg.eigvals(self.a_r * self.W.T)).max()