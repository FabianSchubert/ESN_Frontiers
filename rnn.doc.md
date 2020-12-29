<table summary="heading" width="100%" cellspacing="0" cellpadding="2" border="0">

<tbody>

<tr bgcolor="#7799ee">

<td valign="bottom">   
<font face="helvetica, arial" color="#ffffff">   
<big><big>**[<font color="#ffffff">src</font>](src.html).rnn**</big></big></font></td>

<td valign="bottom" align="right"><font face="helvetica, arial" color="#ffffff">[index](.)  
[/home/fabian/work/repositories/ESN_Frontiers/code/src/rnn.py](file:/home/fabian/work/repositories/ESN_Frontiers/code/src/rnn.py)</font></td>

</tr>

</tbody>

</table>

<table summary="section" width="100%" cellspacing="0" cellpadding="2" border="0">

<tbody>

<tr bgcolor="#aa55cc">

<td colspan="3" valign="bottom">   
<font face="helvetica, arial" color="#ffffff"><big>**Modules**</big></font></td>

</tr>

<tr>

<td bgcolor="#aa55cc"></td>

<td> </td>

<td width="100%">

<table summary="list" width="100%">

<tbody>

<tr>

<td width="25%" valign="top">[numpy](numpy.html)  
</td>

<td width="25%" valign="top">[matplotlib.pyplot](matplotlib.pyplot.html)  
</td>

<td width="25%" valign="top"></td>

<td width="25%" valign="top"></td>

</tr>

</tbody>

</table>

</td>

</tr>

</tbody>

</table>

<table summary="section" width="100%" cellspacing="0" cellpadding="2" border="0">

<tbody>

<tr bgcolor="#ee77aa">

<td colspan="3" valign="bottom">   
<font face="helvetica, arial" color="#ffffff"><big>**Classes**</big></font></td>

</tr>

<tr>

<td bgcolor="#ee77aa"></td>

<td> </td>

<td width="100%">

<dl>

<dt><font face="helvetica, arial">[builtins.object](builtins.html#object)</font></dt>

<dd>

<dl>

<dt><font face="helvetica, arial">[RNN](src.rnn.html#RNN)</font></dt>

</dl>

</dd>

</dl>

<table summary="section" width="100%" cellspacing="0" cellpadding="2" border="0">

<tbody>

<tr bgcolor="#ffc8d8">

<td colspan="3" valign="bottom">   
<font face="helvetica, arial" color="#000000"><a name="RNN">class **RNN**</a>([builtins.object](builtins.html#object))</font></td>

</tr>

<tr bgcolor="#ffc8d8">

<td rowspan="2"></td>

<td colspan="2"><tt>[RNN](#RNN)(**kwargs)  

A discrete time, rate-encoding [RNN](#RNN), implementing flow and variance control.  

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
[f](#RNN-f)(x)  
    The nonlinear activiation function.  
[df_x](#RNN-df_x)(x)  
    First derivative of [f](#RNN-f)(x) as a function of x.  
[df_y](#RNN-df_y)(y):  
    First derivative of [f](#RNN-f)(x), expressed as   
    a function of y = [f](#RNN-f)(x).  
[check_data_in_comp](#RNN-check_data_in_comp)(data)  
    Checks if the array data could serve as  
    an input sequence, given dim_in  
[check_data_out_comp](#RNN-check_data_out_comp)(data)  
    Checks if the array data is compatible  
    with the dimensionality of the output  
    layer dim_out.  
[learn_w_out_trial](#RNN-learn_w_out_trial)(u_in,u_target,reg_fact)  
    Least squares batch training of w_out   
    using an input sequence u_in and   
    a target output u_target. Thikonov   
    regularization is used with a regularization   
    factor reg_fact.  
[predict_data](#RNN-predict_data)(data)  
    Create an output sequence for a given  
    input sequence passed by data.  
    Obviously, this function should be called  
    after training w_out using the method  
    learn_w_out_trial.  
[run_hom_adapt_var_fix](#RNN-run_hom_adapt_var_fix)(u_in)  
    Run homeostatic adaptation on a_r and b  
    under an input sequence u_in, using the  
    networks' homeostatic targets y_mean_target  
    and y_std_target.  
[run_hom_adapt](#RNN-run_hom_adapt)(adapt_rule,u_in)  
    Run either flow control (by passing "flow" via adapt_rule)  
    or variance control (by passing "variance" via adapt_rule)  
    under an input_sequence u_in, using the networks'  
    homeostatic target R_target.  
[run_sample](#RNN-run_sample)(u_in)  
    Return recordings of   
    neural activity y, the recurrent contribution   
    to the membrane potential X_r, and the external   
    contribution to the membrane potential X_e,   
    under an external input sequence u_in  
 </tt></td>

</tr>

<tr>

<td> </td>

<td width="100%">Methods defined here:  

<dl>

<dt><a name="RNN-__init__">**__init__**</a>(self, **kwargs)</dt>

<dd><tt>Initialize self.  See help(type(self)) for accurate signature.</tt></dd>

</dl>

<dl>

<dt><a name="RNN-check_data_in_comp">**check_data_in_comp**</a>(self, data)</dt>

<dd><tt>Check if data has the appropriate  
shape to serve as an input sequence:  
    1.  If dim_in = 1, data may either  
        have shape (T) or (T,1), where  
        T is the number of time steps  
        in the sequence.  
    2.  If dim_in > 1, data must have  
        shape (T,dim_in).</tt></dd>

</dl>

<dl>

<dt><a name="RNN-check_data_out_comp">**check_data_out_comp**</a>(self, data)</dt>

<dd><tt>Check if data has a  
shape that fits the dimension of  
the output layer:  
    1.  If dim_out = 1, data may either  
        have shape (T) or (T,1), where  
        T is the number of time steps  
        in the sequence.  
    2.  If dim_out > 1, data must have  
        shape (T,dim_out).</tt></dd>

</dl>

<dl>

<dt><a name="RNN-df_x">**df_x**</a>(self, x)</dt>

<dd><tt>The derivative df(x)/dx as a function  
of the membrane potential x.  
Default is df(x)/dx = 1 - tanh(x)^2.  
Note that manually changing [f](#RNN-f)(x) does   
note automatically override this function.</tt></dd>

</dl>

<dl>

<dt><a name="RNN-df_y">**df_y**</a>(self, y)</dt>

<dd><tt>The derivative df(x)/dx as a function  
of the activity itself.  
Default is (df(x)/dx)(y) = 1 - y^2.  
Note that manually changing [f](#RNN-f)(x) does   
note automatically override this function.</tt></dd>

</dl>

<dl>

<dt><a name="RNN-f">**f**</a>(self, x)</dt>

<dd><tt>The neural transfer function as a function  
of the membrane potential x.  
Default is [f](#RNN-f)(x) = tanh(x).</tt></dd>

</dl>

<dl>

<dt><a name="RNN-get_R_a">**get_R_a**</a>(self)</dt>

<dd><tt>Returns the spectral radius of the effective  
recurrent weight matrix a_r_i * W_ij.</tt></dd>

</dl>

<dl>

<dt><a name="RNN-learn_w_out_trial">**learn_w_out_trial**</a>(self, u_in, u_target, reg_fact=0.01, show_progress=False, T_prerun=0)</dt>

<dd><tt>This function does linear least squares regression using the  
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
        regression.</tt></dd>

</dl>

<dl>

<dt><a name="RNN-predict_data">**predict_data**</a>(self, data, return_reservoir_rec=False, show_progress=True)</dt>

<dd><tt>Generate an output sequence   

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
        bar when running the network simulation.</tt></dd>

</dl>

<dl>

<dt><a name="RNN-run_hom_adapt">**run_hom_adapt**</a>(self, adapt_rule, u_in=None, sigm_e=1.0, T_skip_rec=1, T=None, adapt_mode='local', norm_flow=True, show_progress=True, y_init=None)</dt>

<dd><tt>Runs homeostatic adaptation on a_r and b using flow  
control (pass "flow" via the adapt_rule parameter) or variance  
control (pass "variance").  

There are two options for the type of external input used   
during adaptation:  

    1.  An input sequence u_in is passed, which  
        is projected onto the network using   
        w_in for each time step. The run time   
        of the simulation is determined by   
        the length of the input sequence.  
    2.  The optional parameters T and sigm_e are  
        passed instead. Then, each node receives  
        independent random Gaussian external input  
        with zero mean and a standard deviation of  
        sigm_e (can either be a scalar or an   
        array of size N). The run time is then   
        given by the parameter T.    

Optional parameters:  
    T_skip_rec : int, default = 1  
        Temporal step size for recording  
        state variables of the system.  
        Reduces memory usage when running very  
        long simulations.   
    show_progress : bool, default = True  
        Specifies whether to show a progress  
        bar when running the network simulation.</tt></dd>

</dl>

<dl>

<dt><a name="RNN-run_hom_adapt_var_fix">**run_hom_adapt_var_fix**</a>(self, u_in=None, sigm_e=1.0, T_skip_rec=1, T=None, show_progress=True)</dt>

<dd><tt>Runs homeostatic adaptation on a_r and b using homeostatic  
targets y_mean_target and y_std_target.   
Local variables a_r_i and b_i are updated according to  

    a_r_i(t+1) = a_r_i(t) + eps_a_r * (y_std_target^2 - (y(t)-y_av(t))^2)  
    b_i(t+1) = b_i(t) + eps_b * (y_mean_target - y(t))  

where y_av(t) is a running average of y(t).  

There are two options for the type of external input used   
during adaptation:  

    1.  An input sequence u_in is passed, which  
        is projected onto the network using   
        w_in for each time step. The run time   
        of the simulation is determined by   
        the length of the input sequence.  
    2.  The optional parameters T and sigm_e are  
        passed instead. Then, each node receives  
        independent random Gaussian external input  
        with zero mean and a standard deviation of  
        sigm_e (can either be a scalar or an   
        array of size N). The run time is then   
        given by the parameter T.    

Optional parameters:  
    T_skip_rec : int, default = 1  
        Temporal step size for recording  
        state variables of the system.  
        Reduces memory usage when running very  
        long simulations.   
    show_progress : bool, default = True  
        Specifies whether to show a progress  
        bar when running the network simulation.</tt></dd>

</dl>

<dl>

<dt><a name="RNN-run_sample">**run_sample**</a>(self, u_in=None, sigm_e=1.0, X_r_init=None, T_skip_rec=1, T=None, show_progress=True)</dt>

<dd><tt>Runs a sample simulation without adaptation and returns  
the recorded neural activity y, the recurrent contribution  
to the membrane potential X_r and the external  
contribution to the membrane potential X_e.  

There are two options for the type of external input used   
during adaptation:  

    1.  An input sequence u_in is passed, which  
        is projected onto the network using   
        w_in for each time step. The run time   
        of the simulation is determined by   
        the length of the input sequence.  
    2.  The optional parameters T and sigm_e are  
        passed instead. Then, each node receives  
        independent random Gaussian external input  
        with zero mean and a standard deviation of  
        sigm_e (can either be a scalar or an   
        array of size N). The run time is then   
        given by the parameter T.    

Optional parameters:  
    X_r_init : numpy float array of size (N)  
        Allows setting the initial recurrent input  
        to a certain set of values. If not specified,  
        X_r is randomly initialized.  
    T_skip_rec : int, default = 1  
        Temporal step size for recording  
        state variables of the system.  
        Reduces memory usage when running very  
        long simulations.   
    show_progress : bool, default = True  
        Specifies whether to show a progress  
        bar when running the network simulation.</tt></dd>

</dl>

* * *

Data descriptors defined here:  

<dl>

<dt>**__dict__**</dt>

<dd><tt>dictionary for instance variables (if defined)</tt></dd>

</dl>

<dl>

<dt>**__weakref__**</dt>

<dd><tt>list of weak references to the object (if defined)</tt></dd>

</dl>

</td>

</tr>

</tbody>

</table>

</td>

</tr>

</tbody>

</table>
