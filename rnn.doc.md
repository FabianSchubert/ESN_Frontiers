<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.0 Transitional//EN">
<html><head><title>Python: module src.rnn</title>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
</head><body bgcolor="#f0f0f8">

<table width="100%" cellspacing=0 cellpadding=2 border=0 summary="heading">
<tr bgcolor="#7799ee">
<td valign=bottom>&nbsp;<br>
<font color="#ffffff" face="helvetica, arial">&nbsp;<br><big><big><strong><a href="src.html"><font color="#ffffff">src</font></a>.rnn</strong></big></big></font></td
><td align=right valign=bottom
><font color="#ffffff" face="helvetica, arial"><a href=".">index</a><br><a href="file:/home/fabian/work/repositories/ESN_Frontiers/code/src/rnn.py">/home/fabian/work/repositories/ESN_Frontiers/code/src/rnn.py</a></font></td></tr></table>
    <p></p>
<p>
<table width="100%" cellspacing=0 cellpadding=2 border=0 summary="section">
<tr bgcolor="#aa55cc">
<td colspan=3 valign=bottom>&nbsp;<br>
<font color="#ffffff" face="helvetica, arial"><big><strong>Modules</strong></big></font></td></tr>
    
<tr><td bgcolor="#aa55cc"><tt>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</tt></td><td>&nbsp;</td>
<td width="100%"><table width="100%" summary="list"><tr><td width="25%" valign=top><a href="numpy.html">numpy</a><br>
</td><td width="25%" valign=top><a href="matplotlib.pyplot.html">matplotlib.pyplot</a><br>
</td><td width="25%" valign=top></td><td width="25%" valign=top></td></tr></table></td></tr></table><p>
<table width="100%" cellspacing=0 cellpadding=2 border=0 summary="section">
<tr bgcolor="#ee77aa">
<td colspan=3 valign=bottom>&nbsp;<br>
<font color="#ffffff" face="helvetica, arial"><big><strong>Classes</strong></big></font></td></tr>
    
<tr><td bgcolor="#ee77aa"><tt>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</tt></td><td>&nbsp;</td>
<td width="100%"><dl>
<dt><font face="helvetica, arial"><a href="builtins.html#object">builtins.object</a>
</font></dt><dd>
<dl>
<dt><font face="helvetica, arial"><a href="src.rnn.html#RNN">RNN</a>
</font></dt></dl>
</dd>
</dl>
 <p>
<table width="100%" cellspacing=0 cellpadding=2 border=0 summary="section">
<tr bgcolor="#ffc8d8">
<td colspan=3 valign=bottom>&nbsp;<br>
<font color="#000000" face="helvetica, arial"><a name="RNN">class <strong>RNN</strong></a>(<a href="builtins.html#object">builtins.object</a>)</font></td></tr>
    
<tr bgcolor="#ffc8d8"><td rowspan=2><tt>&nbsp;&nbsp;&nbsp;</tt></td>
<td colspan=2><tt><a href="#RNN">RNN</a>(**kwargs)<br>
&nbsp;<br>
A&nbsp;discrete&nbsp;time,&nbsp;rate-encoding&nbsp;<a href="#RNN">RNN</a>,&nbsp;implementing&nbsp;flow&nbsp;and&nbsp;variance&nbsp;control.<br>
&nbsp;<br>
Attributes<br>
----------<br>
N&nbsp;:&nbsp;int<br>
&nbsp;&nbsp;&nbsp;&nbsp;Number&nbsp;of&nbsp;nodes&nbsp;in&nbsp;the&nbsp;network<br>
cf&nbsp;:&nbsp;float<br>
&nbsp;&nbsp;&nbsp;&nbsp;The&nbsp;recurrent&nbsp;connection&nbsp;fraction<br>
cf_in&nbsp;:&nbsp;float<br>
&nbsp;&nbsp;&nbsp;&nbsp;The&nbsp;external&nbsp;input&nbsp;connection&nbsp;fraction<br>
a_r&nbsp;:&nbsp;float&nbsp;numpy&nbsp;array&nbsp;of&nbsp;shape&nbsp;(N)<br>
&nbsp;&nbsp;&nbsp;&nbsp;Postsynaptic&nbsp;weight&nbsp;scaling&nbsp;factors&nbsp;for&nbsp;the&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;recurrent&nbsp;membrane&nbsp;potential<br>
b&nbsp;:&nbsp;float&nbsp;numpy&nbsp;array&nbsp;of&nbsp;shape&nbsp;(N)<br>
&nbsp;&nbsp;&nbsp;&nbsp;Biases<br>
mu_w&nbsp;:&nbsp;float<br>
&nbsp;&nbsp;&nbsp;&nbsp;Mean&nbsp;of&nbsp;recurrent&nbsp;weights<br>
mu_w_in&nbsp;:&nbsp;float<br>
&nbsp;&nbsp;&nbsp;&nbsp;Mean&nbsp;of&nbsp;external&nbsp;weights<br>
dim_in&nbsp;:&nbsp;int<br>
&nbsp;&nbsp;&nbsp;&nbsp;Dimension&nbsp;of&nbsp;the&nbsp;input&nbsp;layer.&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;I.e.,&nbsp;the&nbsp;external&nbsp;input&nbsp;weight&nbsp;matrix&nbsp;has<br>
&nbsp;&nbsp;&nbsp;&nbsp;shape&nbsp;(N,dim_in).<br>
dim_out&nbsp;:&nbsp;int<br>
&nbsp;&nbsp;&nbsp;&nbsp;Dimension&nbsp;of&nbsp;the&nbsp;output&nbsp;layer.<br>
&nbsp;&nbsp;&nbsp;&nbsp;I.e.,&nbsp;the&nbsp;output&nbsp;weight&nbsp;matrix&nbsp;has<br>
&nbsp;&nbsp;&nbsp;&nbsp;shape&nbsp;(dim_out,N&nbsp;+&nbsp;1).&nbsp;The&nbsp;additional&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;dimension&nbsp;accounts&nbsp;for&nbsp;a&nbsp;potential&nbsp;bias&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;in&nbsp;the&nbsp;output.<br>
eps_a_r&nbsp;:&nbsp;float&nbsp;numpy&nbsp;array&nbsp;of&nbsp;shape&nbsp;(N)<br>
&nbsp;&nbsp;&nbsp;&nbsp;Local&nbsp;homeostatic&nbsp;adaptation&nbsp;rates&nbsp;for&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;weight&nbsp;scaling&nbsp;factors&nbsp;a_r.<br>
eps_b&nbsp;:&nbsp;float&nbsp;numpy&nbsp;array&nbsp;of&nbsp;shape&nbsp;(N)<br>
&nbsp;&nbsp;&nbsp;&nbsp;Local&nbsp;homeostatic&nbsp;adaptation&nbsp;rates&nbsp;for&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;biases&nbsp;b.<br>
eps_y_mean&nbsp;:&nbsp;float&nbsp;numpy&nbsp;array&nbsp;of&nbsp;shape&nbsp;(N)<br>
&nbsp;&nbsp;&nbsp;&nbsp;Local&nbsp;adaptation&nbsp;rates&nbsp;for&nbsp;trailing&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;averages&nbsp;of&nbsp;the&nbsp;mean&nbsp;activity&nbsp;y.<br>
eps_y_std&nbsp;:&nbsp;float&nbsp;numpy&nbsp;array&nbsp;of&nbsp;shape&nbsp;(N)<br>
&nbsp;&nbsp;&nbsp;&nbsp;Local&nbsp;adaptation&nbsp;rates&nbsp;for&nbsp;trailing&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;averages&nbsp;of&nbsp;the&nbsp;standard&nbsp;devation&nbsp;of&nbsp;the&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;activitiy&nbsp;y.<br>
eps_E_mean:&nbsp;float&nbsp;numpy&nbsp;array&nbsp;of&nbsp;shape&nbsp;(N)<br>
&nbsp;&nbsp;&nbsp;&nbsp;Local&nbsp;adaptation&nbsp;rates&nbsp;for&nbsp;trailing&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;averages&nbsp;of&nbsp;the&nbsp;external&nbsp;input.<br>
eps_E_std:&nbsp;float&nbsp;numpy&nbsp;array&nbsp;of&nbsp;shape&nbsp;(N)<br>
&nbsp;&nbsp;&nbsp;&nbsp;Local&nbsp;adaptation&nbsp;rates&nbsp;for&nbsp;trailing&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;averages&nbsp;of&nbsp;the&nbsp;standard&nbsp;devation&nbsp;of&nbsp;the&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;external&nbsp;input.<br>
y_mean_target&nbsp;:&nbsp;float&nbsp;numpy&nbsp;array&nbsp;of&nbsp;shape&nbsp;(N)<br>
&nbsp;&nbsp;&nbsp;&nbsp;Local&nbsp;homeostatic&nbsp;targets&nbsp;for&nbsp;the&nbsp;mean&nbsp;activity.<br>
y_std_target&nbsp;:&nbsp;float&nbsp;numpy&nbsp;array&nbsp;of&nbsp;shape&nbsp;(N)<br>
&nbsp;&nbsp;&nbsp;&nbsp;Local&nbsp;homeostatic&nbsp;targets&nbsp;for&nbsp;the&nbsp;standard&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;deviation&nbsp;of&nbsp;activity.<br>
R_target&nbsp;:&nbsp;float<br>
&nbsp;&nbsp;&nbsp;&nbsp;Homeostatic&nbsp;target&nbsp;for&nbsp;the&nbsp;spectral&nbsp;radius&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;of&nbsp;the&nbsp;effective&nbsp;weight&nbsp;matrix&nbsp;a_r_i&nbsp;*&nbsp;W_ij.<br>
W&nbsp;:&nbsp;float&nbsp;numpy&nbsp;array&nbsp;of&nbsp;shape&nbsp;(N,N)<br>
&nbsp;&nbsp;&nbsp;&nbsp;The&nbsp;bare&nbsp;recurrent&nbsp;weight&nbsp;matrix&nbsp;with&nbsp;a&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;connection&nbsp;probability&nbsp;cf&nbsp;and&nbsp;weights&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;drawn&nbsp;randomly&nbsp;from&nbsp;a&nbsp;Gaussian&nbsp;distribution&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;with&nbsp;mean&nbsp;mu_w&nbsp;and&nbsp;a&nbsp;standard&nbsp;deviation&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;of&nbsp;1/sqrt(N*cf).&nbsp;The&nbsp;diagonal&nbsp;is&nbsp;always&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;set&nbsp;to&nbsp;zero&nbsp;(no&nbsp;autapses).<br>
w_in&nbsp;:&nbsp;float&nbsp;numpy&nbsp;array&nbsp;of&nbsp;shape&nbsp;(N,dim_in)<br>
&nbsp;&nbsp;&nbsp;&nbsp;External&nbsp;input&nbsp;weight&nbsp;matrix&nbsp;with&nbsp;a&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;connection&nbsp;probability&nbsp;cf_in.&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;Upon&nbsp;initialization,&nbsp;weights&nbsp;are&nbsp;drawn&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;from&nbsp;a&nbsp;Gaussian&nbsp;distribution&nbsp;with&nbsp;mean&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;mu_w_in&nbsp;and&nbsp;a&nbsp;standard&nbsp;devation&nbsp;of&nbsp;1.<br>
w_out&nbsp;:&nbsp;float&nbsp;numpy&nbsp;array&nbsp;of&nbsp;shape&nbsp;(dim_out,N+1)<br>
&nbsp;&nbsp;&nbsp;&nbsp;Output&nbsp;weight&nbsp;matrix.&nbsp;Initialized&nbsp;with&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;uniform&nbsp;random&nbsp;weights.&nbsp;Note&nbsp;that&nbsp;the&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;additional&nbsp;dimension&nbsp;accounts&nbsp;for&nbsp;a&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;potential&nbsp;bias&nbsp;in&nbsp;the&nbsp;output.<br>
&nbsp;<br>
Methods<br>
-------<br>
<a href="#RNN-f">f</a>(x)<br>
&nbsp;&nbsp;&nbsp;&nbsp;The&nbsp;nonlinear&nbsp;activiation&nbsp;function.<br>
<a href="#RNN-df_x">df_x</a>(x)<br>
&nbsp;&nbsp;&nbsp;&nbsp;First&nbsp;derivative&nbsp;of&nbsp;<a href="#RNN-f">f</a>(x)&nbsp;as&nbsp;a&nbsp;function&nbsp;of&nbsp;x.<br>
<a href="#RNN-df_y">df_y</a>(y):<br>
&nbsp;&nbsp;&nbsp;&nbsp;First&nbsp;derivative&nbsp;of&nbsp;<a href="#RNN-f">f</a>(x),&nbsp;expressed&nbsp;as&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;a&nbsp;function&nbsp;of&nbsp;y&nbsp;=&nbsp;<a href="#RNN-f">f</a>(x).<br>
<a href="#RNN-check_data_in_comp">check_data_in_comp</a>(data)<br>
&nbsp;&nbsp;&nbsp;&nbsp;Checks&nbsp;if&nbsp;the&nbsp;array&nbsp;data&nbsp;could&nbsp;serve&nbsp;as<br>
&nbsp;&nbsp;&nbsp;&nbsp;an&nbsp;input&nbsp;sequence,&nbsp;given&nbsp;dim_in<br>
<a href="#RNN-check_data_out_comp">check_data_out_comp</a>(data)<br>
&nbsp;&nbsp;&nbsp;&nbsp;Checks&nbsp;if&nbsp;the&nbsp;array&nbsp;data&nbsp;is&nbsp;compatible<br>
&nbsp;&nbsp;&nbsp;&nbsp;with&nbsp;the&nbsp;dimensionality&nbsp;of&nbsp;the&nbsp;output<br>
&nbsp;&nbsp;&nbsp;&nbsp;layer&nbsp;dim_out.<br>
<a href="#RNN-learn_w_out_trial">learn_w_out_trial</a>(u_in,u_target,reg_fact)<br>
&nbsp;&nbsp;&nbsp;&nbsp;Least&nbsp;squares&nbsp;batch&nbsp;training&nbsp;of&nbsp;w_out&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;using&nbsp;an&nbsp;input&nbsp;sequence&nbsp;u_in&nbsp;and&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;a&nbsp;target&nbsp;output&nbsp;u_target.&nbsp;Thikonov&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;regularization&nbsp;is&nbsp;used&nbsp;with&nbsp;a&nbsp;regularization&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;factor&nbsp;reg_fact.<br>
<a href="#RNN-predict_data">predict_data</a>(data)<br>
&nbsp;&nbsp;&nbsp;&nbsp;Create&nbsp;an&nbsp;output&nbsp;sequence&nbsp;for&nbsp;a&nbsp;given<br>
&nbsp;&nbsp;&nbsp;&nbsp;input&nbsp;sequence&nbsp;passed&nbsp;by&nbsp;data.<br>
&nbsp;&nbsp;&nbsp;&nbsp;Obviously,&nbsp;this&nbsp;function&nbsp;should&nbsp;be&nbsp;called<br>
&nbsp;&nbsp;&nbsp;&nbsp;after&nbsp;training&nbsp;w_out&nbsp;using&nbsp;the&nbsp;method<br>
&nbsp;&nbsp;&nbsp;&nbsp;learn_w_out_trial.<br>
<a href="#RNN-run_hom_adapt_var_fix">run_hom_adapt_var_fix</a>(u_in)<br>
&nbsp;&nbsp;&nbsp;&nbsp;Run&nbsp;homeostatic&nbsp;adaptation&nbsp;on&nbsp;a_r&nbsp;and&nbsp;b<br>
&nbsp;&nbsp;&nbsp;&nbsp;under&nbsp;an&nbsp;input&nbsp;sequence&nbsp;u_in,&nbsp;using&nbsp;the<br>
&nbsp;&nbsp;&nbsp;&nbsp;networks'&nbsp;homeostatic&nbsp;targets&nbsp;y_mean_target<br>
&nbsp;&nbsp;&nbsp;&nbsp;and&nbsp;y_std_target.<br>
<a href="#RNN-run_hom_adapt">run_hom_adapt</a>(adapt_rule,u_in)<br>
&nbsp;&nbsp;&nbsp;&nbsp;Run&nbsp;either&nbsp;flow&nbsp;control&nbsp;(by&nbsp;passing&nbsp;"flow"&nbsp;via&nbsp;adapt_rule)<br>
&nbsp;&nbsp;&nbsp;&nbsp;or&nbsp;variance&nbsp;control&nbsp;(by&nbsp;passing&nbsp;"variance"&nbsp;via&nbsp;adapt_rule)<br>
&nbsp;&nbsp;&nbsp;&nbsp;under&nbsp;an&nbsp;input_sequence&nbsp;u_in,&nbsp;using&nbsp;the&nbsp;networks'<br>
&nbsp;&nbsp;&nbsp;&nbsp;homeostatic&nbsp;target&nbsp;R_target.<br>
<a href="#RNN-run_sample">run_sample</a>(u_in)<br>
&nbsp;&nbsp;&nbsp;&nbsp;Return&nbsp;recordings&nbsp;of&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;neural&nbsp;activity&nbsp;y,&nbsp;the&nbsp;recurrent&nbsp;contribution&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;to&nbsp;the&nbsp;membrane&nbsp;potential&nbsp;X_r,&nbsp;and&nbsp;the&nbsp;external&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;contribution&nbsp;to&nbsp;the&nbsp;membrane&nbsp;potential&nbsp;X_e,&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;under&nbsp;an&nbsp;external&nbsp;input&nbsp;sequence&nbsp;u_in<br>&nbsp;</tt></td></tr>
<tr><td>&nbsp;</td>
<td width="100%">Methods defined here:<br>
<dl><dt><a name="RNN-__init__"><strong>__init__</strong></a>(self, **kwargs)</dt><dd><tt>Initialize&nbsp;self.&nbsp;&nbsp;See&nbsp;help(type(self))&nbsp;for&nbsp;accurate&nbsp;signature.</tt></dd></dl>

<dl><dt><a name="RNN-check_data_in_comp"><strong>check_data_in_comp</strong></a>(self, data)</dt><dd><tt>Check&nbsp;if&nbsp;data&nbsp;has&nbsp;the&nbsp;appropriate<br>
shape&nbsp;to&nbsp;serve&nbsp;as&nbsp;an&nbsp;input&nbsp;sequence:<br>
&nbsp;&nbsp;&nbsp;&nbsp;1.&nbsp;&nbsp;If&nbsp;dim_in&nbsp;=&nbsp;1,&nbsp;data&nbsp;may&nbsp;either<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;have&nbsp;shape&nbsp;(T)&nbsp;or&nbsp;(T,1),&nbsp;where<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;T&nbsp;is&nbsp;the&nbsp;number&nbsp;of&nbsp;time&nbsp;steps<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;in&nbsp;the&nbsp;sequence.<br>
&nbsp;&nbsp;&nbsp;&nbsp;2.&nbsp;&nbsp;If&nbsp;dim_in&nbsp;&gt;&nbsp;1,&nbsp;data&nbsp;must&nbsp;have<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;shape&nbsp;(T,dim_in).</tt></dd></dl>

<dl><dt><a name="RNN-check_data_out_comp"><strong>check_data_out_comp</strong></a>(self, data)</dt><dd><tt>Check&nbsp;if&nbsp;data&nbsp;has&nbsp;a<br>
shape&nbsp;that&nbsp;fits&nbsp;the&nbsp;dimension&nbsp;of<br>
the&nbsp;output&nbsp;layer:<br>
&nbsp;&nbsp;&nbsp;&nbsp;1.&nbsp;&nbsp;If&nbsp;dim_out&nbsp;=&nbsp;1,&nbsp;data&nbsp;may&nbsp;either<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;have&nbsp;shape&nbsp;(T)&nbsp;or&nbsp;(T,1),&nbsp;where<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;T&nbsp;is&nbsp;the&nbsp;number&nbsp;of&nbsp;time&nbsp;steps<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;in&nbsp;the&nbsp;sequence.<br>
&nbsp;&nbsp;&nbsp;&nbsp;2.&nbsp;&nbsp;If&nbsp;dim_out&nbsp;&gt;&nbsp;1,&nbsp;data&nbsp;must&nbsp;have<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;shape&nbsp;(T,dim_out).</tt></dd></dl>

<dl><dt><a name="RNN-df_x"><strong>df_x</strong></a>(self, x)</dt><dd><tt>The&nbsp;derivative&nbsp;df(x)/dx&nbsp;as&nbsp;a&nbsp;function<br>
of&nbsp;the&nbsp;membrane&nbsp;potential&nbsp;x.<br>
Default&nbsp;is&nbsp;df(x)/dx&nbsp;=&nbsp;1&nbsp;-&nbsp;tanh(x)^2.<br>
Note&nbsp;that&nbsp;manually&nbsp;changing&nbsp;<a href="#RNN-f">f</a>(x)&nbsp;does&nbsp;<br>
note&nbsp;automatically&nbsp;override&nbsp;this&nbsp;function.</tt></dd></dl>

<dl><dt><a name="RNN-df_y"><strong>df_y</strong></a>(self, y)</dt><dd><tt>The&nbsp;derivative&nbsp;df(x)/dx&nbsp;as&nbsp;a&nbsp;function<br>
of&nbsp;the&nbsp;activity&nbsp;itself.<br>
Default&nbsp;is&nbsp;(df(x)/dx)(y)&nbsp;=&nbsp;1&nbsp;-&nbsp;y^2.<br>
Note&nbsp;that&nbsp;manually&nbsp;changing&nbsp;<a href="#RNN-f">f</a>(x)&nbsp;does&nbsp;<br>
note&nbsp;automatically&nbsp;override&nbsp;this&nbsp;function.</tt></dd></dl>

<dl><dt><a name="RNN-f"><strong>f</strong></a>(self, x)</dt><dd><tt>The&nbsp;neural&nbsp;transfer&nbsp;function&nbsp;as&nbsp;a&nbsp;function<br>
of&nbsp;the&nbsp;membrane&nbsp;potential&nbsp;x.<br>
Default&nbsp;is&nbsp;<a href="#RNN-f">f</a>(x)&nbsp;=&nbsp;tanh(x).</tt></dd></dl>

<dl><dt><a name="RNN-get_R_a"><strong>get_R_a</strong></a>(self)</dt><dd><tt>Returns&nbsp;the&nbsp;spectral&nbsp;radius&nbsp;of&nbsp;the&nbsp;effective<br>
recurrent&nbsp;weight&nbsp;matrix&nbsp;a_r_i&nbsp;*&nbsp;W_ij.</tt></dd></dl>

<dl><dt><a name="RNN-learn_w_out_trial"><strong>learn_w_out_trial</strong></a>(self, u_in, u_target, reg_fact=0.01, show_progress=False, T_prerun=0)</dt><dd><tt>This&nbsp;function&nbsp;does&nbsp;linear&nbsp;least&nbsp;squares&nbsp;regression&nbsp;using&nbsp;the<br>
neural&nbsp;activities&nbsp;acquired&nbsp;from&nbsp;running&nbsp;the&nbsp;network<br>
with&nbsp;an&nbsp;input&nbsp;sequence&nbsp;u_in,&nbsp;and&nbsp;an&nbsp;output&nbsp;target&nbsp;<br>
sequence&nbsp;u_target.&nbsp;Thikonov&nbsp;regularization&nbsp;is&nbsp;used.<br>
&nbsp;<br>
Optional&nbsp;parameters:<br>
&nbsp;&nbsp;&nbsp;&nbsp;reg_fact&nbsp;:&nbsp;float,&nbsp;default&nbsp;=&nbsp;0.01<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Set&nbsp;the&nbsp;regularization&nbsp;factor<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;of&nbsp;the&nbsp;linear&nbsp;regression.<br>
&nbsp;&nbsp;&nbsp;&nbsp;show_progress&nbsp;:&nbsp;bool,&nbsp;default&nbsp;=&nbsp;False<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Specifies&nbsp;whether&nbsp;to&nbsp;show&nbsp;a&nbsp;progress<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;bar&nbsp;when&nbsp;running&nbsp;the&nbsp;network&nbsp;simulation.<br>
&nbsp;&nbsp;&nbsp;&nbsp;T_prerun&nbsp;:&nbsp;int,&nbsp;default&nbsp;=&nbsp;0<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Optionally,&nbsp;the&nbsp;first&nbsp;T_prerun&nbsp;time<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;steps&nbsp;can&nbsp;be&nbsp;omitted&nbsp;for&nbsp;the&nbsp;linear<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;regression.</tt></dd></dl>

<dl><dt><a name="RNN-predict_data"><strong>predict_data</strong></a>(self, data, return_reservoir_rec=False, show_progress=True)</dt><dd><tt>Generate&nbsp;an&nbsp;output&nbsp;sequence&nbsp;<br>
&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;u_out_i(t)&nbsp;=&nbsp;sum_j=1^N&nbsp;w_out_ij&nbsp;y_j(t)<br>
&nbsp;<br>
under&nbsp;a&nbsp;given&nbsp;input<br>
sequence&nbsp;given&nbsp;by&nbsp;data.<br>
&nbsp;<br>
Optional&nbsp;parameters:<br>
&nbsp;&nbsp;&nbsp;&nbsp;return_reservoir_rec&nbsp;:&nbsp;bool,&nbsp;default&nbsp;=&nbsp;False<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If&nbsp;True,&nbsp;the&nbsp;function&nbsp;returns<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(out,y),&nbsp;where&nbsp;out&nbsp;is&nbsp;the&nbsp;generated&nbsp;output<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;sequence&nbsp;and&nbsp;y&nbsp;is&nbsp;the&nbsp;recorded&nbsp;neural&nbsp;activity.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If&nbsp;False,&nbsp;only&nbsp;out&nbsp;is&nbsp;returned.<br>
&nbsp;&nbsp;&nbsp;&nbsp;show_progress&nbsp;:&nbsp;bool,&nbsp;default&nbsp;=&nbsp;True<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Specifies&nbsp;whether&nbsp;to&nbsp;show&nbsp;a&nbsp;progress<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;bar&nbsp;when&nbsp;running&nbsp;the&nbsp;network&nbsp;simulation.</tt></dd></dl>

<dl><dt><a name="RNN-run_hom_adapt"><strong>run_hom_adapt</strong></a>(self, adapt_rule, u_in=None, sigm_e=1.0, T_skip_rec=1, T=None, adapt_mode='local', norm_flow=True, show_progress=True, y_init=None)</dt><dd><tt>Runs&nbsp;homeostatic&nbsp;adaptation&nbsp;on&nbsp;a_r&nbsp;and&nbsp;b&nbsp;using&nbsp;flow<br>
control&nbsp;(pass&nbsp;"flow"&nbsp;via&nbsp;the&nbsp;adapt_rule&nbsp;parameter)&nbsp;or&nbsp;variance<br>
control&nbsp;(pass&nbsp;"variance").<br>
&nbsp;<br>
There&nbsp;are&nbsp;two&nbsp;options&nbsp;for&nbsp;the&nbsp;type&nbsp;of&nbsp;external&nbsp;input&nbsp;used&nbsp;<br>
during&nbsp;adaptation:<br>
&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;1.&nbsp;&nbsp;An&nbsp;input&nbsp;sequence&nbsp;u_in&nbsp;is&nbsp;passed,&nbsp;which<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;is&nbsp;projected&nbsp;onto&nbsp;the&nbsp;network&nbsp;using&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;w_in&nbsp;for&nbsp;each&nbsp;time&nbsp;step.&nbsp;The&nbsp;run&nbsp;time&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;of&nbsp;the&nbsp;simulation&nbsp;is&nbsp;determined&nbsp;by&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the&nbsp;length&nbsp;of&nbsp;the&nbsp;input&nbsp;sequence.<br>
&nbsp;&nbsp;&nbsp;&nbsp;2.&nbsp;&nbsp;The&nbsp;optional&nbsp;parameters&nbsp;T&nbsp;and&nbsp;sigm_e&nbsp;are<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;passed&nbsp;instead.&nbsp;Then,&nbsp;each&nbsp;node&nbsp;receives<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;independent&nbsp;random&nbsp;Gaussian&nbsp;external&nbsp;input<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;with&nbsp;zero&nbsp;mean&nbsp;and&nbsp;a&nbsp;standard&nbsp;deviation&nbsp;of<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;sigm_e&nbsp;(can&nbsp;either&nbsp;be&nbsp;a&nbsp;scalar&nbsp;or&nbsp;an&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;array&nbsp;of&nbsp;size&nbsp;N).&nbsp;The&nbsp;run&nbsp;time&nbsp;is&nbsp;then&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;given&nbsp;by&nbsp;the&nbsp;parameter&nbsp;T.&nbsp;&nbsp;<br>
&nbsp;<br>
Optional&nbsp;parameters:<br>
&nbsp;&nbsp;&nbsp;&nbsp;T_skip_rec&nbsp;:&nbsp;int,&nbsp;default&nbsp;=&nbsp;1<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Temporal&nbsp;step&nbsp;size&nbsp;for&nbsp;recording<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;state&nbsp;variables&nbsp;of&nbsp;the&nbsp;system.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Reduces&nbsp;memory&nbsp;usage&nbsp;when&nbsp;running&nbsp;very<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;long&nbsp;simulations.&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;show_progress&nbsp;:&nbsp;bool,&nbsp;default&nbsp;=&nbsp;True<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Specifies&nbsp;whether&nbsp;to&nbsp;show&nbsp;a&nbsp;progress<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;bar&nbsp;when&nbsp;running&nbsp;the&nbsp;network&nbsp;simulation.</tt></dd></dl>

<dl><dt><a name="RNN-run_hom_adapt_var_fix"><strong>run_hom_adapt_var_fix</strong></a>(self, u_in=None, sigm_e=1.0, T_skip_rec=1, T=None, show_progress=True)</dt><dd><tt>Runs&nbsp;homeostatic&nbsp;adaptation&nbsp;on&nbsp;a_r&nbsp;and&nbsp;b&nbsp;using&nbsp;homeostatic<br>
targets&nbsp;y_mean_target&nbsp;and&nbsp;y_std_target.&nbsp;<br>
Local&nbsp;variables&nbsp;a_r_i&nbsp;and&nbsp;b_i&nbsp;are&nbsp;updated&nbsp;according&nbsp;to<br>
&nbsp;&nbsp;&nbsp;&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;a_r_i(t+1)&nbsp;=&nbsp;a_r_i(t)&nbsp;+&nbsp;eps_a_r&nbsp;*&nbsp;(y_std_target^2&nbsp;-&nbsp;(y(t)-y_av(t))^2)<br>
&nbsp;&nbsp;&nbsp;&nbsp;b_i(t+1)&nbsp;=&nbsp;b_i(t)&nbsp;+&nbsp;eps_b&nbsp;*&nbsp;(y_mean_target&nbsp;-&nbsp;y(t))<br>
&nbsp;&nbsp;&nbsp;&nbsp;<br>
where&nbsp;y_av(t)&nbsp;is&nbsp;a&nbsp;running&nbsp;average&nbsp;of&nbsp;y(t).<br>
&nbsp;<br>
There&nbsp;are&nbsp;two&nbsp;options&nbsp;for&nbsp;the&nbsp;type&nbsp;of&nbsp;external&nbsp;input&nbsp;used&nbsp;<br>
during&nbsp;adaptation:<br>
&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;1.&nbsp;&nbsp;An&nbsp;input&nbsp;sequence&nbsp;u_in&nbsp;is&nbsp;passed,&nbsp;which<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;is&nbsp;projected&nbsp;onto&nbsp;the&nbsp;network&nbsp;using&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;w_in&nbsp;for&nbsp;each&nbsp;time&nbsp;step.&nbsp;The&nbsp;run&nbsp;time&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;of&nbsp;the&nbsp;simulation&nbsp;is&nbsp;determined&nbsp;by&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the&nbsp;length&nbsp;of&nbsp;the&nbsp;input&nbsp;sequence.<br>
&nbsp;&nbsp;&nbsp;&nbsp;2.&nbsp;&nbsp;The&nbsp;optional&nbsp;parameters&nbsp;T&nbsp;and&nbsp;sigm_e&nbsp;are<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;passed&nbsp;instead.&nbsp;Then,&nbsp;each&nbsp;node&nbsp;receives<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;independent&nbsp;random&nbsp;Gaussian&nbsp;external&nbsp;input<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;with&nbsp;zero&nbsp;mean&nbsp;and&nbsp;a&nbsp;standard&nbsp;deviation&nbsp;of<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;sigm_e&nbsp;(can&nbsp;either&nbsp;be&nbsp;a&nbsp;scalar&nbsp;or&nbsp;an&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;array&nbsp;of&nbsp;size&nbsp;N).&nbsp;The&nbsp;run&nbsp;time&nbsp;is&nbsp;then&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;given&nbsp;by&nbsp;the&nbsp;parameter&nbsp;T.&nbsp;&nbsp;<br>
&nbsp;<br>
Optional&nbsp;parameters:<br>
&nbsp;&nbsp;&nbsp;&nbsp;T_skip_rec&nbsp;:&nbsp;int,&nbsp;default&nbsp;=&nbsp;1<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Temporal&nbsp;step&nbsp;size&nbsp;for&nbsp;recording<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;state&nbsp;variables&nbsp;of&nbsp;the&nbsp;system.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Reduces&nbsp;memory&nbsp;usage&nbsp;when&nbsp;running&nbsp;very<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;long&nbsp;simulations.&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;show_progress&nbsp;:&nbsp;bool,&nbsp;default&nbsp;=&nbsp;True<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Specifies&nbsp;whether&nbsp;to&nbsp;show&nbsp;a&nbsp;progress<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;bar&nbsp;when&nbsp;running&nbsp;the&nbsp;network&nbsp;simulation.</tt></dd></dl>

<dl><dt><a name="RNN-run_sample"><strong>run_sample</strong></a>(self, u_in=None, sigm_e=1.0, X_r_init=None, T_skip_rec=1, T=None, show_progress=True)</dt><dd><tt>Runs&nbsp;a&nbsp;sample&nbsp;simulation&nbsp;without&nbsp;adaptation&nbsp;and&nbsp;returns<br>
the&nbsp;recorded&nbsp;neural&nbsp;activity&nbsp;y,&nbsp;the&nbsp;recurrent&nbsp;contribution<br>
to&nbsp;the&nbsp;membrane&nbsp;potential&nbsp;X_r&nbsp;and&nbsp;the&nbsp;external<br>
contribution&nbsp;to&nbsp;the&nbsp;membrane&nbsp;potential&nbsp;X_e.<br>
&nbsp;<br>
There&nbsp;are&nbsp;two&nbsp;options&nbsp;for&nbsp;the&nbsp;type&nbsp;of&nbsp;external&nbsp;input&nbsp;used&nbsp;<br>
during&nbsp;adaptation:<br>
&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;1.&nbsp;&nbsp;An&nbsp;input&nbsp;sequence&nbsp;u_in&nbsp;is&nbsp;passed,&nbsp;which<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;is&nbsp;projected&nbsp;onto&nbsp;the&nbsp;network&nbsp;using&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;w_in&nbsp;for&nbsp;each&nbsp;time&nbsp;step.&nbsp;The&nbsp;run&nbsp;time&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;of&nbsp;the&nbsp;simulation&nbsp;is&nbsp;determined&nbsp;by&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the&nbsp;length&nbsp;of&nbsp;the&nbsp;input&nbsp;sequence.<br>
&nbsp;&nbsp;&nbsp;&nbsp;2.&nbsp;&nbsp;The&nbsp;optional&nbsp;parameters&nbsp;T&nbsp;and&nbsp;sigm_e&nbsp;are<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;passed&nbsp;instead.&nbsp;Then,&nbsp;each&nbsp;node&nbsp;receives<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;independent&nbsp;random&nbsp;Gaussian&nbsp;external&nbsp;input<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;with&nbsp;zero&nbsp;mean&nbsp;and&nbsp;a&nbsp;standard&nbsp;deviation&nbsp;of<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;sigm_e&nbsp;(can&nbsp;either&nbsp;be&nbsp;a&nbsp;scalar&nbsp;or&nbsp;an&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;array&nbsp;of&nbsp;size&nbsp;N).&nbsp;The&nbsp;run&nbsp;time&nbsp;is&nbsp;then&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;given&nbsp;by&nbsp;the&nbsp;parameter&nbsp;T.&nbsp;&nbsp;<br>
&nbsp;<br>
Optional&nbsp;parameters:<br>
&nbsp;&nbsp;&nbsp;&nbsp;X_r_init&nbsp;:&nbsp;numpy&nbsp;float&nbsp;array&nbsp;of&nbsp;size&nbsp;(N)<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Allows&nbsp;setting&nbsp;the&nbsp;initial&nbsp;recurrent&nbsp;input<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;to&nbsp;a&nbsp;certain&nbsp;set&nbsp;of&nbsp;values.&nbsp;If&nbsp;not&nbsp;specified,<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;X_r&nbsp;is&nbsp;randomly&nbsp;initialized.<br>
&nbsp;&nbsp;&nbsp;&nbsp;T_skip_rec&nbsp;:&nbsp;int,&nbsp;default&nbsp;=&nbsp;1<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Temporal&nbsp;step&nbsp;size&nbsp;for&nbsp;recording<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;state&nbsp;variables&nbsp;of&nbsp;the&nbsp;system.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Reduces&nbsp;memory&nbsp;usage&nbsp;when&nbsp;running&nbsp;very<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;long&nbsp;simulations.&nbsp;<br>
&nbsp;&nbsp;&nbsp;&nbsp;show_progress&nbsp;:&nbsp;bool,&nbsp;default&nbsp;=&nbsp;True<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Specifies&nbsp;whether&nbsp;to&nbsp;show&nbsp;a&nbsp;progress<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;bar&nbsp;when&nbsp;running&nbsp;the&nbsp;network&nbsp;simulation.</tt></dd></dl>

<hr>
Data descriptors defined here:<br>
<dl><dt><strong>__dict__</strong></dt>
<dd><tt>dictionary&nbsp;for&nbsp;instance&nbsp;variables&nbsp;(if&nbsp;defined)</tt></dd>
</dl>
<dl><dt><strong>__weakref__</strong></dt>
<dd><tt>list&nbsp;of&nbsp;weak&nbsp;references&nbsp;to&nbsp;the&nbsp;object&nbsp;(if&nbsp;defined)</tt></dd>
</dl>
</td></tr></table></td></tr></table>
</body></html>