# Local Regulation of the Spectral Radius of Echo State Networks

This repository contains an implementation of our local
homeostatic adaptation rules, termed "flow control" and
"variance control" for locally controlling the spectral 
radius of echo states.

## Usage

The following python packages need to installed:

  - NumPy
  - Matplotlib
  - Seaborn
  - tqdm

The network class RNN can be found in the src.rnn module.
After initialization, the function run_hom_adapt() starts
a simulation run with homeostatic adaptation, either using
flow control (passing "flow" to the function) or variance
control (passing "variance" to the function).

For further details, please check the 
<a href="https://itp.uni-frankfurt.de/~fschubert/rnn_doc/">
Documentation of the RNN class</a>.

Code for all figures can be found in the folder Figures.
Each folder contains a run_sim.py file that has to be run first,
generating the data for the respective figure. 
The plot is then generated using plot.py 
in the same folder.
Note that the respective run_sim.py and plot.py scripts should
be called from the code base folder using 

"python3 -m Figures.\<figure name\>.run_sim" and 

"python3 -m Figures.\<figure name\>.plot".

Note that some simulations may take a long time to run.



