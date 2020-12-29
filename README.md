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
after initialization, the function run_hom_adapt() starts
a simulation run with homeostatic adaptation, either using
flow control (passing "flow" to the function) or variance
control (passing "variance" to the function).

For further details, please check the <a href="rnn.doc.md">Documentation
of the RNN class</a>.

Code for all figures can be found in the folder Figures.
Note that the respective run_sim.py and plot.py scripts should
be called from the code base folder using 

"python3 -m Figures.\<figure name\>.run_sim" or

"python3 -m Figures.\<figure name\>.plot".

Note that some simulations take a long to run.



