========
PyDHAMed
========

DHAMed -Dynamic Histogram Analysis extended to detailed balance
===============================================================

Input are transition counts between states/bins and biases (if any).
Biases specify the differences in the potential energy functions in the different
simulation windows/runs.

To run DHAMed from a list of count matrices and an array specfying the
biases in each simulation (window) is required.

To see how DHAMed can be used to extract free energies from biased simulations
look at the example Jupyter notebook provided. 
https://github.com/bio-phys/PyDHAMed/blob/master/pydhamed/cg-rna/cg_RNA_duplex_formation.ipynb

Installation
============

To install PyDHAMed clone or download the repository

```bash
git clone https://github.com/bio-phys/PyDHAMed.git 
cd PyDHAMed
```
 
 Then install with pip

```bash
pip install . 
```

Inputs
======
    
The list of the individual count matrices C contain the transition counts
between the different states (or bins in umbrella sampling). C[i,j] where
i is the product state and j the reactent state. The first row contains
thus all the transitions into state 0.The first column C[:,0] all 
transition out of state 0.
    
The bias array contains a bias value for each state and for each simulation
(or window in umbrella sampling). The bias array has the shape N rows nwin 
columns and contains the bias acting on each state in each simulation (window).
The bias NEEDS to be given in units to kB_T.
    
Most parameters besides count_list and bias_ar are only relevant for testing
and further code developement. 
    

To run DHAMed:
==============

.. code:: python

  # import DHAMed functions 
  from optimize_dhamed import *
  from determine_transition_counts import count_matrix

  # determine transition counts for each trajectory
  # Each frame in a trajectory needs to be assigned to one of the the n states
  # of the system
  for traj in traj_list:
    count_list.append(count_matrix(traj, n_states=n))

  # Bias - need to specfiy the bias acting on each of the n states in the nwin simulation. 
  bias_ar = np.zeros((n, nwin))
  for i in range(n)
      bias_ar[i,:] = np.loadtxt("bias"+i)

  # run optimization
  og = run_dhamed(count_list, bias_ar)
 
DHAMed example:
===============

A Jupyter notebook with an example calculation is provided in the folder cg-RNA.
https://github.com/bio-phys/PyDHAMed/blob/master/pydhamed/cg-rna/cg_RNA_duplex_formation.ipynb


References:
===========
Dynamic Histogram Analysis To Determine Free Energies and Rates from biased 
Simulations, L. S. Stelzl, A. Kells, E. Rosta, G. Hummer, J. Chem. Theory Comput.,
2017, http://pubs.acs.org/doi/abs/10.1021/acs.jctc.7b00373
