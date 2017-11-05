import numpy as np
from collections import Counter, defaultdict
import os
import pyprind

def count_matrix(traj, lag=1, n_states=None):
    """
    determine transition counts from a trajectory with a given lag time
    
    C[i,j] where i is the product state and j the reactent state.
   
    The first row contains thus all the transitions into state 0.
    The first colmun C[:,0] all transition out of state 0.
    
    """
    if n_states is None:
        n_states = np.max(traj) # 0 or zero based indexing?
    b = np.zeros((n_states, n_states))
    
    for (x,y), c in Counter(zip(traj[:-lag], traj[lag:])).iteritems():
        #b[x-1, y-1] = c
        b[int(y), int(x)] =c
    
    return b
        
        
def loop_traj_count_matrix(traj_dict, lag=1, n_states=None, trj1_index="0"):
    
    count_matrix_dict = {}
    
    for k, v in traj_dict.items():
        count_matrix_dict[k] = count_matrix(v, lag=lag, n_states=n_states)
    
    # assuming zero based trajectory indexing
    count_matrix_comb = count_matrix_dict[trj1_index]
    
    for k, v in count_matrix_dict.items():
        count_matrix_comb += v
        
    return count_matrix_comb
    
    
def index2d_1d(i,j,M=100):
    return i*M + j
