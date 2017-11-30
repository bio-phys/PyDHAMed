from __future__ import print_function
from collections import defaultdict, Counter
import numpy as np


def pop_from_tba_eq_traj(tba, verbose=False, n_states=32):
    p_ar = np.zeros(n_states)
    traj_time = len(tba) * 1.0
    for s,c in Counter(tba).iteritems():
        if verbose:
            print(s, c)
        p_i = c/ float(traj_time)
        #p_l.append(p_i)
        p_ar[s-1] = p_i
    return p_ar


def block_average_pop_eq_tba(tba, n_blocks, n_states=32):
    tba_bl = np.split(tba, n_blocks)
    pop_bl_ar = np.zeros((n_states, n_blocks))

    for bi, b in enumerate(tba_bl):
        pop_bl_ar[:,bi] = pop_from_tba_eq_traj(b)
    return pop_bl_ar
