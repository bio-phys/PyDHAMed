import numpy as np
import numba
import time
from scipy.optimize import *
from prepare_dhamed import *


@numba.jit(nopython=True)
def effective_log_likelihood_count_list(g,  ip, jp, ti, tj, vi, vj, nk, nijp,
                                       jit_gradient=False):
    #g[-1] = 0
    #g = np.append(g_i, 0)
    xlogp = 0 
    for ipair, i in enumerate(ip):
        j = jp[ipair]
        _vi = vi[ipair]
        _vj = vj[ipair]
        #print i, j
        w = 0.5 * (_vi - g[i] + _vj - g[j])
        taui = ti[ipair]*np.exp(_vi-g[i] -w)
        tauj = tj[ipair]*np.exp(_vj-g[j] -w)
        xlogp += nijp[ipair]* (np.log(taui+tauj)+w)

    return xlogp + np.sum(nk*g)


def effective_log_likelihood_count_ref(g,  ip, jp, ti, tj, vi, vj, nk, njip):
    #g[-1] = 0
    xlogp = 0 
    for ipair, i in enumerate(ip):
        j = jp[ipair]
        _vi = vi[ipair]
        _vj = vj[ipair]
        #print i, j
        w = 0.5 * (_vi - g[i] + _vj - g[j])
        taui = ti[ipair]*np.exp(_vi-g[i] -w)
        tauj = tj[ipair]*np.exp(_vj-g[j] -w)
        xlogp += nijp[ipair]* (np.log(taui+tauj)+w)

    return xlogp + np.sum(nk*g)
    
    
@numba.jit(nopython=True)
#@numba.jit(numba.float64[:](numba.float64[:],numba.types.int8[:],numba.types.int8[:],
#          numba.float64[:],numba.float64[:],numba.float64[:],numba.float64[:],
#          numba.types.int8[:],numba.types.int8[:]),nopython=True)   
def grad_dhamed_likelihood(g,  ip, jp, ti, tj, vi, vj, nk, nijp):
    #g[-1] = 0
    grad = np.zeros(g.shape)
    grad  += nk
    for ipair, i in enumerate(ip):
        j = jp[ipair]
        vij = np.exp(vj[ipair]-g[j]-vi[ipair]+g[i])
        # don't think I need to test if ti exists
        if ti[ipair] > 0:
            grad[i] += -nijp[ipair] / (1.0 + tj[ipair]*vij/ti[ipair])
        if tj[ipair] >0 :
            grad[j] += -nijp[ipair] / (1.0 + ti[ipair]/(vij*tj[ipair]))
    return grad


def grad_dhamed_likelihood_ref(g,  ip, jp, ti, tj, vi, vj, nk, nijp):
    #g[-1] = 0
    grad = np.zeros(g.shape)
    grad  += nk
    for ipair, i in enumerate(ip):
        j = jp[ipair]
        vij = np.exp(vj[ipair]-g[j]-vi[ipair]+g[i])
        # don't think I need to test if ti exists
        if ti[ipair] > 0:
            grad[i] += -nijp[ipair] / (1.0 + tj[ipair]*vij/ti[ipair])
        if tj[ipair] >0 :
            grad[j] += -nijp[ipair] / (1.0 + ti[ipair]/(vij*tj[ipair]))
    return grad
    
    

def wrapper_ll(g_prime, g, ip, jp, ti, tj, vi, vj, nk, nijp,
               jit_gradient=False):
    """
    Adding the extra zero when minimizing N-1 relative weights.
    """
    g_i = np.append(g_prime, [0], axis=0)
    l = effective_log_likelihood_count_list(g_i,  ip, jp, ti, tj, vi, vj, nk, nijp)
    return l
    

def grad_dhamed_likelihood_ref_0(g_prime, g,  ip, jp, ti, tj, vi, vj, nk, nijp,
                                jit_gradient=False):
    g = np.append(g_prime, [0], axis=0)
    grad = np.zeros(g.shape[0] )
    grad[:-1]  += nk[:-1]
    if jit_gradient:
        grad = _loop_grad_dhamed_likelihood_0_jit(grad,g, ip, jp, ti, tj, vi, vj, nijp)
    else:
        grad = _loop_grad_dhamed_likelihood_0(grad,g, ip, jp, ti, tj, vi, vj, nijp)
    return grad[:-1] 
 
 
#@numba.jit(nopython=True)
def _loop_grad_dhamed_likelihood_0(grad, g,  ip, jp, ti, tj, vi, vj, nijp):
    for ipair, i in enumerate(ip):
        j = jp[ipair]
        vij = np.exp(vj[ipair]-g[j]-vi[ipair]+g[i])
        # don't think I need to test if ti exists
        if ti[ipair] > 0:
            grad[i] += -nijp[ipair] / (1.0 + tj[ipair]*vij/ti[ipair])
        if tj[ipair] >0 :
            grad[j] += -nijp[ipair] / (1.0 + ti[ipair]/(vij*tj[ipair]))
    return grad


_loop_grad_dhamed_likelihood_0_jit = numba.jit(_loop_grad_dhamed_likelihood_0, nopython=True)
    
    
def run_dhamed(count_list, bias_ar, numerical_gradients=False, g_init=None,
               jit_gradient=False, last_g_zero=True, **kwargs):
    """
    Run DHAMed from a list of count matrices and an array specfying the
    biases in each simulation (window).
    
    The list of the individual count matrices C contain the transition counts
    between the different states (or bins in umbrella sampling). C[i,j] where
    i is the product state and j the reactent state. The first row contains
    thus all the transitions into state 0.The first column C[:,0] all 
    transition out of state 0.
    
    The bias array contains a bias value for each state and for each simulation
    (or window in umbrella sampling. The bias NEEDS to be given in units to kBT.
    
    Most parameters besides count_list and bias_ar are only relevant for testing
    and further code developement. 
    
    The function takes keywords arguments for fmin_bfgs() such as the gtol and 
    maxiter.
    
    Parameters:
    -----------
    count_list: list of arrays, NxN the transition counts for
                each simulation (window)
    bias_ar: array, (Nxnwin) the bias acting on each state in each
             simulation (window)
    numerical_gradients: Boolean. default False, use analytical gradients.
    g_init: initial log-weights,          
    
    Returns:
    --------
    og: array-like, optimized log-weights
    """           
    
    n_states = count_list[0].shape[0]
    n_windows = bias_ar.shape[1]
    
    if np.all(g_init) is None:
       g_init = np.zeros(n_states)
       
    #u_min = np.min(bias_ar, axis=0)
    #bias_ar -= u_min
      
    n_out, ip, jp, vi, vj, ti, tj, nijp = generate_dhamed_input(count_list, bias_ar,
                                                                n_states, n_windows)
    start = time.time()
    
    if numerical_gradients:
       fprime=None
       #print fprime
    else:
         if jit_gradient:
            fprime = grad_dhamed_likelihood
         else:
              fprime=grad_dhamed_likelihood_ref
    
    #print g_init, ip -1, jp -1, ti, tj, vi, vj, n_out, nijp
    
    # ip - 1, jp -1 : to get zero based indices
    l0 = effective_log_likelihood_count_list(g_init*1.0, ip -1, jp -1, ti, tj, vi, vj,
                                              n_out, nijp)
    print ("loglike-start {}".format(l0))
    
    if last_g_zero:
       og = min_dhamed_bfgs(g_init, ip, jp, ti, tj, vi, vj, n_out, nijp, jit_gradient=jit_gradient,
                            **kwargs)
    
    else:
         og = fmin_bfgs(effective_log_likelihood_count_list, g_init*1.0,
                        args=( ip -1, jp -1, ti, tj, vi, vj, n_out, nijp), 
              fprime=fprime, **kwargs)
    end = time.time()
    print "time elapsed {} s".format(end-start)    
    
    #correct optimal log weights by adding back umin 
    #output free energies are relative to the last bias!
    return og #+ u_min #- u_min[-1] 
    

def min_dhamed_bfgs(g_init, ip, jp, ti, tj, vi, vj, n_out, nijp, jit_gradient=False,
                    **kwargs):
    """
    Find the optimal weights to solve the DHAMed equations by 
    determining the N-1 optimal relative weights of the states.
    
    Parameters:
    -----------
    g_init: array, N entries, initial log weights
    ip: array of integers, 
    
    """                
    g = g_init.copy()
    g_prime = g[:-1].T   
    # ip - 1, jp -1 : to get zero based indices
    print wrapper_ll(g_prime,g, ip-1, jp-1, ti, tj, vi, vj, n_out, nijp, jit_gradient)
    og = fmin_bfgs(wrapper_ll, g_prime,
                   args=(g, ip -1, jp -1, ti, tj, vi, vj, n_out, nijp, jit_gradient), 
                   fprime=grad_dhamed_likelihood_ref_0, **kwargs)
    return np.append(og, 0)
