import importlib, sys 
from importlib import reload

import numpy as np

import scipy.integrate as integrate
import scipy.special as special

from scipy.optimize import fsolve, root

import random as rand 
import params as gv 

THETA = 1 
TOLERANCE=1e-8 
MAXITER=100 

def get_m0_m1_mf(kappa=gv.KAPPA, J=gv.J, J2=gv.J2, K=gv.K, ext_inputs=gv.ext_inputs, n_pop=gv.n_pop, x0=None, verbose=0): 
    
    J = np.array(J).reshape(n_pop, n_pop) 
    if(verbose): 
        print('J', np.array(J), 'Jee', J[0][0]) 
        print('ext_inputs', ext_inputs) 
        print('K', K, 'kappa', kappa) 
        
    if x0 is None:
        x0 = np.array([rand.random() for i in range(0,3*n_pop)] ) 
    
    u0 = x0[0:n_pop] 
    u1 = x0[n_pop:2*n_pop] 
    alpha = x0[2*n_pop:3*n_pop]  
    
    # y = fsolve(self_consistent_eqs,x0) # to fix fsolve is not always converging ... 
    y = root(lambda x : self_consistent_eqs(x, kappa=kappa, K=K, J=J, J2=J2), x0, method='lm') 
    
    counter = 0 
    output = self_consistent_eqs(x0, kappa=kappa, K=K, J=J, J2=J2) 
    while any( output > TOLERANCE ): 
        
        x0 = np.array([rand.random() for i in range(0,3*n_pop)] ) 
        # y = fsolve(self_consistent_eqs,x0) # to fix fsolve is not always converging ... 
        y = root(lambda x : self_consistent_eqs(x, kappa=kappa, K=K, J=J, J2=J2), x0, method='lm', tol=TOLERANCE) 
        output = self_consistent_eqs(x0, kappa=kappa, K=K, J=J, J2=J2) 
        
        if counter>=MAXITER :
            print( any(output > 1e-6) ) 
            print( output ) 
            print('ERROR: max number of iterations reached') 
            break 
        
        counter+=1 
    
    u0 = y.x[0:n_pop] 
    u1 = y.x[n_pop:2*n_pop] 
    alpha = y.x[2*n_pop:3*n_pop] 
    
    m0 = vec_m0_func(u0, u1, alpha)[0] 
    m1 = vec_m1_func(u0, u1, alpha)[0] 
    
    if verbose: 
        print('mf m0', m0, 'm1', m1 ) 
    
    return m0, m1, y.x

def self_consistent_eqs(x, kappa=gv.KAPPA, J=gv.J, J2=gv.J2, K=gv.K, ext_inputs=gv.ext_inputs, n_pop=gv.n_pop):

    mean = x[0:n_pop]
    var = np.abs( x[n_pop:2*n_pop] )
    overlap = x[2*n_pop:3*n_pop]
        
    mean_eq = mean / np.sqrt(K) - ( ext_inputs + np.dot(J, vec_m0_func(u0, u1, alpha)[0] ) ) # add [0] if using quad 
    var_eq = var - np.abs( np.array( [kappa/2.0 * J[0][0] * vec_m1_func(u0, u1, alpha)[0][0], 0 ] ) ) 
    overlap_eq = alpha - np.abs( np.dot(J2, vec_m0_func(u0, u1, alpha)[0] ) ) 
    
    eqs = np.array([u0_eq, u1_eq, alpha_eq]) 
    return eqs.flatten() 

def overlap_func():
    return integrate.quad(integrand, 0, np.pi, args=(u0, u1, alpha) ) 
    
def u_theta(theta, u0, u1): 
    return u0 + u1 * np.cos(2*theta) 

def integrand(theta, u0, u1, alpha): 
    return quench_avg_Phi(u_theta(theta, u0, u1), alpha ) / np.pi 

def integrand2(theta, u0, u1, alpha): 
    return 2.0 * quench_avg_Phi(u_theta(theta, u0, u1), alpha ) * np.cos(2.0*theta) / np.pi 

def m0_func(u0, u1, alpha): 
    return integrate.quad(integrand, 0, np.pi, args=(u0, u1, alpha) ) 

def m1_func(u0, u1, alpha): 
    return integrate.quad(integrand2, 0, np.pi, args=(u0, u1, alpha) ) 
    
def quench_avg_Phi(a, b): 
    return 0.5*special.erfc( (THETA-a ) / np.sqrt(2.* np.abs(b) ) ) 
    
vec_m0_func = np.vectorize(m0_func)
vec_m1_func = np.vectorize(m1_func)

if __name__ == "__main__":
    gv.init_param() 
    get_m0_m1_mf(kappa=gv.KAPPA, J=gv.J, K=gv.K, ext_inputs=gv.ext_inputs, n_pop=gv.n_pop, verbose=1)
