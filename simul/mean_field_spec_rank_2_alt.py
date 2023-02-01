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
    
    J = np.array(J) 
    if(verbose): 
        print('J', np.array(J)) 
        print('ext_inputs', ext_inputs) 
        print('K', K, 'kappa', kappa) 
    
    if x0 is None: 
        x0 = np.array([rand.random() for i in range(0,4*n_pop)] ) 
    
    u00 = x0[0:n_pop] 
    u10 = x0[n_pop:2*n_pop] 
    u01 = x0[2*n_pop:3*n_pop] 
    alpha = x0[3*n_pop:4*n_pop] 
    
    # y = fsolve(self_consistent_eqs,x0) # to fix fsolve is not always converging ... 
    y = root(lambda x : self_consistent_eqs(x, kappa=kappa, J=J, J2=J2, K=K, ext_inputs=ext_inputs, n_pop=n_pop), x0, method='lm') 
    
    # counter = 0 
    # while any(self_consistent_eqs(x0, kappa=kappa, K=K) > TOLERANCE ): 
        
    #     x0 = np.array([rand.random() for i in range(0,4*n_pop)] ) 
    #     # y = fsolve(self_consistent_eqs,x0) # to fix fsolve is not always converging ... 
    #     y = root(lambda x : self_consistent_eqs(x, kappa=kappa, K=K), x0, method='lm', tol=TOLERANCE) 
        
    #     if counter>=MAXITER : 
    #         print( any(self_consistent_eqs(x0, kappa=kappa, K=K)> 1e-6) ) 
    #         print( self_consistent_eqs(x0, kappa=kappa, K=K) ) 
    #         print('ERROR: max number of iterations reached') 
    #         break 
        
    #     counter+=1 
    
    u00 = y.x[0:n_pop] 
    
    u10 = y.x[n_pop:2*n_pop] 
    u01 = y.x[2*n_pop:3*n_pop] 
    
    alpha = y.x[3*n_pop:4*n_pop] 
    
    m00 = vec_m00_func(u00, u10, u01, alpha)[0] 
    m10 = vec_m10_func(u00, u10, u01, alpha)[0] 
    m01 = vec_m01_func(u00, u10, u01, alpha)[0] 
    
    if verbose: 
        print('mf m00', m00, 'm10', m10, 'm01', m01) 
    
    return m00, m10, m01, y.x 

def self_consistent_eqs(x, kappa=gv.KAPPA, J=gv.J, J2=gv.J2, K=gv.K, ext_inputs=gv.ext_inputs, n_pop=gv.n_pop): 
    u00 = x[0:n_pop] # mean input 
    u10 = np.abs( x[n_pop:2*n_pop] ) # first fourier moment of the input 
    u01 = np.abs( x[2*n_pop:3*n_pop] ) # first fourier moment of the input 
    alpha = np.abs( x[3*n_pop:4*n_pop] ) # variance 
    
    u00_eq = u00 / np.sqrt(K) - ( ext_inputs + np.dot(J, vec_m00_func(u00, u10, u01, alpha)[0] ) ) # add [0] if using quad 
    
    u10_eq = u10 - np.abs( np.array( [kappa/2.0 * J[0,0] * vec_m10_func(u00, u10, u01, alpha)[0][0], 0 ] ) ) 
    u01_eq = u01 - np.abs( np.array( [0.5*kappa/2.0 * J[0,0] * vec_m01_func(u00, u10, u10,  alpha)[0][0], 0 ] ) ) 
    
    alpha_eq = alpha - np.abs( np.dot(J2, vec_m00_func(u00, u10, u01, alpha)[0] ) ) 
    
    eqs = np.array([u00_eq, u10_eq, u01_eq,  alpha_eq]) 
    return eqs.flatten() 

def u_theta(theta, phi, u00, u10, u01): 
    return u00 + u10 * np.cos(2*theta) + u01 * np.cos(2*phi)  

def integrand00(theta, phi, u00, u10, u01, alpha): 
    return quench_avg_Phi(u_theta(theta, phi, u00, u10, u01), alpha ) / np.pi / np.pi

def m00_func(u00, u10, u01, alpha): 
    return integrate.dblquad(integrand00, 0, np.pi, lambda phi: 0, lambda phi: np.pi, args=(u00, u10, u01, alpha) ) 

def integrand10(theta, phi, u00, u10, u01, alpha): 
    return 2.0 * quench_avg_Phi(u_theta(theta, phi, u00, u10, u01), alpha ) * np.cos(2.0*theta) / np.pi / np.pi 

def m10_func(u00, u10, u01, alpha): 
    return integrate.dblquad(integrand10, 0, np.pi, lambda phi: 0, lambda phi: np.pi, args=(u00, u10, u01, alpha) ) 

def integrand01(theta, phi, u00, u10, u01, alpha): 
    return 2.0 * quench_avg_Phi(u_theta(theta, phi, u00, u10, u01), alpha ) * np.cos(2.0*phi) / np.pi / np.pi 

def m01_func(u00, u10, u01, alpha): 
    return integrate.dblquad(integrand01, 0, np.pi, lambda phi: 0, lambda phi: np.pi, args=(u00, u10, u01, alpha) ) 

def quench_avg_Phi(a, b): 
    return 0.5*special.erfc( (THETA-a ) / np.sqrt(2.* np.abs(b) ) ) 
    
vec_m00_func = np.vectorize(m00_func) 
vec_m10_func = np.vectorize(m10_func) 
vec_m01_func = np.vectorize(m01_func) 

if __name__ == "__main__":
    gv.init_param() 
    get_m0_m1_mf(kappa=gv.KAPPA, J=gv.J, K=gv.K, ext_inputs=gv.ext_inputs, n_pop=gv.n_pop, verbose=1) 
