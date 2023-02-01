import importlib, sys 
from importlib import reload 

import numpy as np 

import scipy.integrate as integrate
import scipy.special as special

from scipy.optimize import fsolve, root

import random as rand 
import params as gv 

THETA = 1 
TAU_E = 20.0 
TOLERANCE=1e-6
MAXITER=100

def get_m0_m1_mf(kappa=gv.KAPPA, J=gv.J, J2=gv.J2, K=gv.K, ext_inputs=gv.ext_inputs, n_pop=gv.n_pop, JEE=gv.JEE, JEE2=gv.JEE2, x0=None, verbose=0): 
    # JEE=6 
    # JEE2=6*6 
    
    J = np.array(J) 
    if(verbose): 
        print('J', np.array(J)) 
        print('ext_inputs', ext_inputs) 
        print('K', K, 'kappa', kappa) 
    
    if x0 is None:
        x0 = np.array([rand.random() for i in range(0,3*n_pop)] ) 
    
    u0 = x0[0:n_pop] 
    u1 = x0[n_pop:2*n_pop] 
    alpha = x0[2*n_pop:3*n_pop]  
    
    # y = fsolve(self_consistent_eqs,x0) # to fix fsolve is not always converging ... 
    y = root(lambda x : self_consistent_eqs(x, kappa=kappa, K=K, J=J, J2=J2, ext_inputs=ext_inputs, n_pop=n_pop, JEE=JEE, JEE2=JEE2), x0, method='lm') 
    
    # counter = 0 
    # while any(self_consistent_eqs(x0, kappa=kappa, K=K) > TOLERANCE ): 
        
    #     x0 = np.array([rand.random() for i in range(0,3*n_pop)] ) 
    #     # y = fsolve(self_consistent_eqs,x0) # to fix fsolve is not always converging ... 
    #     y = root(lambda x : self_consistent_eqs(x, kappa=kappa, K=K, J=J, J2=J2, ext_inputs=ext_inputs, n_pop=n_pop, JEE=JEE, JEE2=JEE2), x0, method='lm', tol=TOLERANCE) 
    
    #     if counter>=MAXITER : 
    #         print( any(self_consistent_eqs(x0, kappa=kappa, K=K)> 1e-6) ) 
    #         print( self_consistent_eqs(x0, kappa=kappa, K=K) ) 
    #         print('ERROR: max number of iterations reached') 
    #         break 
    
    #     counter+=1 
    
    u0 = y.x[0:n_pop] 
    u1 = y.x[n_pop:2*n_pop] 
    alpha = y.x[2*n_pop:3*n_pop] 
    
    m0 = vec_m0_func(u0, u1, alpha)[0] 
    m1 = vec_m1_func(u0, u1, alpha)[0] 
    
    if verbose: 
        print('mf m0', m0, 'm1', m1 , 'error', self_consistent_eqs(y.x), 'stp',  JEE * proba_release(u0[0], u1[0], alpha[0]) ) 
    
    return m0, m1, y.x

def self_consistent_eqs(x, kappa=gv.KAPPA, J=gv.J, J2=gv.J2, K=gv.K, ext_inputs=gv.ext_inputs, n_pop=gv.n_pop, JEE=gv.JEE, JEE2=gv.JEE2): 
    u0 = x[0:n_pop] # mean input 
    u1 = np.abs( x[n_pop:2*n_pop] ) # first fourier moment of the input 
    alpha = np.abs( x[2*n_pop:3*n_pop] ) # variance 
    
    J[0][0] = JEE * proba_release(u0[0], u1[0], alpha[0]) 
    J2[0][0] = JEE2 * proba_release(u0[0], u1[0], alpha[0]) 
    
    u0_eq = u0 / np.sqrt(K) - ( ext_inputs + np.dot(J, vec_m0_func(u0, u1, alpha)[0] ) ) # add [0] if using quad 
    u1_eq = u1 - np.abs( np.array( [kappa/2.0 * J[0,0] * vec_m1_func(u0, u1, alpha)[0][0], 0 ] ) ) 
    alpha_eq = alpha - np.abs( np.dot(J2, vec_m0_func(u0, u1, alpha)[0] ) ) 
    
    eqs = np.array([u0_eq, u1_eq, alpha_eq]) 
    return eqs.flatten() 

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

def rho(u0, u1, alpha, s):
    mi = m0_func(u0, u1, alpha)[0] 
    # for binary: rho(t) = mi * (1.0 - mi) / ( TAU_E * ( 1.0 - 2.0 * mi ) ) * ( exp(-t * mi / TAU_E ) -  exp(- t * (1.0 -mi) / TAU_E ) ) => laplace transform below 
    return mi * (1.0 - mi) / ( TAU_E * ( 1.0 - 2.0 * mi ) ) * ( 1.0 / ( s + mi/TAU_E ) -  1.0 / ( s + (1.0-mi) / TAU_E ) ) 

def proba_release(u0, u1, alpha): 

    # Mongillo
    # F = rho(u0, u1, alpha, 1.0 / gv.TAU_FAC ) + 1e-18 
    # D = rho(u0, u1, alpha, 1.0 / gv.TAU_REC ) 
    # H = rho(u0, u1, alpha, 1.0 / gv.TAU_REC + 1.0 / gv.TAU_FAC ) 
    
    # y_avg = gv.USE * F / ( 1.0 - ( 1.0 - gv.USE ) * F ) 
    # xy_avg = (1.0 - H/F) * y_avg 
    # x_avg = ( 1.0 - ( 1.0 + (1-gv.USE) * xy_avg ) * D ) / ( 1.0 - ( 1.0 - gv.USE) * D ) 
    
    # return gv.USE * x_avg + (1.0 - gv.USE) * xy_avg 

    # Markram
    m = m0_func(u0, u1, alpha)[0]
    u_st = gv.USE / ( 1.0 - ( 1.0 - gv.USE ) * np.exp(-1.0/m/gv.TAU_FAC) ) 
    x_st = ( 1.0 - np.exp(-1.0/m/gv.TAU_REC) ) / ( 1.0 - (1.0 - u_st * np.exp(-1.0/m/gv.TAU_REC) ) )
    
    return u_st*x_st 

if __name__ == "__main__":
    gv.init_param() 
    get_m0_m1_mf(kappa=gv.KAPPA, J=gv.J, K=gv.K, ext_inputs=gv.ext_inputs, n_pop=gv.n_pop, JEE=gv.JEE, JEE2=gv.JEE2, verbose=1) 
