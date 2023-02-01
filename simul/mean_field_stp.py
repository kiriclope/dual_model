import importlib, sys
from importlib import reload

import numpy as np

import scipy.integrate as integrate
import scipy.special as special

from scipy.optimize import fsolve, root

import random as rand
import params as gv 

THETA = 1.0
TAU_E = 20.0

def rho(a,b,s):
    mi = quench_avg_Phi(a, b) 
    # for binary: rho(t) = mi * (1.0 - mi) / ( TAU_E * ( 1.0 - 2.0 * mi ) ) * ( exp(-t * mi / TAU_E ) -  exp(- t * (1.0 -mi) / TAU_E ) ) => laplace transform below 
    return mi * (1.0 - mi) / ( TAU_E * ( 1.0 - 2.0 * mi ) ) * ( 1.0/( s + mi/TAU_E ) -  1.0 / ( s + (1.0-mi) / TAU_E ) ) 

def proba_release(a,b): 
    
    F = rho(a, b, 1.0 / gv.TAU_FAC) 
    D = rho(a, b, 1.0 / gv.TAU_REC) 
    H = rho(a, b, 1.0 / gv.TAU_REC + 1.0 / gv.TAU_FAC) 
    
    y_avg = gv.USE * F / ( 1.0 - ( 1.0 - gv.USE ) * F ) 
    xy_avg = (1.0 - H/F) * y_avg 
    x_avg = ( 1.0 - ( 1.0 + (1-gv.USE) * xy_avg ) * D ) / ( 1.0 - ( 1.0 - gv.USE) * D ) 
    
    return gv.USE * x_avg + (1.0 - gv.USE) * xy_avg 

def inputs_dist(): 
    print('J', np.array(gv.J)) 
    print('ext_inputs', gv.ext_inputs) 
    
    x0 = [rand.random() for i in range(0,2*gv.n_pop)] 
    
    print('x0', x0, 'self_cst_eqs', self_consistent_eqs(x0)) 

    # y = fsolve(self_consistent_eqs,x0) # to fix fsolve is not always converging ... 
    # mean_inputs = y[0:gv.n_pop] 
    # var_inputs = np.abs(y[gv.n_pop:2*gv.n_pop]) 
    
    y = root(self_consistent_eqs, x0, method='lm') 
    mean_inputs = y.x[0:gv.n_pop] 
    var_inputs = y.x[gv.n_pop:2*gv.n_pop] 
    
    # counter = 0 
    # while any(self_consistent_eqs(x0)>1e-6):
        
    #     # y = fsolve(self_consistent_eqs,x0) # to fix fsolve is not always converging ... 
    #     # mean_inputs = y[0:gv.n_pop] 
    #     # var_inputs = np.abs(y[gv.n_pop:2*gv.n_pop]) 
        
    #     y = root(self_consistent_eqs, x0, method='lm') 
    #     mean_inputs = y.x[0:gv.n_pop] 
    #     var_inputs = y.x[gv.n_pop:2*gv.n_pop] 
        
    #     x0 = [rand.random() for i in range(0,2*gv.n_pop)] 
        
    #     if counter>=100 : 
    #         print( any(self_consistent_eqs(x0)> 1e-6) ) 
    #         print( self_consistent_eqs(x0) ) 
    #         print('ERROR: max number of iterations reached') 
    #         break 
        
    #     counter+=1 
    
    print('mf rates ', vec_quench_avg_Phi(mean_inputs, var_inputs ), 'error', self_consistent_eqs(y.x) ) 
    return y.x 

def self_consistent_eqs(x): 
    mean_input = x[0:gv.n_pop] 
    var_input = np.abs(x[gv.n_pop:2*gv.n_pop]) 
    
    gv.J[0][0] = gv.JEE * proba_release(mean_input[0], var_input[0]) 
    gv.J2[0][0] = gv.JEE2 * proba_release(mean_input[0], var_input[0]) 
    
    mean_eq = mean_input / np.sqrt(gv.K) - ( gv.ext_inputs + np.dot(gv.J, vec_quench_avg_Phi(mean_input, var_input) ) ) # add [0] if using quad 
    var_eq = var_input - np.abs( np.dot( gv.J2, vec_quench_avg_Phi2(mean_input, var_input) ) ) 
    
    eqs = np.array([mean_eq, var_eq]) 
    return eqs.flatten() 

def phi(x): # normal distribution 
    return np.exp(-0.5*x*x) / np.sqrt(2*np.pi) 

def Phi(x): # transfert function 
    # return 0.5 *(1.0 + special.erf(x/np.sqrt(2)) )  # CDF of the normal distribution 
    # if(x>0):  # Threshold linear 
    #     return x 
    # else: 
    #     return 0 

    return np.heaviside(x-THETA, 0) 
    
def integrand(x, a, b): 
    return Phi(a + np.sqrt(b) * x) * phi(x) 

def integrand2(x, a, b): 
    return Phi(a + np.sqrt(b) * x) * Phi(a + np.sqrt(b) * x) * phi(x) 

def quench_avg_Phi(a, b):
    # if(gv.model=='lif'):
    #     # return integrate.quad(integrand, -np.inf, np.inf, args=(a,b)) 
    #     return Phi(a/np.sqrt(1.0 + b))
    # if(gv.model=='binary'):
    return 0.5*special.erfc( (THETA-a ) / np.sqrt(2.*b) ) 
    
def quench_avg_Phi2(a, b): 
    # if(gv.model=='lif'):
    #     return integrate.quad(integrand2, -np.inf, np.inf, args=(a,b)) 
    # if(gv.model=='binary'):
    return 0.5*special.erfc( (THETA-a) / np.sqrt(2.*b) ) 

vec_Phi = np.vectorize(Phi) 
vec_quench_avg_Phi = np.vectorize(quench_avg_Phi)
vec_quench_avg_Phi2 = np.vectorize(quench_avg_Phi2)

if __name__ == "__main__":
    gv.init_param() 
    inputs_dist()
