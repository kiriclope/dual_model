import importlib, sys
from importlib import reload

import numpy as np

import scipy.integrate as integrate
import scipy.special as special

from scipy.optimize import fsolve, root

import random as rand
import params as gv 

THETA = 1  

def inputs_dist():
    gv.init_param()
    print('J', np.array(gv.J)) 
    print('ext_inputs', gv.ext_inputs) 
    
    x0 = [rand.random()/2.0 for i in range(0,2*gv.n_pop)] 
    # y = fsolve(self_consistent_eqs,x0) # to fix fsolve is not always converging ... 
    y = root(self_consistent_eqs, x0, method='lm') 
    
    mean_inputs = y.x[0:gv.n_pop] 
    var_inputs = y.x[gv.n_pop:2*gv.n_pop] 
    
    print('mf rates ', vec_quench_avg_Phi(mean_inputs, var_inputs ) ) 
    return y 

def self_consistent_eqs(x): 
    mean_input = x[0:gv.n_pop] 
    var_input = x[gv.n_pop:2*gv.n_pop] 
    
    if gv.n_pop == 2 : 
        mean_eq = mean_input / np.sqrt(gv.K) - ( gv.ext_inputs + gv.J.dot(vec_quench_avg_Phi(mean_input, var_input)[0] ) ) # add [0] if using quad 
        var_eq = var_input - (gv.J*gv.J).dot(vec_quench_avg_Phi2(mean_input, var_input)[0]) 
    else: 
        mean_eq = mean_input / np.sqrt(gv.K) - ( gv.ext_inputs - gv.J * vec_quench_avg_Phi(mean_input, var_input)[0] ) # add [0] if using quad 
        var_eq = var_input - (gv.J*gv.J) * vec_quench_avg_Phi2(mean_input, var_input)[0] 
        
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
    # return integrate.quad(integrand, -np.inf, np.inf, args=(a,b)) 
    # return Phi(a/np.sqrt(1.0 + b)) 
    return 0.5*special.erfc( (THETA-a ) / np.sqrt(2.*b) ) 
    
def quench_avg_Phi2(a, b): 
    # return integrate.quad(integrand2, -np.inf, np.inf, args=(a,b)) 
    return 0.5*special.erfc( (THETA-a ) / np.sqrt(2.*b) ) 

vec_Phi = np.vectorize(Phi) 
vec_quench_avg_Phi = np.vectorize(quench_avg_Phi)
vec_quench_avg_Phi2 = np.vectorize(quench_avg_Phi2)

