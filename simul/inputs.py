import sys, os, importlib
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import params as gv 
importlib.reload(sys.modules['params']) 

gv.init_param()

from balance_inputs_dist import inputs_dist, vec_Phi
from utils import *
from get_m1 import *
# mean_field = inputs_dist() 
# mean_var = mean_field.x

# if gv.n_pop==2 :
#     mf_inputs_E = np.random.normal(loc=mean_var[0], scale=np.sqrt(mean_var[2]), size=gv.n_size) 
#     mf_inputs_I = np.random.normal(loc=mean_var[1], scale=np.sqrt(mean_var[3]), size=gv.n_size) 
# else : 
#     mf_inputs = np.random.normal(loc=mean_var[0], scale=np.sqrt(mean_var[1]), size=gv.n_size) 
time, inputs = get_time_inputs(path=gv.path) 
print('time', time.shape, 'inputs', inputs.shape) 
try:
    _, ff_inputs = get_time_ff_inputs(path=gv.path) 
except:
    ff_inputs = np.sqrt(gv.K) * gv.ext_inputs 
    
figtitle = 'inputs' 
fig = plt.figure(figtitle, figsize=(5.663 * 2, 3.5*2) ) 

if gv.n_pop==1:
    
    ext_inputs = np.sqrt(gv.K) * gv.ext_inputs * np.ones((len(time), inputs.shape[-1]) ) 
    net_inputs = inputs + ext_inputs
        
    ax = fig.add_subplot(int('121'))
    
    plt.plot(time, mean_inputs, 'b', lw=2)
    plt.plot(time, np.mean(ext_inputs, axis=-1), 'r', lw=2) 
    plt.plot(time, np.mean(net_inputs, axis=-1), 'k', lw=2) 
    
    plt.xlabel('time (ms)') 
    plt.ylabel('inputs (mA)') 
    
    ax = fig.add_subplot(int('122')) 
        
    plt.hist( np.mean( ext_inputs, axis=0), color='r', histtype='step') 
    plt.hist( np.mean( inputs, axis=0), color='b', histtype='step') 
    plt.hist( np.mean( net_inputs, axis=0), color='k', histtype='step') 
    
    plt.hist( mf_inputs, color='k', ls='--', histtype='step') 
    
else:
    
    bins = [int(5/.05), -1] 
    
    for i_pop in range(gv.n_pop):
        ax = fig.add_subplot( int('22%d'% (i_pop+1) ) ) 
        
        # averaged over neurons 
        E_inputs = np.nanmean(ff_inputs[bins[0]:bins[1], 0, :gv.n_size[0]], axis=1) + np.nanmean(inputs[bins[0]:bins[1], 0, :gv.n_size[0]], axis=1) 
        I_inputs = np.nanmean(inputs[bins[0]:bins[1], 1, :gv.n_size[0]], axis=1)

        # neuron = np.random.randint(gv.n_size[i_pop]) 
        
        # E_inputs = ff_inputs[:, 0, neuron] + inputs[:, 0, neuron] 
        # I_inputs = inputs[:, 1, neuron] 
        
        net_inputs =  E_inputs + I_inputs 
        
        plt.plot(time[bins[0]:bins[1]], E_inputs, 'r', lw=1) 
        plt.plot(time[bins[0]:bins[1]], net_inputs, 'k', lw=1) 
        plt.plot(time[bins[0]:bins[1]], I_inputs, 'b', lw=1) 
                
        plt.xlabel('Time (ms)') 
        plt.ylabel('Inputs (mA)')
        
        ax = fig.add_subplot(int('22%d' %(i_pop+3))) 

        # averaged over time
        E_inputs = np.nanmean(ff_inputs[bins[0]:bins[1], 0, :gv.n_size[0]], axis=0) + np.nanmean( inputs[bins[0]:bins[1], 0, :gv.n_size[0]], axis=0) 
        I_inputs = np.nanmean(inputs[bins[0]:bins[1], 1, :gv.n_size[0]], axis=0)
            
        net_inputs =  E_inputs + I_inputs 
        
        plt.hist(E_inputs , color='r', histtype='step') 
        plt.hist(I_inputs , color='b', histtype='step') 
        plt.hist(net_inputs , color='k', histtype='step') 
        
        # if i_pop==0: 
        #     plt.hist( mf_inputs_E, color='k', ls='--', histtype='step') 
        # else: 
        #     plt.hist( mf_inputs_I, color='k', ls='--', histtype='step') 
        
        plt.xlabel('Inputs (mA)') 
        plt.ylabel('Count') 

plt.show()
