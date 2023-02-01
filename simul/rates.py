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

from plot_settings import SetPlotParams
SetPlotParams(5, 2) 

# mean_field = inputs_dist() 
# mean_var = mean_field.x 

# if gv.n_pop==2 :
#     mf_inputs_E = np.random.normal(loc=mean_var[0], scale=np.sqrt(mean_var[2]), size=gv.n_size) 
#     mf_inputs_I = np.random.normal(loc=mean_var[1], scale=np.sqrt(mean_var[3]), size=gv.n_size) 
    
#     mf_rates_E = vec_Phi(mf_inputs_E)*1000
#     mf_rates_I = vec_Phi(mf_inputs_I)*1000 
# else :
#     mf_inputs = np.random.normal(loc=mean_var[0], scale=np.sqrt(mean_var[1]), size=gv.n_size) 
#     mf_rates = vec_Phi(mf_inputs) 
    

time, rates = get_time_rates(path=gv.path) 
mean_rates = np.nanmean(rates, axis=-1) 

print(time.shape)
# print(mean_rates.shape) 

avg_mean_rates = np.nanmean(mean_rates, axis=0) 
print('avg_mean_rates', avg_mean_rates) 

figtitle = 'rates_' + gv.folder 
# fig = plt.figure(figtitle, figsize=(1.25*1.618*1.5*2, 1.618*1.25)) 
fig = plt.figure(figtitle) 
ax = fig.add_subplot(int('121'))

# for _ in range(5) :
#     i_neuron = np.random.randint(0, n_neurons) 
#     plt.plot(time, rates[..., i_neuron], alpha=0.25) 

for i_pop in range(gv.n_pop):
    plt.plot(time, mean_rates[:,i_pop], color=gv.pal[i_pop]) 
   
plt.xlabel('Time (ms)') 
plt.ylabel('Rates (Hz)') 
# plt.ylim([0, 1]) 
plt.xlim([0, 2]) 

ax = fig.add_subplot(int('122')) 

avg_rates = np.nanmean(rates, axis=0) 
print(avg_rates.shape)
# plt.hist(avg_rates, histtype='step') 

for i_pop in range(gv.n_pop):
    plt.hist(avg_rates[i_pop], histtype='step') 

# if gv.n_pop==2:
#     plt.hist(mf_rates_I, histtype='step', ls='--', color='b') 
#     plt.hist(mf_rates_E, histtype='step', ls='--', color='r') 
# else:
#     plt.hist(mf_rates, histtype='step', ls='--', color='b') 

plt.xlabel('Rates (Hz)')
plt.ylabel('Count')

plt.show()
