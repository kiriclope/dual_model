import sys, os, importlib
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import params as gv 
importlib.reload(sys.modules['params']) 
from utils import pad_along_axis

gv.init_param()
    
raw_volt = pd.read_csv(gv.path + '/mem_volt.dat', sep='\s+', header=None).to_numpy() 
print(raw_volt.shape)

time = raw_volt[:,0] 
print('time', time.shape)

volt = np.delete(raw_volt, [0], axis=1)
print('volt', volt.shape) 

if gv.n_pop!=1:                
    volt = pad_along_axis(volt, gv.n_size[0]-gv.n_size[1])         
    n_neurons = int(volt.shape[1]/2) 
    volt = np.reshape(volt, (volt.shape[0], 2, n_neurons)) 
else: 
    n_neurons = volt.shape[1] 
    volt = np.reshape(volt, (volt.shape[0], 1, n_neurons)) 
    
print('volt', volt.shape)

mean_volt = np.mean(volt, axis=-1) 

# print(mean_volt.shape) 

avg_mean_volt = np.mean(mean_volt, axis=0) 
print('avg_mean_volt', avg_mean_volt) 

figtitle = 'mem_volt'
fig = plt.figure(figtitle, figsize=(1.25*1.618*1.5*6, 1.618*1.25*6)) 
# plt.ylabel('Mem. volt. (mV)') 

for i in range(6) :
    ax = fig.add_subplot(6, 1, i+1) 
    i_neuron = np.random.randint(0, n_neurons)
    
    if i<3: 
        plt.plot(time, volt[:, 0, i_neuron], alpha=0.5, color='r') 
    else: 
        plt.plot(time, volt[:, 1, i_neuron], alpha=0.5, color='b') 
    
    plt.xlabel('Time (ms)') 
    plt.xlim([0, 1000])
    plt.ylim([-80, 30]) 
    
plt.show()
