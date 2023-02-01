import sys, os, importlib
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import params as gv 
importlib.reload(sys.modules['params']) 

if gv.IF_SPEC == 1:
    if len(sys.argv)>0:
        gv.KAPPA = float(sys.argv[1]) 
    
if gv.IF_INI_COND==1:    
    if len(sys.argv)>0:
        if gv.IF_SPEC:
            gv.INI_COND_ID = int(sys.argv[2])
        else:
            gv.INI_COND_ID = int(sys.argv[1])

gv.init_param()

from balance_inputs_dist import inputs_dist, vec_Phi
from utils import *

time, rates = get_time_rates(path=gv.path) 

figtitle = 'rates_' + gv.folder 
fig = plt.figure(figtitle, figsize=(1.25*1.618*1.5, 1.618*1.25)) 

for i_pop in range(gv.n_pop):
    counter=0
    while(counter<5):
        i_neuron = np.random.randint(0, gv.n_size[i_pop]) 
        plt.plot(time, rates[:,i_pop, i_neuron], color=gv.pal[i_pop]) 
        counter+=1 
        
plt.xlabel('Time (ms)') 
plt.ylabel('Rates (Hz)') 

plt.show()
