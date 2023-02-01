import sys, os, importlib
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import params as gv
importlib.reload(sys.modules['params']) 

gv.init_param()

u_stp = pd.read_csv(gv.path + '/u_stp.dat', sep='\s+').to_numpy() 
x_stp = pd.read_csv(gv.path + '/x_stp.dat', sep='\s+').to_numpy()

# u_stp = np.genfromtxt(gv.path + 'u_stp.dat') ;
# x_stp = np.genfromtxt(gv.path + 'x_stp.dat') ; 

print('u', u_stp.shape, 'x', x_stp.shape) 

# u_stp = np.loadtxt(gv.path + 'u_stp.dat') ;
# x_stp = np.loadtxt(gv.path + 'x_stp.dat') ; 

u_time = u_stp[:,0] 
x_time = x_stp[:,0] 

u_stp = np.delete(u_stp,[0],axis=1)
x_stp = np.delete(x_stp,[0],axis=1)

print('time', u_time.shape,'u', u_stp.shape, 'x', x_stp.shape)

figtitle = 'stp_vars'
fig = plt.figure(figtitle, figsize=(1.25*1.618*1.5, 1.618*1.25)) 

i_neuron=0
    
plt.plot(u_time/1000, u_stp[:,i_neuron], 'r', label='u') 
plt.plot(x_time/1000, x_stp[:,i_neuron], 'b', label='x')    
plt.plot(x_time/1000, u_stp[:,i_neuron] * x_stp[:,i_neuron], 'k', label='ux') 

    
plt.xlabel('time (s)') 
plt.ylabel('STP variables') 
plt.ylim([0,1])
plt.legend() 
    
plt.show()
