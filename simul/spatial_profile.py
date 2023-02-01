import sys, os, importlib
from importlib import reload
from scipy.signal import savgol_filter 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import params as gv 
importlib.reload(sys.modules['params']) 

from get_m1 import *
from utils import *

gv.init_param()
    
filter_rates = pd.read_csv(gv.path + '/filter_rates.dat', sep='\s+').to_numpy()
time = filter_rates[:,0] / 1000
rates = np.delete(filter_rates, [0], axis=1) 

if gv.n_pop!=1:
    rates = pad_along_axis(rates, gv.n_size[0]-gv.n_size[1]) 
    n_neurons = int(rates.shape[1]/2) 
    rates = np.reshape(rates, (rates.shape[0], 2, n_neurons))  
else: 
    n_neurons = rates.shape[0] 
    
mean_rates = np.nanmean(rates, axis=-1) 
m1, phi, smooth_rates = get_m1_phi_smooth_rates(rates)

figtitle = 'spatial_profile_' + gv.folder 

if gv.folder.find('off')!=-1 or gv.folder.find('on')!=-1:
    figtitle = 'spatial_profile'

fig = plt.figure(figtitle, figsize=(1.618*2*1.5*2, 1.618*2*2)) 

ax = fig.add_subplot(2,2,1) 
for i_pop in range(gv.n_pop): 
    plt.plot(time, mean_rates[:,i_pop], lw=1, color=gv.pal[i_pop]) 
    
plt.xlabel('Time (s)') 
plt.ylabel('Rates (Hz)') 
add_vlines()

ax = fig.add_subplot(2,2,2)

avg_m1=[] 
for i_pop in range(gv.n_pop-1) :     
    plt.plot(time, m1[i_pop], '-', color=gv.pal[i_pop]) 
    
plt.xlabel('Time (s)') 
plt.ylabel('Bump Amplitude (Hz)') 
add_vlines()

ax = fig.add_subplot(2,2,4) 

avg_phi = [] 
for i_pop in range(gv.n_pop-1) : 
    plt.plot(time, 2 * phi[i_pop] * 180 / np.pi - 180 , color=gv.pal[i_pop]) 

if(gv.IF_TRIALS and gv.IF_INI_COND):
    plt.hlines( 2*( (gv.TRIAL_ID * 180 / gv.N_TRIALS) % 180) - 180, 0, 10, ls='--') 
    plt.hlines( 2*( (1-gv.TRIAL_ID * 180 / gv.N_TRIALS) % 180) - 180, 0, 10, ls='--') 
else:
    plt.hlines( 2*( (gv.PHI_EXT * 180) % 180) - 180, 0, 10, ls='--') 
    plt.hlines( 2*( (1-gv.PHI_EXT * 180) % 180) - 180, 0, 10, ls='--', color='b') 
    
plt.xlabel('Time (s)') 
plt.ylabel('Bump Phase (°)')
plt.yticks([-180, -90, 0, 90, 180])
add_vlines()    
    
ax = fig.add_subplot(2,2,3) 

BL_rates = np.mean(np.nanmean(rates[0:8], axis=0), axis=-1) # over time

avg_rates = np.nanmean(rates[int(4/gv.T_WINDOW):int(5/gv.T_WINDOW)], axis=0) # over time 
for i_pop in range(gv.n_pop-1): 
    
    pop_rates = avg_rates[i_pop] 
    pop_rates = pop_rates[~np.isnan(pop_rates)] 
    
    smooth_avg_rates = circular_convolution(pop_rates, int(pop_rates.shape[0]*.01) ) 
    # smooth_avg_rates = np.flip(smooth_avg_rates, axis=-1) 
    avg_m1, avg_phi = decode_bump(smooth_avg_rates) 
    
    smooth_avg_rates = np.roll(smooth_avg_rates, int((avg_phi/np.pi - 0.5 ) *gv.n_size[i_pop])) 
    
    print('BL', BL_rates[i_pop], '[<m0>]', np.mean(pop_rates), '[<m1>]', avg_m1, '[<phi>]', 2*avg_phi*180/np.pi-180) 
    
    theta = np.linspace(-180, 180, gv.n_size[i_pop]) 
    # cos_func =  np.mean(pop_rates) + avg_m1 * np.cos( theta*np.pi/180 - 2*avg_phi ) 
    # plt.plot(theta, cos_func, '--', color=gv.pal[i_pop])         
    # print(theta.shape, cos_func.shape) 
    
    plt.plot(theta, smooth_avg_rates, color=gv.pal[i_pop]) 

plt.xlabel('Prefered Location (°)')
plt.xticks([-180, -90, 0, 90, 180])
plt.ylabel('Rates (Hz)') 

plt.show()
