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
# m1, phi, smooth_rates = get_m1_phi_smooth_rates(rates[:int(5/gv.T_WINDOW)])

figtitle = 'spatial_profile_' + gv.folder 

if gv.folder.find('off')!=-1 or gv.folder.find('on')!=-1:
    figtitle = 'spatial_profile'

fig = plt.figure(figtitle, figsize=(2.427*3, 1.5))

BL_rates = np.nanmean(rates[0:int(2/gv.T_WINDOW)], axis=0) # over time 
stim_rates = np.nanmean(rates[int(2/gv.T_WINDOW):int(3/gv.T_WINDOW)], axis=0) # over time 
delay_rates = np.nanmean(rates[int(6/gv.T_WINDOW):int(8/gv.T_WINDOW)], axis=0) # over time
# delay_rates = np.nanmean(rates[-int(3/gv.T_WINDOW):], axis=0) # over time 

# print('rates', np.mean(rates[int(2/(gv.T_WINDOW)-3),0]), np.mean(rates[int(2/(gv.T_WINDOW)-2),0]), np.mean(rates[int(2/(gv.T_WINDOW)-1),0]), np.mean(rates[int(2/(gv.T_WINDOW)),0])) 

i_pop=0
theta = np.linspace(-180, 180, gv.n_size[i_pop]) 

ax = fig.add_subplot(1,3,1) 

pop_rates = BL_rates[i_pop] 
pop_rates = pop_rates[~np.isnan(pop_rates)] 

smooth_BL_rates = circular_convolution(pop_rates, int(pop_rates.shape[0]*.01) ) 
BL_m1, BL_phi = decode_bump(smooth_BL_rates) 
# smooth_BL_rates = np.roll(smooth_BL_rates, int((BL_phi/np.pi - 0.5 ) *gv.n_size[i_pop])) 

print('BL', BL_rates[i_pop], '[<m0>]', np.mean(pop_rates), '[<m1>]', BL_m1, '[<phi>]', 2*BL_phi*180/np.pi-180) 

plt.plot(theta, smooth_BL_rates, color=gv.pal[0]) 

plt.title('Baseline')
plt.xlabel('Prefered Location (°)')
plt.xticks([-180, -90, 0, 90, 180])
plt.ylabel('Rates (Hz)') 
plt.ylim([0, 30])

ax = fig.add_subplot(1,3,2) 

pop_rates = stim_rates[i_pop] 
pop_rates = pop_rates[~np.isnan(pop_rates)] 

smooth_stim_rates = circular_convolution(pop_rates, int(pop_rates.shape[0]*.01) ) 
stim_m1, stim_phi = decode_bump(smooth_stim_rates)  
smooth_stim_rates = np.roll(smooth_stim_rates, int((stim_phi/np.pi - 0.5 ) *gv.n_size[i_pop])) 

print('BL', BL_rates[i_pop], '[<m0>]', np.mean(pop_rates), '[<m1>]', stim_m1, '[<phi>]', 2*stim_phi*180/np.pi-180) 

plt.plot(theta, smooth_stim_rates, color=gv.pal[0]) 

plt.title('Stimulation')
plt.xlabel('Prefered Location (°)')
plt.xticks([-180, -90, 0, 90, 180])
plt.ylabel('Rates (Hz)') 
plt.ylim([0, 30])

ax = fig.add_subplot(1,3,3) 
    
pop_rates = delay_rates[i_pop] 
pop_rates = pop_rates[~np.isnan(pop_rates)] 

smooth_delay_rates = circular_convolution(pop_rates, int(pop_rates.shape[0]*.01) ) 
delay_m1, delay_phi = decode_bump(smooth_delay_rates)

smooth_delay_rates = np.roll(smooth_delay_rates, int((delay_phi/np.pi - 0.5 ) *gv.n_size[i_pop])) 

print('BL', BL_rates[i_pop], '[<m0>]', np.mean(pop_rates), '[<m1>]', delay_m1, '[<phi>]', 2*delay_phi*180/np.pi-180) 

plt.plot(theta, smooth_delay_rates, color=gv.pal[0]) 
plt.title('Delay')
plt.xlabel('Prefered Location (°)')
plt.xticks([-180, -90, 0, 90, 180])
plt.ylabel('Rates (Hz)') 
plt.ylim([0, 30])
plt.show()

