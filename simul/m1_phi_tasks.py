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
from rank_utils import * 
from plot_settings import *
SetPlotParams()

alpha = [1,.0] 
gv.init_param() 

time, rates = get_time_rates(path=gv.path)
m0 = np.nanmean(rates, axis=-1).T
avg_m0 = np.nanmean(m0, axis=-1) 

m1, phi, smooth_rates = get_m1_phi_smooth_rates(rates, osc=0) 

print('m0', avg_m0, 'm1', np.mean(m1, axis=-1), 'smooth rates', smooth_rates.shape ) 

if(gv.RANK==2):
    time, rates_perm = get_time_rates(MAP=1, path=gv.path, con_path=gv.con_path) 
    print(rates_perm.shape) 
    m1_perm, phi_perm, smooth_rates_perm = get_m1_phi_smooth_rates(rates_perm) 
    print('smooth_rates_perm', smooth_rates_perm.shape) 

figtitle = 'm1_phi_task_' + gv.folder 
if(gv.IF_DPA):
    figtitle+= '_DPA'
if(gv.IF_DUAL):
    figtitle+= '_DUAL'

# fig = plt.figure(figtitle, figsize=(1.25*1.618*1.5*2, 1.618*1.25*gv.RANK) ) 
fig = plt.figure(figtitle, figsize=set_size(125*2, n_subplots=[2,2]) ) 

ax = plt.subplot(gv.RANK,2,1)
plt.suptitle('Sample map') 

for i_pop in range(gv.n_pop) : 
    plt.plot(time, m1[i_pop], '-', color=gv.pal[i_pop] ) 

plt.xlabel('Time (s)') 
plt.ylabel('Amplitude (Hz)') 
plt.xlim([0, 14])
add_vlines() 

ax = plt.subplot(gv.RANK,2,2) 

for i_pop in range(gv.n_pop) :
    # plt.plot(time, phi[i_pop], color=gv.pal[i_pop], alpha=alpha[i_pop]) 
    time2, phi2 = phitoPi(time, phi[i_pop])
    plt.plot(time2, phi2, color=gv.pal[i_pop], alpha=alpha[i_pop]) 
    
plt.xlabel('Time (s)') 
plt.ylabel('Phase (rad)') 
plt.yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
           ['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$'])
plt.ylim([0, np.pi])
plt.xlim([0, 14])
plt.hlines(gv.PHI_EXT/2, 0, 14, ls='--')

add_vlines() 

if(gv.RANK==2):
    
    ax = plt.subplot(2,2,3) 
    ax.set_title('Distractor map')
    
    for i_pop in range(gv.n_pop) : 
        plt.plot(time, m1_perm[i_pop], '-', color=gv.pal[i_pop]) 
    
    plt.xlabel('Time (s)') 
    plt.ylabel('Amplitude (Hz)') 
    add_vlines() 
    plt.xlim([0, 14])
    
    ax = plt.subplot(2,2,4) 

    for i_pop in range(gv.n_pop) : 
        plt.plot(time, phi_perm[i_pop], color=gv.pal[i_pop], alpha=alpha[i_pop]) 
    
    plt.xlabel('Time (s)') 
    plt.ylabel('Phase (rad)')
    plt.yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
               ['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$'])
    plt.xlim([0, 14])
    plt.ylim([0, np.pi])
    add_vlines() 
    plt.hlines(0, 0, 14, ls='--') 
plt.show()
