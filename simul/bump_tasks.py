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

alpha = [1,.05] 
gv.init_param() 

time, rates = get_time_rates(path=gv.path)
m0 = np.nanmean(rates, axis=-1) 
avg_m0 = np.nanmean(m0, axis=0) 

m1, phi, smooth_rates = get_m1_phi_smooth_rates(rates) 

if(gv.RANK==2):
    time, rates_perm = get_time_rates(MAP=1, path=gv.path, con_path=gv.con_path) 
    m1_perm, phi_perm, smooth_rates_perm = get_m1_phi_smooth_rates(rates_perm) 

if(gv.IF_DPA):
    figtitle = 'm0_m1_phi_DPA_' + gv.folder 
elif(gv.IF_DUAL):
    figtitle = 'm0_m1_phi_DUAL_' + gv.folder
elif(gv.IF_DRT):
    figtitle = 'm0_m1_phi_DRT_' + gv.folder
else:
    figtitle = 'm0_m1_phi_tasks_' + gv.folder

# fig, ax = plt.subplots(4, gv.RANK, figsize=(1.25*1.618*1.5*gv.RANK, 1.618*1.5*4), num=figtitle) 
fig, ax = plt.subplots(3, gv.RANK, figsize=set_size(200, n_subplots=[2,4]), num=figtitle) 

print(ax.shape)
epochs = ['Sample', 'Distractor', 'Test'] 
time = np.around(time,2)
print(time.shape)

plt.suptitle('Activity profile')
ax[0][0].set_title('Sample Map')
ax[0][1].set_title('Distractor Map')

for i_epoch in range(len(epochs)):
    # if i_epoch==0:
    #     bins = np.arange(np.where(time==1)[0][0],np.where(time==2)[0][0]) 
    if i_epoch==0:
        bins = np.arange(np.where(time==2)[0][0],np.where(time==3)[0][0]) 
    if i_epoch==1:
        bins = np.arange(np.where(time==4.5)[0][0],np.where(time==5.5)[0][0]) 
    if i_epoch==2:
        bins = np.arange(np.where(time==9)[0][0],np.where(time==10)[0][0]) 
        
    for i_pop in range(gv.n_pop): 
        theta = np.linspace(0, np.pi, gv.n_size[i_pop]) 
        ax[i_epoch][0].plot(theta, np.mean(smooth_rates[i_pop, bins, :gv.n_size[i_pop]], axis=0), '-', color=gv.pal[i_pop]) 

    ax[i_epoch][0].set_xlabel('$\\theta$ (rad)') 
    ax[i_epoch][0].set_ylabel('Activity profile (Hz)') 
    ax[i_epoch][0].set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]) 
    ax[i_epoch][0].set_xticklabels(['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$']) 
    
    if gv.RANK==2:
        for i_pop in range(gv.n_pop): 
            theta = np.linspace(0, np.pi, gv.n_size[i_pop]) 
            ax[i_epoch][1].plot(theta, np.mean(smooth_rates_perm[i_pop, bins, :gv.n_size[i_pop]], axis=0), '-', color=gv.pal[i_pop]) 
    

        ax[i_epoch][1].set_xlabel('$\phi$ (rad)') 
        # ax[i_epoch][1].set_ylabel('Activity profile (Hz)') 
        ax[i_epoch][1].set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]) 
        ax[i_epoch][1].set_xticklabels(['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$']) 
