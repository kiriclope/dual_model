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

alpha = [1,.05] 
gv.init_param() 

time, rates = get_time_rates(path=gv.path)
m0 = np.nanmean(rates, axis=-1) 
avg_m0 = np.nanmean(m0, axis=0) 

m1, phi, smooth_rates = get_m1_phi_smooth_rates(rates, osc=0)
# print(m1_osc) 

if(gv.RANK==2):
    time, rates_perm = get_time_rates(MAP=1, path=gv.path, con_path=gv.con_path) 
    print(rates_perm.shape) 
    m1_perm, phi_perm, smooth_rates_perm = get_m1_phi_smooth_rates(rates_perm) 

    print('smooth_rates_perm', smooth_rates_perm.shape)

if(gv.IF_DPA):
    figtitle = 'm0_m1_phi_DPA_' + gv.folder 
elif(gv.IF_DUAL):
    figtitle = 'm0_m1_phi_DUAL_' + gv.folder
elif(gv.IF_DRT):
    figtitle = 'm0_m1_phi_DRT_' + gv.folder
else:
    figtitle = 'm0_m1_phi_tasks_' + gv.folder

fig = plt.figure(figtitle, figsize=(1.25*1.618*1.5*3, 1.618*1.25*gv.RANK)) 

ax = plt.subplot(gv.RANK,3,1) 
    
# for i_pop in range(gv.n_pop): 
#     plt.plot(time, m0[:,i_pop], lw=2, color=gv.pal[i_pop]) 

# plt.xlabel('Time (s)') 
# plt.ylabel('$\\nu^0$ (Hz)') 
# add_vlines()

time = np.around(time,2)
bins = np.arange(np.where(time==2)[0][0],np.where(time==3)[0][0]) 

for i_pop in range(gv.n_pop): 
    theta = np.linspace(0, np.pi, gv.n_size[i_pop]) 
    ax.plot(theta, np.mean(smooth_rates[i_pop, bins, :gv.n_size[i_pop]], axis=0), '-', color=gv.pal[i_pop], alpha=alpha[i_pop] ) 
    
ax.set_xlabel('$\\theta$ (rad)') 
ax.set_ylabel('$Rates(\\theta_0)$ (Hz)') 
plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
           ['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$'])

ax = plt.subplot(gv.RANK,3,2)

for i_pop in range(gv.n_pop) : 
    plt.plot(time, m1[i_pop], '-', color=gv.pal[i_pop], alpha=alpha[i_pop]) 

plt.xlabel('Time (s)') 
plt.ylabel('Amplitude (Hz)') 
add_vlines()

ax = plt.subplot(gv.RANK,3,3) 

def phitoPi(time, phi):

    new_time = []
    new_phi = [] 

    new_time.append(time[0]) 
    new_phi.append(phi[0]) 
    
    for i in range(phi.shape[0]-1):
        
        if phi[i]>phi[i+1] and phi[i+1]<=np.pi/8 :
            new_time.append((time[i+1]+time[i])/2.0) 
            new_phi.append(0) 
            
        elif phi[i]<phi[i+1] and phi[i+1]>=7*np.pi/8 :
            new_time.append((time[i+1]+time[i])/2.0) 
            new_phi.append(0) 
        else:
            new_time.append(time[i+1]) 
            new_phi.append(phi[i+1]) 

    return new_time, new_phi

for i_pop in range(gv.n_pop) :
    # plt.plot(time, phi[i_pop], color=gv.pal[i_pop], alpha=alpha[i_pop])
    
    time2, phi2 = phitoPi(time, phi[i_pop])
    plt.plot(time2, phi2, color=gv.pal[i_pop], alpha=alpha[i_pop]) 
    
plt.xlabel('Time (s)') 
plt.ylabel('Phase (rad)') 
# plt.yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
#            ['$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2 \pi$']) 
plt.yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
           ['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$'])

add_vlines()    

if(gv.RANK==2):

    ax = plt.subplot(2,3,4) 
    
    for i_pop in range(gv.n_pop): 
        theta = np.linspace(0, np.pi, gv.n_size[i_pop]) 
        ax.plot(theta, np.mean(smooth_rates_perm[i_pop, bins, :gv.n_size[i_pop]], axis=0), '-', color=gv.pal[i_pop], alpha=alpha[i_pop]) 
    
    ax.set_xlabel('$\phi$ (rad)') 
    ax.set_ylabel('$Rates(\phi)$ (Hz)') 
    plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
               ['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$']) 
        
    ax = plt.subplot(2,3,5) 
    
    for i_pop in range(gv.n_pop) : 
        plt.plot(time, m1_perm[i_pop], '-', color=gv.pal[i_pop]) 
    
    plt.xlabel('Time (s)') 
    plt.ylabel('Amplitude (Hz)') 
    add_vlines() 
    
    ax = plt.subplot(2,3,6) 

    for i_pop in range(gv.n_pop) : 
        plt.plot(time, phi_perm[i_pop], color=gv.pal[i_pop], alpha=alpha[i_pop]) 
    
    plt.xlabel('Time (s)') 
    plt.ylabel('Phase (rad)') 
    plt.yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
               ['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$'])
    
    # plt.yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
    #            ['$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2 \pi$']) 
    
    add_vlines() 

plt.show()
