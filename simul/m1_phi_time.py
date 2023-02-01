import sys, os, importlib
from importlib import reload
from scipy.signal import savgol_filter 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import params as gv 
importlib.reload(sys.modules['params'])
import get_m1
importlib.reload(sys.modules['get_m1']) 

from get_m1 import * 
from utils import * 
from rank_utils import * 

alpha = [1,.05]

gv.init_param() 
        
time, rates = get_time_rates(path=gv.path)
m0 = np.nanmean(rates, axis=-1).T
m1, phi, smooth_rates = get_m1_phi_smooth_rates(rates) 

print('m0', np.nanmean(m0[..., 12:], axis=-1) , 'm1', np.mean(m1[..., 12:], axis=-1),
      'phi', 2 * np.mean(phi[..., 12:], axis=-1) * 180 / np.pi - 180, 'smooth rates', smooth_rates.shape ) 

if(gv.RANK==2):
    time, rates_perm = get_time_rates(MAP=1, path=gv.path, con_path=gv.con_path) 
    print(rates_perm.shape) 
    m1_perm, phi_perm, smooth_rates_perm = get_m1_phi_smooth_rates(rates_perm) 
    print('smooth_rates_perm', smooth_rates_perm.shape) 
    
figtitle = 'm1_phi_time' + gv.folder 
if gv.folder.find('off')!=-1 or gv.folder.find('on')!=-1:
    figtitle = 'm1_phi_time'
    
fig = plt.figure(figtitle, figsize=(2.427*2, 1.5*2))

ax = plt.subplot(2,2,1)

for i_pop in range(gv.n_pop-1) : 
    plt.plot(time, m0[i_pop], '-', color=gv.pal[i_pop], alpha=alpha[i_pop])

plt.xlabel('Time (s)') 
plt.ylabel('Rates (Hz)')
# plt.xlim([0, 10]) 
add_vlines()

ax = plt.subplot(2,2,2)

for i_pop in range(gv.n_pop-1) : 
    plt.plot(time, m1[i_pop], '-', color=gv.pal[i_pop], alpha=alpha[i_pop])

plt.xlabel('Time (s)') 
plt.ylabel('Bump Amplitude (Hz)')
# plt.xlim([0, 8]) 
add_vlines()

ax = plt.subplot(2,2,3)

for i_pop in range(gv.n_pop-1) : 
    plt.plot(time, m1[i_pop] / m0[i_pop], '-', color=gv.pal[i_pop], alpha=alpha[i_pop])

plt.xlabel('Time (s)') 
plt.ylabel('Rel. Bump Amplitude (Hz)')
# plt.xlim([0, 8]) 
plt.ylim([0, 1.5]) 
add_vlines()

ax = plt.subplot(2,2,4) 

for i_pop in range(gv.n_pop-1) :
    # plt.plot(time, 2*phi[i_pop]*180/np.pi - 2*((gv.PHI_CUE * 180) % 180), color=gv.pal[i_pop], alpha=alpha[i_pop])
    plt.plot(time, 2*phi[i_pop]*180/np.pi - 180, color=gv.pal[i_pop], alpha=alpha[i_pop])
    
    if(gv.IF_TRIALS and gv.IF_INI_COND):
        plt.hlines( 2*( (gv.TRIAL_ID * 180 / gv.N_TRIALS) % 180) - 180, 0, 8, ls='--') 
        # plt.hlines( 2*( (180 - gv.TRIAL_ID * 180 / gv.N_TRIALS) % 180) - 180, 0, 8, color='b', ls='--') 
        print('phi', 2*np.mean(phi[i_pop, -4:]) * 180 / np.pi - 180, 'phi_trial', 2*(gv.TRIAL_ID * 180 / gv.N_TRIALS) % 180 - 180 ) 
    else:
        plt.hlines( 2*((gv.PHI_CUE * 180) % 180) - 180, 0, 8, ls='--') 
        # plt.hlines( 2*((180-gv.PHI_CUE * 180) % 180) - 180, 0, 8, color='b', ls='--') 
        print('phi', 2*np.mean(phi[i_pop, -4:]) * 180 / np.pi - 180, 'phi_ext', 2*( (gv.PHI_CUE * 180) % 180) - 180) 

plt.xlabel('Time (s)')
# plt.xlim([0, 8]) 
plt.ylabel('Bump Phase (°)')
plt.ylim([-180, 180])
plt.yticks([-180, -90, 0, 90, 180])
# plt.ylabel('Drift (°)')
# plt.xlim([0, 8]) 
# plt.ylim([-30, 30])
# plt.yticks([-30, -15, 0, 15, 30])
add_vlines()

# plt.yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
#            ['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$'])
# plt.xlim([0, 10])

if(gv.RANK==2):

    ax = plt.subplot(2,3,4) 
    
    for i_pop in range(gv.n_pop): 
        theta = np.linspace(0, np.pi, gv.n_size[i_pop]) 
        ax.plot(theta, np.nanmean(smooth_rates_perm[i_pop, :, :gv.n_size[i_pop]], axis=0), '-', color=gv.pal[i_pop], alpha=alpha[i_pop]) 
    
    ax.set_xlabel('$\\theta_1$ (rad)') 
    ax.set_ylabel('$\\nu(\\theta_1)$ (Hz)') 
    plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
               ['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$']) 
    
    ax = plt.subplot(2,3,5) 
    
    for i_pop in range(gv.n_pop) : 
        plt.plot(time, m1_perm[i_pop]/m0[i_pop], '-', color=gv.pal[i_pop]) 
    
    plt.xlabel('Time (s)') 
    plt.ylabel('$\\nu^{(1)}_1 / \\nu^{(0)}_1$ (Hz)') 
    # plt.xlim([0, 10])
    ax = plt.subplot(2,3,6) 

    for i_pop in range(gv.n_pop) : 
        plt.plot(time, phi_perm[i_pop], color=gv.pal[i_pop], alpha=alpha[i_pop]) 
    
    plt.xlabel('Time (s)') 
    plt.ylabel('$\phi_1$ (rad)') 
    plt.yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
               ['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$'])
    # plt.xlim([0, 10])
# plt.savefig(figtitle + '.svg', dpi=300)
        
# plt.show()
