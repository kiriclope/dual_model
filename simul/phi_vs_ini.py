import sys, os, importlib
from importlib import reload 
from scipy.signal import savgol_filter 

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from joblib import Parallel, delayed

import params as gv 
importlib.reload(sys.modules['params']) 
import progressbar as pgb 

from write import replace_global
from get_m1 import *
from utils import *

gv.init_param()

ini_list = np.arange(0, 11, 1) 

def parloop(path, ini_id, con_path=gv.con_path, MAP=0, verbose=0): 
    
    path += '/ini_cond_%d' % ini_id ; 
    
    if(verbose):
        print(path)
        
    try:
        time, rates = get_time_rates(MAP, path, con_path)
        if(verbose):
            print('time', time.shape, 'rates', rates.shape) 
        
        m0 = np.nanmean(rates, axis=-1).T # mean over neurons 
        
        if(verbose): 
            print('m0', m0.shape) 
    
        m1, phi, smooth_rates = get_m1_phi_smooth_rates(rates)
        if(verbose):
            print('m1', m1.shape, 'phi', phi.shape)

        m1 = circular_convolution(m1, int(m1.shape[-1]*.25) ) # over time 
        phi = circular_convolution(phi, int(phi.shape[-1]*.25) ) # over time
        
        return time, m0, m1, phi
    
    except : 
        pass 
    
parloop(gv.path, 0, verbose=1) 

with pgb.tqdm_joblib( pgb.tqdm(desc='computing m1 and phi', total=ini_list.shape[0] ) ) as progress_bar: 
    
    time_list, m0_list, m1_list, phi_list = zip( *Parallel(n_jobs=-25)(delayed(parloop)(gv.path, ini_id, MAP=gv.MAP) 
                                                                       for ini_id in ini_list ) ) 
    
time_list = np.array(time_list)
m0_list = np.array(m0_list) 
m1_list = np.array(m1_list) 
phi_list = np.array(phi_list) 

m0_list = np.swapaxes(m0_list, 0, 1) 
m1_list = np.swapaxes(m1_list, 0, 1) 
phi_list = np.swapaxes(phi_list, 0, 1) 

print('time', time_list.shape, 'm0', m0_list.shape, 'm1', m1_list.shape, 'phi', phi_list.shape) 

avg_m0 = np.nanmean(m0_list, axis=-1) 
avg_m1 = np.nanmean(m1_list, axis=-1) 
avg_phi = np.nanmean(phi_list, axis=-1) 

print('m0', np.mean(avg_m0, axis=-1), 'm1', np.mean(avg_m1, axis=-1), 'phi', np.mean(avg_phi, axis=-1))

figtitle = 'phi_vs_ini_' + gv.folder + '_MAP_%d' % gv.MAP 
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(1.25*1.618*1.5*3, 1.618*1.5*2), num=figtitle) 

ax1.set(ylim=(0, 20)) 
ax2.set(ylim=(0, 1)) 
ax3.set(ylim=(0, np.pi)) 

for i_pop in range(gv.n_pop): 
    ax1.plot(time_list.T, m0_list[i_pop].T, color= gv.pal[i_pop], alpha=0.25) 
ax1.set_xlabel('Time (s)') 
ax1.set_ylabel('$\\nu^{(0)}_%d$ (Hz)' % gv.MAP) 

for i_pop in range(gv.n_pop): 
    ax2.plot(time_list.T, m1_list[i_pop].T, color= gv.pal[i_pop], alpha=0.25) 
ax2.set_xlabel('Time (s)') 
ax2.set_ylabel('$\\nu^{(1)}_%d$ (Hz)' % gv.MAP) 

for i_pop in range(gv.n_pop):
    ax3.plot(time_list.T, phi_list[i_pop].T, color= gv.pal[i_pop], alpha=0.25) 
ax3.set_xlabel('Time (s)') 
ax3.set_ylabel('$\phi_%d$ (rad)' % gv.MAP) 
ax3.set_yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]) 
ax3.set_yticklabels(['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$']) 

for i_pop in range(gv.n_pop): 
    ax4.hist(avg_m0[i_pop], color= gv.pal[i_pop], alpha=0.25) 
ax4.set_xlabel('$\\nu^{(0)}_%d$ (Hz)' % gv.MAP) 
ax4.set_ylabel('Count') 

for i_pop in range(gv.n_pop): 
    ax5.hist(avg_m1[i_pop], color= gv.pal[i_pop], alpha=0.25) 
ax5.set_xlabel('$\\nu^{(1)}_%d $ (Hz)' % gv.MAP) 
ax5.set_ylabel('Count') 

for i_pop in range(gv.n_pop): 
    ax6.hist(avg_phi[i_pop], color= gv.pal[i_pop], alpha=0.25) 
ax6.set_ylabel('$\phi_%d$ (rad)' % gv.MAP ) 
ax6.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]) 
ax6.set_xticklabels(['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$']) 

plt.show() 
