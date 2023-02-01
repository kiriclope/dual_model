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

gv.IF_SPEC=0 
gv.IF_LOW_RANK=0 
gv.IF_INI_COND=0
gv.IF_TRIALS=0
gv.init_param() 

NORM_M1=0 
gain_list = np.arange(0, 1.1, .1) 
trial_list = np.arange(0, 21, 1) 

MEAN_KSI = gv.MEAN_KSI 
VAR_KSI = gv.VAR_KSI 
print(gv.path) 

def parloop(gain, trial, path=gv.path, folder=gv.folder, MAP=gv.MAP, con_path=gv.con_path, RANK=gv.RANK, verbose=0) : 
    
    path += '/gain_%.1f' % gain ; 
    con_path += '/gain_%.1f' % gain ; 
    
    path += '/trial_%d' % trial ; 
    con_path += '/trial_%d' % trial ; 
    
    if verbose: 
        print(path) 
        
    try: 
        time, rates = get_time_rates(MAP, path, con_path) 
        
        if(verbose): 
            print('time', time.shape, 'rates', rates.shape) 
        
        avg_rates = np.nanmean(rates.copy(), axis=0) # avg over time 
        m0 = np.nanmean(avg_rates.copy(), axis=-1) # mean over neurons 
        
        if verbose: 
            print('m0', m0, '<m0>', avg_rates.shape) 
        
        m1, phi, smooth_rates = get_m1_phi_smooth_rates(rates.copy()) 
        
        m1 = np.nanmean(m1, axis=-1) 
        phi = np.nanmean(phi, axis=-1) 
        
        # m1, phi, smooth_rates= get_avg_m1_phi_smooth_rates(avg_rates) 
        
        if verbose: 
            print('m1', m1) 
            
    except :
        m0 = np.zeros(2) * np.nan 
        m1 = np.zeros(2) * np.nan 
        phi = np.zeros(2) * np.nan 
    
    return m0, m1, phi

parloop(0, 0, verbose=1) 

with pgb.tqdm_joblib( pgb.tqdm(desc='computing m0 and m1', total=int(gain_list.shape[0] * trial_list.shape[0]) ) ) as progress_bar: 
    
    m0_list, m1_list, phi_list = zip( *Parallel(n_jobs=-1, backend="multiprocessing")(delayed(parloop)( gain, trial, gv.path, gv.folder) 
                                                           for gain in gain_list 
                                                           for trial in trial_list ) ) 

m0_list = np.array(m0_list).T 
m1_list = np.array(m1_list).T 
phi_list = np.array(phi_list).T 

m0_list = m0_list.reshape(gv.n_pop, gain_list.shape[0], trial_list.shape[0]) 
m1_list = m1_list.reshape(gv.n_pop, gain_list.shape[0], trial_list.shape[0]) 
phi_list = phi_list.reshape(gv.n_pop, gain_list.shape[0], trial_list.shape[0]) 

print('m0', m0_list.shape, 'm1', m1_list.shape, 'phi', phi_list.shape)
print(m0_list)
                
mean_m0 = np.nanmean(m0_list, axis=-1) 
mean_m1 = np.nanmean(m1_list, axis=-1) 
mean_phi = np.nanmean(phi_list, axis=-1) 

std_m0 = np.nanstd(m0_list, axis=-1) 
std_m1 = np.nanstd(m1_list, axis=-1) 
std_phi = np.nanstd(phi_list, axis=-1) 

print('m0', mean_m0) 
print('m1', mean_m1) 

figname = 'm0_m1_phi_gain_' + gv.folder 
fig, axis = plt.subplots(1, 3, figsize=(1.25*1.618*1.5*3, 1.618*1.5), num=figname) 

axis[0].set(ylim=(0, np.ceil( np.nanmax(m0_list)*12 ) /10 ) ) 
axis[0].set_xlabel('gain (a.u.)') 
if(gv.model=='lif'): 
    axis[0].set_ylabel('$\\nu^{(0)}$ (Hz)') 
if(gv.model=='binary'): 
    axis[0].set_ylabel('$m^{(0)}$ (a.u.)')
        
if(NORM_M1):
    axis[1].set(ylim=(0, np.ceil( np.nanmax(m1_list/m0_list)*12 ) /10 ) )
else:
    axis[1].set(ylim=(0, np.ceil( np.nanmax(m1_list)*12 ) /10 ) ) 
axis[1].set_xlabel('gain (a.u.)') 
if(gv.model=='lif'): 
    if(NORM_M1):
        axis[1].set_ylabel('$\\nu^{(1)} / \\nu^{(0)}$ (Hz)') 
    else:
        axis[1].set_ylabel('$\\nu^{(1)}$ (Hz)') 
if(gv.model=='binary'): 
    if(NORM_M1):
        axis[1].set_ylabel('$m^{(1)}/m^{(0)}$')
    else:
        axis[1].set_ylabel('$m^{(1)}$') 

# axis[1].legend(loc='center left', bbox_to_anchor=(1, 0.5)) 

for i_pop in range(gv.n_pop):
    axis[0].plot(gain_list, mean_m0[i_pop], '-o', color = gv.pal[i_pop]) 
    axis[0].plot(gain_list, m0_list[i_pop,:], 'x', color = gv.pal[i_pop], alpha=0.2) 
    axis[0].fill_between(gain_list, mean_m0[i_pop]-std_m0[i_pop], mean_m0[i_pop]+std_m0[i_pop], alpha=.1, color=gv.pal[i_pop])
    
if(NORM_M1):
    for i_pop in range(gv.n_pop):
        axis[1].plot(gain_list, mean_m1[i_pop]/mean_m0[i_pop], '-o', color = gv.pal[i_pop], label=gv.label[i_pop]) 
        axis[1].fill_between(gain_list, (mean_m1[i_pop]-std_m1[i_pop])/mean_m0[i_pop], (mean_m1[i_pop]+std_m1[i_pop])/mean_m0[i_pop], alpha=.1, color=gv.pal[i_pop]) 

    for i in range(m1_list.shape[1]):
        axis[1].plot(gain_list[i] * np.ones(m1_list[0,i].shape[0]) , m1_list[0,i]/mean_m0[0,i], 'x', color = gv.pal[0], alpha=1) 
else:
    for i_pop in range(gv.n_pop):
        axis[1].plot(gain_list, mean_m1[i_pop], '-o', color = gv.pal[i_pop], label=gv.label[i_pop]) 
        axis[1].fill_between(gain_list, mean_m1[i_pop]-std_m1[i_pop], mean_m1[i_pop]+std_m1[i_pop], alpha=.1, color=gv.pal[i_pop]) 
        axis[1].plot(gain_list, m1_list[0,:], 'x', color = gv.pal[0], alpha=1) 
        
axis[2].set(ylim=(0, np.pi ) )
axis[2].set_xlabel('gain (a.u.)')
axis[2].set_ylabel('$\phi $ (rad)') 
axis[2].set_yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]) 
axis[2].set_yticklabels(['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$']) 

# for i_pop in range(gv.n_pop):
#     axis[2].plot(gain_list, mean_phi[i_pop], '-o', color = gv.pal[i_pop], label=gv.label[i_pop]) 
#     axis[2].fill_between(gain_list, mean_phi[i_pop]-std_phi[i_pop], mean_phi[i_pop]+std_phi[i_pop], alpha=.1, color=gv.pal[i_pop]) 

axis[2].plot(gain_list, phi_list[0,:], 'x', color = gv.pal[0], alpha=0.5) 
# axis[2].plot(gain_list, phi_list[0,:], 'x', color = gv.pal[0], alpha=0.5) 

plt.show()
