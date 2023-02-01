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

gv.IF_STP = 1 
gv.IF_SPEC = 1 
gv.IF_INI_COND=0
gv.IF_TRIALS=0
gv.HYSTERESIS=0

gv.init_param() 

J0_list = np.arange(0.001, .02, 0.001)
print(J0_list)
ini_list = np.arange(0, 11, 1) 

def parloop(J0, ini, path=gv.path, folder=gv.folder, verbose=0) : 
    
    new_path = path + '/m0_%.3f' % J0 
    new_path += '/ini_cond_%d' % ini ; 
    
    if verbose: 
        print(new_path) 
    
    try : 
        
        time, rates = get_time_rates(MAP=0, path=new_path) 
        if(verbose): 
            print('time', time.shape, 'rates', rates.shape) 
        
        avg_rates = np.nanmean(rates, axis=0) # avg over time 
        m0 = np.nanmean(avg_rates, axis=-1) # mean over neurons 
        
        if verbose: 
            print('m0', m0, '<m0>', avg_rates.shape) 
        
        m1, phi, smooth_rates = get_avg_m1_phi_smooth_rates(avg_rates) 
        
        if verbose: 
            print('m1', m1) 
            
    except :
        
        m0 = np.zeros(2) * np.nan 
        m1 = np.zeros(2) * np.nan 
        
    return m0, m1 

parloop(0.005, 0, verbose=1) 

with pgb.tqdm_joblib( pgb.tqdm(desc='computing m0 and m1', total=int(J0_list.shape[0] * ini_list.shape[0]) ) ) as progress_bar: 
    
    m0_list, m1_list = zip( *Parallel(n_jobs=-2)(delayed(parloop)(J0, ini, verbose=0) 
                                                 for J0 in J0_list 
                                                 for ini in ini_list ) ) 

m0_list = np.array(m0_list).T 
m1_list = np.array(m1_list).T 

m0_list = m0_list.reshape(gv.n_pop, J0_list.shape[0], ini_list.shape[0]) 
m1_list = m1_list.reshape(gv.n_pop, J0_list.shape[0], ini_list.shape[0]) 

mean_m0 = np.mean(m0_list,axis=-1) 
mean_m1 = np.mean(m1_list,axis=-1) 

std_m0 = np.std(m0_list,axis=-1) 
std_m1 = np.std(m1_list,axis=-1) 

print('m0', mean_m0) 
print('m1', mean_m1) 

figtitle = 'm0_m1_J0_' + gv.folder 
fig = plt.figure(figtitle, figsize=(1.25*1.618*1.5*2, 1.618*1.25)) 

ax = fig.add_subplot(1,2,1) 
for i_pop in range(gv.n_pop): 
    plt.plot(J0_list, mean_m0[i_pop], '-o', color = gv.pal[i_pop]) 
    plt.fill_between(J0_list, mean_m0[i_pop]-std_m0[i_pop], mean_m0[i_pop]+std_m0[i_pop], alpha=.1, color=gv.pal[i_pop]) 
    
plt.xlabel('$\\nu_{ext}$') 
plt.ylabel('$ \\nu_0$ (Hz)') 

ax = fig.add_subplot(1,2,2) 
for i_pop in range(gv.n_pop): 
    plt.plot(J0_list, mean_m1[i_pop], '-o', color = gv.pal[i_pop], label=gv.label[i_pop]) 
    plt.plot(J0_list, m1_list[i_pop,:], 'o', color = gv.pal[i_pop], alpha=0.2) 
    plt.fill_between(J0_list, mean_m1[i_pop]-std_m1[i_pop], mean_m1[i_pop]+std_m1[i_pop], alpha=.1, color=gv.pal[i_pop]) 
    
plt.xlabel('$\\nu_{ext}$') 
plt.ylabel('$ \\nu_1$ (Hz)') 

ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) 

plt.show() 
