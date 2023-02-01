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
from utils import *

gv.IF_INI_COND=0
gv.IF_TRIALS=0

gv.RANK=1 
gv.MAP=0

gv.init_param() 

if(gv.IF_STP): 
    Jee_list = np.arange(2.0, 3.00, .05) 
    Jee_list = np.arange(0.2, 2.1, .1) 
else: 
    Jee_list = np.arange(0.5, 1.05, .05) 
    Jee_list = np.arange(0., 1.1, .1) 
    
ini_list = np.arange(0, 20+1, 1) 

def parloop(Jee, ini, path=gv.path, folder=gv.folder, MAP=gv.MAP, con_path=gv.con_path, verbose=0) : 
    
    # new_folder = folder + '_Jee_%.3f' % Jee 
    new_folder ='I0_1.00_J0_%.2f' % Jee 
    new_path = path.replace(folder, new_folder) 
    new_path += '/ini_cond_%d' % ini 
    
    if verbose: 
        print(new_path) 
    
    try: 
        time, rates = get_time_rates(MAP, new_path, con_path) 
        
        if(verbose): 
            print('time', time.shape, 'rates', rates.shape) 
            print(con_path) 
        
        avg_rates = np.nanmean(rates.copy(), axis=0) # avg over time 
        m0 = np.nanmean(avg_rates.copy(), axis=-1) # mean over neurons 
        
        if verbose: 
            print('m0', m0, '<m0>', avg_rates.shape) 
            
        m1 = get_overlap(rates.copy(), con_path=con_path, MAP=0)        
        m1 = np.nanmean(m1, axis=-1) 
                
        if verbose:
            print('m1', m1) 
            
    except :
        m0 = np.zeros(1) * np.nan 
        m1 = np.zeros(1) * np.nan 
            
    return m0, m1

parloop(0.1,0, verbose=1) 

with pgb.tqdm_joblib( pgb.tqdm(desc='computing m0 and m1', total=int(Jee_list.shape[0] * ini_list.shape[0]) ) ) as progress_bar: 
    
    m0_list, m1_list = zip( *Parallel(n_jobs=-1, backend="multiprocessing")(delayed(parloop)( Jee, ini, gv.path, gv.folder) 
                                                                            for Jee in Jee_list 
                                                                            for ini in ini_list ) ) 

m0_list = np.array(m0_list).T 
m1_list = np.array(m1_list).T 

m0_list = m0_list.reshape(gv.n_pop, Jee_list.shape[0], ini_list.shape[0]) 
m1_list = m1_list.reshape(gv.n_pop, Jee_list.shape[0], ini_list.shape[0]) 
                
mean_m0 = np.nanmean(m0_list,axis=-1) 
mean_m1 = np.nanmean(m1_list,axis=-1) 

std_m0 = np.nanstd(m0_list,axis=-1) 
std_m1 = np.nanstd(m1_list,axis=-1) 

print('m0', mean_m0) 
print('m1', mean_m1) 

figname = 'm0_m1_phi_Jee_' + gv.folder 
fig, axis = plt.subplots(1, 2, figsize=(1.25*1.618*1.5*2, 1.618*1.5), num=figname) 

axis[0].set(ylim=(0, np.ceil( np.nanmax(m0_list)*12 ) /10 ) ) 
axis[0].set_xlabel('$J_{EE}$ (a.u.)') 
axis[0].set_ylabel('$\\nu^{(0)}$ (Hz)') 
        
# axis[1].set(ylim=(0, np.ceil( np.nanmax(m1_list)*12 ) /10 ) )     
axis[1].set_xlabel('$J_{EE}$ (a.u.)')
axis[1].set_ylabel('$\\nu^{(1)}$ (Hz)') 

for i_pop in range(gv.n_pop):
    axis[0].plot(Jee_list, mean_m0[i_pop], '-o', color = gv.pal[i_pop]) 
    axis[0].plot(Jee_list, m0_list[i_pop,:], 'x', color = gv.pal[i_pop], alpha=0.2) 
    axis[0].fill_between(Jee_list, mean_m0[i_pop]-std_m0[i_pop], mean_m0[i_pop]+std_m0[i_pop], alpha=.1, color=gv.pal[i_pop])
    
for i_pop in range(gv.n_pop):
    axis[1].plot(Jee_list, mean_m1[i_pop], '-o', color = gv.pal[i_pop], label=gv.label[i_pop]) 
    axis[1].fill_between(Jee_list, mean_m1[i_pop]-std_m1[i_pop], mean_m1[i_pop]+std_m1[i_pop], alpha=.1, color=gv.pal[i_pop]) 
    axis[1].plot(Jee_list, m1_list[0,:], 'x', color = gv.pal[0], alpha=1) 
        
plt.show() 
