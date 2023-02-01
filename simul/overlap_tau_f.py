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
from plot_settings import SetPlotParams
SetPlotParams()

gv.IF_SPEC=0
gv.IF_LOW_RANK=0
gv.IF_STP=0
gv.IF_INI_COND=0
gv.IF_TRIALS=0

gv.RANK=2
gv.MAP=0

gv.KAPPA=10
gv.init_param() 

tau_list = np.arange(200, 1300, 100) 
# tau_list = np.arange(0.05, .55, .05) 
ini_list = np.arange(0, 11, 1) 

MEAN_KSI = gv.MEAN_KSI 
VAR_KSI = gv.VAR_KSI 

def parloop(tau_f, ini, path=gv.path, RANK=gv.RANK, MAP=gv.MAP, verbose=0, con_path=gv.con_path, kappa=gv.KAPPA, tau_r=gv.TAU_REC, use=gv.USE, ksi_path=gv.ksi_path, n_size=gv.n_size) : 
    
    path += '/STP/Tf_%d_Tr_%d_U_%.2f' % (tau_f, tau_r, use)          
    # path += '/STP/Tf_%d_Tr_%d_U_%.2f' % (750, tau_r, tau_f) 
   
    if(RANK==1): 
        path += '/low_rank/kappa_%.2f' % kappa
        ksi_path = con_path + '/low_rank/rank_1/seed_ksi_2'
        con_path += '/low_rank/kappa_%.2f' % kappa
    if(RANK==2): 
        path += '/low_rank/kappa_%.2f_kappa_1_%.2f' % (14, 12)
        ksi_path = con_path + '/low_rank/rank_2/seed_ksi_2'
        con_path += '/low_rank/kappa_%.2f_kappa_1_%.2f' % (14, 12)
        
    # print(ksi_path) 
    path += '/ini_cond_%d' % ini ; 
    
    # path += '/seed_%d' % 2 
    # con_path += '/seed_%d' % 2 
    
    if verbose: 
        print(path) 
    
    try :
        time, rates = get_time_rates(0, path, con_path) 
        if(verbose):
            print('time', time.shape, 'rates', rates.shape)
        
        avg_rates = np.nanmean(rates, axis=0) # avg over time         
        mean_rates = np.nanmean(avg_rates, axis=-1) # mean over neurons 
        
        if verbose: 
            print('rates', mean_rates, '<rates>', avg_rates.shape) 
        
        overlap = get_overlap(rates.copy(), ksi_path=ksi_path, MAP=0, n_size=n_size) 
        overlap = np.nanmean(overlap, axis=-1) 

        if(RANK==2):
            overlap_1 = get_overlap(rates.copy(), ksi_path=ksi_path, MAP=1, n_size=n_size) 
            overlap_1 = np.nanmean(overlap_1, axis=-1) 
        else:
            overlap_1 = overlap

        if verbose:
            print('overlap', overlap) 
            
    except :
        
        mean_rates = np.zeros(2) * np.nan 
        overlap = np.zeros(2) * np.nan 
        overlap_1 = np.zeros(2) * np.nan 
    
    return mean_rates, overlap, overlap_1

parloop(200, 0, verbose=1)

with pgb.tqdm_joblib( pgb.tqdm(desc='computing rates and overlap', total=int(tau_list.shape[0] * ini_list.shape[0]) ) ) as progress_bar: 
    
    rates_list, overlap_list, overlap_1_list = zip( *Parallel(n_jobs=-2, backend='multiprocessing')(delayed(parloop)(tau_f, ini, verbose=0) 
                                                                                for tau_f in tau_list for ini in ini_list ) )  

rates_list = np.array(rates_list).T 
overlap_list = np.array(overlap_list).T 
overlap_1_list = np.array(overlap_1_list).T 

rates_list = rates_list.reshape(gv.n_pop, tau_list.shape[0], ini_list.shape[0]) 
overlap_list = overlap_list.reshape(gv.n_pop, tau_list.shape[0], ini_list.shape[0]) 
overlap_1_list = overlap_1_list.reshape(gv.n_pop, tau_list.shape[0], ini_list.shape[0]) 

print('rates', rates_list.shape, 'overlap', overlap_list.shape)

mean_rates = np.nanmean(rates_list,axis=-1) 
mean_overlap = np.nanmean(overlap_list,axis=-1) 
mean_overlap_1 = np.nanmean(overlap_1_list,axis=-1) 

std_rates = np.nanstd(rates_list,axis=-1) 
std_overlap = np.nanstd(overlap_list,axis=-1) 
std_overlap_1 = np.nanstd(overlap_1_list,axis=-1) 

print('<rates>', mean_rates) 
print('<overlap>', mean_overlap) 

figtitle = 'rates_overlap_tau_f_' + gv.folder 
fig = plt.figure(figtitle, figsize=(1.25*1.618*1.5*(1+gv.RANK), 1.618*1.25)) 
ax = fig.add_subplot(1,1+gv.RANK,1) 
for i_pop in range(gv.n_pop):
    plt.plot(tau_list, mean_rates[i_pop], '-o', color = gv.pal[i_pop]) 
    plt.fill_between(tau_list, mean_rates[i_pop]-std_rates[i_pop], mean_rates[i_pop]+std_rates[i_pop], alpha=.1, color=gv.pal[i_pop]) 
    
plt.xlabel('$\\tau_{fac}$ (ms)') 
plt.ylabel('Rates (Hz)') 
# plt.ylim([0, np.ceil(np.nanmax(mean_rates)*12 )/10]) 

ax = fig.add_subplot(1,1+gv.RANK,2) 
for i_pop in range(gv.n_pop):
    plt.plot(tau_list, mean_overlap[i_pop], '-o', color = gv.pal[i_pop]) 
    plt.fill_between(tau_list, mean_overlap[i_pop]-std_overlap[i_pop], mean_overlap[i_pop]+std_overlap[i_pop], alpha=.1, color=gv.pal[i_pop]) 
    
plt.plot(tau_list, overlap_list[0], 'x', color = gv.pal[0])
plt.xlabel('$\\tau_{fac}$ (ms)') 
plt.ylabel('Sample Overlap (Hz)') 
# plt.ylim([0, np.ceil(np.nanmax(mean_overlap)*12 )/10]) 

if(gv.RANK==2):
    ax = fig.add_subplot(1,1+gv.RANK,3) 
    for i_pop in range(gv.n_pop):
        plt.plot(tau_list, mean_overlap_1[i_pop], '-o', color = gv.pal[i_pop]) 
        plt.fill_between(tau_list, mean_overlap_1[i_pop]-std_overlap_1[i_pop],
                         mean_overlap_1[i_pop]+std_overlap_1[i_pop], alpha=.1, color=gv.pal[i_pop]) 
        
    plt.plot(tau_list, overlap_1_list[0], 'x', color = gv.pal[0])
    plt.xlabel('$\\tau_{fac}$ (ms)') 
    plt.ylabel('Sample Overlap (Hz)') 
    
plt.show() 
