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
gv.IF_STP=0
gv.IF_INI_COND=0
gv.IF_TRIALS=0

gv.RANK=1 
gv.MAP=0
gv.init_param() 

tau_list = np.arange(200, 2100, 100) 
# tau_list = np.arange(0.05, .55, .05) 
ini_list = np.arange(0, 2, 1) 

def parloop(tau_f, ini, path=gv.path, RANK=gv.RANK, MAP=gv.MAP, verbose=0, con_path=gv.con_path, kappa=gv.KAPPA, tau_r=gv.TAU_REC, use=gv.USE, MAP_SEED=gv.MAP_SEED) : 

    path += '/STP/Tf_%d_Tr_%d_U_%.2f' % (tau_f, tau_r, use) 
    
    # if(RANK==1):
    #     path += '/spec/kappa_%.2f' % kappa ; 
    #     con_path += '/spec/kappa_%.2f' % kappa ; 
    # if(RANK==2):
    #     path += '/spec/kappa_%.2f_kappa_1_%.2f' %  (kappa, kappa) ; 
    #     con_path += '/spec/kappa_%.2f_kappa_1_%.2f' %  (kappa, kappa) ;
        
    #     path += '/seed_%d' % (MAP_SEED) 
    #     con_path += '/seed_%d' % (MAP_SEED) 
    
    path += '/ini_cond_%d' % ini ; 
    
    if verbose: 
        print(path) 
    
    # try :
    if 0==0:
        time, rates = get_time_rates(MAP, path, con_path) 
        if(verbose):
            print('time', time.shape, 'rates', rates.shape)
        
        avg_rates = np.nanmean(rates, axis=0) # avg over time         
        m0 = np.nanmean(avg_rates, axis=-1) # mean over neurons 
        
        if verbose: 
            print('m0', m0, '<m0>', avg_rates.shape) 
        
        m1, phi, smooth_rates = get_m1_phi_smooth_rates(rates, osc=0)
        
        m1 = np.nanmean(m1, axis=-1) 
        phi = np.nanmean(phi, axis=-1)
        
        # m1, phi, smooth_rates = get_avg_m1_phi_smooth_rates(avg_rates) 
        
        if verbose:
            print('m1', m1) 
            
    # except :
    #     m0 = np.zeros(2) * np.nan 
    #     m1 = np.zeros(2) * np.nan 
    
    return m0, m1 

parloop(200, 0, verbose=1)

with pgb.tqdm_joblib( pgb.tqdm(desc='computing m0 and m1', total=int(tau_list.shape[0] * ini_list.shape[0]) ) ) as progress_bar: 
    
    m0_list, m1_list = zip( *Parallel(n_jobs=-2, backend='multiprocessing')(delayed(parloop)(tau_f, ini, verbose=0) 
                                                                            for tau_f in tau_list for ini in ini_list ) )  

m0_list = np.array(m0_list).T 
m1_list = np.array(m1_list).T 

m0_list = m0_list.reshape(gv.n_pop, tau_list.shape[0], ini_list.shape[0]) 
m1_list = m1_list.reshape(gv.n_pop, tau_list.shape[0], ini_list.shape[0]) 

mean_m0 = np.nanmean(m0_list,axis=-1) 
mean_m1 = np.nanmean(m1_list,axis=-1) 

std_m0 = np.nanstd(m0_list,axis=-1) 
std_m1 = np.nanstd(m1_list,axis=-1) 

print('<m0>', mean_m0) 
print('<m1>', mean_m1) 

figtitle = 'm0_m1_tau_f_' + gv.folder 
fig = plt.figure(figtitle, figsize=(1.25*1.618*1.5*2, 1.618*1.25)) 
ax = fig.add_subplot(1,2,1) 
for i_pop in range(gv.n_pop):
    plt.plot(tau_list, mean_m0[i_pop], '-o', color = gv.pal[i_pop]) 
    plt.fill_between(tau_list, mean_m0[i_pop]-std_m0[i_pop], mean_m0[i_pop]+std_m0[i_pop], alpha=.1, color=gv.pal[i_pop])
    
plt.xlabel('$\\tau_{fac}$') 
plt.ylabel('$\\nu^{(0)}$ (Hz)') 
# plt.ylim([0, np.ceil(np.nanmax(mean_m0)*12 )/10]) 

ax = fig.add_subplot(1,2,2) 
for i_pop in range(gv.n_pop):
    plt.plot(tau_list, mean_m1[i_pop], '-o', color = gv.pal[i_pop]) 
    plt.fill_between(tau_list, mean_m1[i_pop]-std_m1[i_pop], mean_m1[i_pop]+std_m1[i_pop], alpha=.1, color=gv.pal[i_pop]) 
    
plt.plot(tau_list, m1_list[0,:], 'x', color = gv.pal[0])

# plt.ylim([0, np.ceil(np.nanmax(mean_m1)*12 )/10]) 

plt.xlabel('$\\tau_{fac}$') 
plt.ylabel('$\\nu^{(1)}$ (Hz)') 
# plt.ylim([0, 2]) 

plt.show() 
