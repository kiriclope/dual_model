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

kappa_list = [] 
m0_list = [] 
m1_list = [] 

gv.init_param() 

path = '/homecentral/alexandre.mahrach/IDIBAPS/cpp/model/simulations/%s/%dpop/%s/N%d/K%d/' % (gv.model, gv.n_pop, gv.folder, gv.n_neurons, gv.K) 

if(gv.IF_STP): 
    path += 'STP/Tf_%d_Tr_%d_U_%.2f/' % (gv.TAU_FAC, gv.TAU_REC, gv.USE) 

def parloop(path, kappa, trial) :
    
    path += 'spec/kappa_%.2f/' % kappa
    path += 'trial_%d/' % trial ; 
    
    try : 
        filter_rates = pd.read_csv(path + 'filter_rates.dat', sep='\s+').to_numpy() 
        rates = np.delete(filter_rates, [0], axis=1) 

        n_neurons = int(rates.shape[1]/2) 
        rates = np.reshape(rates, (rates.shape[0], 2, n_neurons)) 
        
        avg_rates = np.mean(rates, axis=0) # avg -> over time 
        
        # filter_rates = savgol_filter(avg_rates,  int(avg_rates.shape[-1]/10 * 2 + 1),
        #                              polyorder=0, deriv=0, axis=-1,
        #                              mode='wrap') 
        
        # fft_rates = np.fft.rfft(filter_rates, axis=-1) / gv.n_size 
        # m1 = 2*np.absolute(fft_rates[...,1].real) 
        
        m0 = np.mean(avg_rates, axis=-1) 
        
        smooth_avg_rates = circular_convolution(avg_rates, int(avg_rates.shape[-1]*.01) ) 
        m1 = compute_m1(smooth_avg_rates) 
        
    except :
        m0 = np.zeros(2) * np.nan 
        m1 = np.zeros(2) * np.nan 
    
    return m0, m1 

kappa_list = np.arange(0, 11, 1) 
trial_list = np.arange(0, 11, 1) 

with pgb.tqdm_joblib( pgb.tqdm(desc='computing m0 and m1', total=int(kappa_list.shape[0] * trial_list.shape[0]) ) ) as progress_bar: 
    
    m0_list, m1_list = zip( *Parallel(n_jobs=-25)(delayed(parloop)(path, kappa, trial) 
                                                 for kappa in kappa_list 
                                                 for trial in trial_list ) ) 

m0_list = np.array(m0_list).T 
m1_list = np.array(m1_list).T 

m0_list = m0_list.reshape(gv.n_pop, kappa_list.shape[0], trial_list.shape[0]) 
m1_list = m1_list.reshape(gv.n_pop, kappa_list.shape[0], trial_list.shape[0]) 

mean_m0 = np.nanmean(m0_list,axis=-1) 
mean_m1 = np.nanmean(m1_list,axis=-1) 

std_m0 = np.nanstd(m0_list,axis=-1) 
std_m1 = np.nanstd(m1_list,axis=-1) 

print('m0', mean_m0) 
print('m1', mean_m1) 

print(m1_list.shape)

figtitle = 'm0_m1_kappa_' + gv.folder 
fig = plt.figure(figtitle, figsize=(1.25*1.618*1.5*2, 1.618*1.25)) 
ax = fig.add_subplot(1,2,1) 
for i_pop in range(gv.n_pop):
    plt.plot(kappa_list, mean_m0[i_pop], '-o', color = gv.pal[i_pop]) 
    plt.fill_between(kappa_list, mean_m0[i_pop]-std_m0[i_pop], mean_m0[i_pop]+std_m0[i_pop], alpha=.1, color=gv.pal[i_pop])
    
plt.xlabel('$\kappa$') 
plt.ylabel('$\\nu^0$ (Hz)') 
# plt.ylim([0, 100]) 

ax = fig.add_subplot(1,2,2) 
for i_pop in range(gv.n_pop):
    plt.plot(kappa_list, mean_m1[i_pop], '-o', color = gv.pal[i_pop], label=gv.label[i_pop]) 
    plt.fill_between(kappa_list, mean_m1[i_pop]-std_m1[i_pop], mean_m1[i_pop]+std_m1[i_pop], alpha=.1, color=gv.pal[i_pop]) 

plt.plot(kappa_list, m1_list[0,:], 'x', color = gv.pal[0]) 
    
plt.xlabel('$\kappa$') 
plt.ylabel('$\\nu^1$ (Hz)') 
# plt.ylim([0, 2]) 
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) 

plt.show() 
