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

def parloop(path, tau_f, tau_r, use, kappa, trial) : 
    
    path += 'STP/Tf_%d_Tr_%d_U_%.2f/' % (tau_f, tau_r, use) 
    path += 'spec/kappa_%.2f/' % kappa 
    path += 'trial_%d/' % trial ;
    
    try : 
        filter_rates = pd.read_csv(path + 'filter_rates.dat', sep='\s+').to_numpy() 
        rates = np.delete(filter_rates, [0], axis=1) 

        n_neurons = int(rates.shape[1]/2) 
        rates = np.reshape(rates, (rates.shape[0], 2, n_neurons)) 
        
        avg_rates = np.mean(rates, axis=0) # avg -> over time 
                
        m0 = np.mean(avg_rates, axis=-1) 
        
        smooth_avg_rates = circular_convolution(avg_rates, int(avg_rates.shape[-1] * .01 ) ) 
        m1 = compute_m1(smooth_avg_rates) 
        
    except : 
        m0 = np.zeros(2) * np.nan 
        m1 = np.zeros(2) * np.nan 
    
    return m0, m1 

tau_list = np.arange(250, 1250, 250) 
trial_list = np.arange(0, 6, 1) 

with pgb.tqdm_joblib( pgb.tqdm(desc='computing m0 and m1', total=int(tau_list.shape[0] * trial_list.shape[0]) ) ) as progress_bar: 
    
    m0_list, m1_list = zip( *Parallel(n_jobs=-25)(delayed(parloop)(path, tau_f, gv.TAU_REC, gv.USE, gv.KAPPA, trial) 
                                                 for tau_f in tau_list 
                                                 for trial in trial_list ) ) 

m0_list = np.array(m0_list).T 
m1_list = np.array(m1_list).T 

m0_list = m0_list.reshape(gv.n_pop, tau_list.shape[0], trial_list.shape[0]) 
m1_list = m1_list.reshape(gv.n_pop, tau_list.shape[0], trial_list.shape[0]) 

mean_m0 = np.mean(m0_list,axis=-1) 
mean_m1 = np.mean(m1_list,axis=-1) 

std_m0 = np.std(m0_list,axis=-1) 
std_m1 = np.std(m1_list,axis=-1) 

print('m0', mean_m0) 
print('m1', mean_m1) 

figtitle = 'm0_m1_tau_f_' + gv.folder 
fig = plt.figure(figtitle, figsize=(1.25*1.618*1.5*2, 1.618*1.25)) 
ax = fig.add_subplot(1,2,1) 
for i_pop in range(gv.n_pop):
    plt.plot(tau_list, mean_m0[i_pop], '-o', color = gv.pal[i_pop]) 
    plt.fill_between(tau_list, mean_m0[i_pop]-std_m0[i_pop], mean_m0[i_pop]+std_m0[i_pop], alpha=.1, color=gv.pal[i_pop])
    
plt.xlabel('$\\tau_{fac}$') 
plt.ylabel('$m_0$ (Hz)') 
# plt.ylim([0, 100]) 

ax = fig.add_subplot(1,2,2) 
for i_pop in range(gv.n_pop):
    plt.plot(tau_list, mean_m1[i_pop], '-o', color = gv.pal[i_pop]) 
    plt.fill_between(tau_list, mean_m1[i_pop]-std_m1[i_pop], mean_m1[i_pop]+std_m1[i_pop], alpha=.1, color=gv.pal[i_pop]) 
    
plt.xlabel('$\\tau_{fac}$') 
plt.ylabel('$m_1$ (Hz)') 
# plt.ylim([0, 2]) 

plt.show() 
