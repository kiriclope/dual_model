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

m0_list = []
m1_list = []

gv.init_param() 

path = '/homecentral/alexandre.mahrach/IDIBAPS/cpp/model/simulations/%dpop/%s/N%d/K%d/' % (gv.n_pop, gv.folder, gv.n_neurons, gv.K) 
path += 'spec/kappa_%.2f/' % gv.KAPPA

def parloop(path, trial) :
    
    path += 'trial_%d/' % trial ; 
    
    try : 
        filter_rates = pd.read_csv(path + 'filter_rates.dat', sep='\s+').to_numpy() 
        rates = np.delete(filter_rates, [0], axis=1)
        
        n_neurons = int(rates.shape[1]/2) 
        rates = np.reshape(rates, (rates.shape[0], 2, n_neurons)) 
        
        avg_rates = np.mean(rates, axis=0) # avg -> over time 
        
        filter_rates = savgol_filter(avg_rates, int(250 * 2 + 1),
                                     polyorder=0, deriv=0, axis=-1,
                                     mode='mirror')
        
        fft_rates = np.fft.rfft(filter_rates, axis=-1) / gv.n_size 
        
        m0 = np.mean(avg_rates, axis=-1) 
        m1 = np.absolute(fft_rates[...,1].real) 
    except : 
        m0 = np.zeros(2) * np.nan 
        m1 = np.zeros(2) * np.nan 
        
    return m0, m1 

trial_list = np.arange(0, 150, 1) 

with pgb.tqdm_joblib( pgb.tqdm(desc='computing m0 and m1', total=trial_list.shape[0] ) ) as progress_bar: 
    
    m0_list, m1_list = zip( *Parallel(n_jobs=-1)(delayed(parloop)(path, trial) 
                                                 for trial in trial_list ) ) 

m0_list = np.array(m0_list).T
m1_list = np.array(m1_list).T  

print('m0', m0_list.shape, 'm1', m1_list.shape) 
figtitle = 'm0_m1_kappa_%.2f_trials_' % gv.KAPPA + gv.folder 
fig = plt.figure(figtitle, figsize=(1.25*1.618*1.5*2, 1.618*1.25)) 
ax = fig.add_subplot(1,2,1) 

for i_pop in range(gv.n_pop): 
    plt.hist(m0_list[i_pop], histtype='step', color=gv.pal[i_pop]) 

plt.ylabel('Count') 
plt.xlabel('$m_0$ (Hz)') 

ax = fig.add_subplot(1,2,2) 
for i_pop in range(gv.n_pop): 
    plt.hist(m1_list[i_pop], histtype='step', color=gv.pal[i_pop])
    
plt.ylabel('Count') 
plt.xlabel('$m_1$ (Hz)') 

plt.show() 
