import sys, importlib
from importlib import reload 
from scipy.signal import savgol_filter 

import numpy as np 
import matplotlib.pyplot as plt

from joblib import Parallel, delayed 

import params as gv 
importlib.reload(sys.modules['params']) 
import progressbar as pgb 

from write import replace_global 
from utils import * 
from get_m1 import * 

K_list = np.arange(1000, 5000, 1000) 
# K_list = np.array([500, 1000, 2000, 2500, 3000,3500, 4000]) 

gv.init_param() 

def parloop(K, path=gv.path, verbose=0, RANK=gv.RANK, MAP=gv.MAP, con_path=gv.con_path, old_K=gv.K): 
    
    new_K = 'K%d' % K 
    new_path = path.replace('K%d' % old_K, new_K) 
    
    # if(RANK==1): 
    #     new_path += '/spec/kappa_%.2f' % kappa ; 
    # if(RANK==2): 
    #     new_path += '/spec/kappa_%.2f_kappa_1_%.2f' % kappa ; 
    
    if verbose:
        print(new_path)
        
    # try :
        
    time, rates = get_time_rates(MAP, new_path, con_path) 
    if(verbose):
        print('time', time.shape, 'rates', rates.shape)
        
    avg_rates = np.nanmean(rates, axis=0) # avg over time         
    m0 = np.nanmean(avg_rates, axis=-1) # mean over neurons 
        
    if verbose: 
        print('m0', m0, '<m0>', avg_rates.shape) 
        
    m1, phi, smooth_rates = get_avg_m1_phi_smooth_rates(avg_rates) 
        
    if verbose:
        print('m1', m1) 
            
    # except :
    #     m0 = np.zeros(2) * np.nan 
    #     m1 = np.zeros(2) * np.nan 
    
    return m0, m1 
            
# parloop(path, 500, verbose=1)

with pgb.tqdm_joblib( pgb.tqdm(desc='computing m0 and m1', total= K_list.shape[0] ) ) as progress_bar: 
    
    m0_list, m1_list = zip( *Parallel(n_jobs=-1)(delayed(parloop)(K, verbose=0) for K in K_list ) ) 

m0_list = np.array(m0_list).T 
m1_list = np.array(m1_list).T 

sqrt_K_list = np.sqrt(K_list)

m0_list = m0_list.reshape(gv.n_pop, K_list.shape[0]) * sqrt_K_list
m1_list = m1_list.reshape(gv.n_pop, K_list.shape[0]) * sqrt_K_list

print(m0_list)

figname = 'm0_m1_K_' + gv.folder 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(1.25*1.618*1.5*2, 1.618*1.5), num=figname) 

ax1.set(xlim=(sqrt_K_list[0], sqrt_K_list[-1]), ylim=(0, np.nanmax(m0_list)  * 1.2 ) ) 
ax1.set_xlabel('$K$') 
ax1.set_ylabel('$\\nu^0$ (Hz)') 


m0_fit=np.zeros((2,2))
for i_pop in range(gv.n_pop): 
    m0_fit[i_pop] = np.polyfit(sqrt_K_list, m0_list[i_pop], 1) 
    ax1.plot(sqrt_K_list, m0_list[i_pop,:], '-o', color = gv.pal[i_pop]) 

print(m0_fit[:,0])

ax2.set(xlim=(sqrt_K_list[0], sqrt_K_list[-1]), ylim=(0, np.nanmax(m1_list) * 1.2  ) ) 
ax2.set_xlabel('$K$') 
ax2.set_ylabel('$\\nu^1$ (Hz)') 

m1_fit=np.zeros((2,2))
for i_pop in range(gv.n_pop):
    m1_fit[i_pop] = np.polyfit(sqrt_K_list, m1_list[i_pop], 1) 
    ax2.plot(sqrt_K_list, m1_list[i_pop], '-o', color = gv.pal[i_pop], label=gv.label[i_pop]) 

print(m1_fit[:,0]) 

ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5)) 

plt.show() 
