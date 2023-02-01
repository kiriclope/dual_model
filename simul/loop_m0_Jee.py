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

Ie_list = np.arange(0.25, 2, .25) 
Jee_list = np.arange(1.0, 4.1, .1) 

gv.IF_STP = 1
gv.IF_SPEC = 1

gv.init_param() 

def parloop(Ie, Jee, path=gv.path, folder=gv.folder, verbose=0) : 

    new_folder = folder + '_Ie_%.2f_Jee_%.2f' % (Ie, Jee) 
    new_path = path.replace(folder, new_folder) 
    
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
        
        # m1, phi, smooth_rates, m1_time = get_m1_phi_smooth_rates(rates) 
        # m1 = np.mean(m1, axis=-1) 
        
        if verbose : 
            print('m1', m1) 
            
    except : 
        m0 = np.zeros(2) * np.nan 
        m1 = np.zeros(2) * np.nan 
    
    return m0, m1 

parloop(0.25, 1.0, verbose=1) 

with pgb.tqdm_joblib( pgb.tqdm(desc='computing m0 and m1', total=int(Jee_list.shape[0] * Ie_list.shape[0]) ) ) as progress_bar: 
    
    m0_list, m1_list, m1_time_list = zip( *Parallel(n_jobs=-2)(delayed(parloop)(Ie, Jee) 
                                                               for Jee in Jee_list 
                                                               for Ie in Ie_list ) ) 

m0_list = np.array(m0_list).T 
m1_list = np.array(m1_list).T 
m1_time_list = np.array(m1_time_list).T 

m0_list = m0_list.reshape(gv.n_pop, Jee_list.shape[0], Ie_list.shape[0])
m1_list = m1_list.reshape(gv.n_pop, Jee_list.shape[0], Ie_list.shape[0]) 
m1_time_list = m1_time_list.reshape(gv.n_pop, Jee_list.shape[0], Ie_list.shape[0]) 

figtitle = 'm0_m1_Ie_Jee_' + gv.folder 

fig = plt.figure(figtitle, figsize=(1.25*1.618*1.5*2, 1.618*1.25)) 
ax = fig.add_subplot(1,3,1) 
    
plt.imshow(m0_list[0], cmap='jet', vmin=0, vmax=50, origin='lower', extent=[Ie_list[0] , Ie_list[-1], Jee_list[0] , Jee_list[-1]]) 
plt.grid(False)
plt.colorbar()
plt.ylabel('$J_{EE}$ (au)') 
plt.xlabel('$I_E$ (au)') 

ax = fig.add_subplot(1,3,2) 
plt.imshow(m1_list[0], cmap='jet', vmin=0, vmax=5, origin='lower', extent=[Ie_list[0] , Ie_list[-1], Jee_list[0] , Jee_list[-1]]) 
plt.grid(False) 
plt.colorbar()
plt.ylabel('$J_{EE}$ (au)') 
plt.xlabel('$I_E$ (au)') 

ax = fig.add_subplot(1,3,3) 
plt.imshow(m1_time_list[0], cmap='jet', vmin=0, vmax=5, origin='lower', extent=[Ie_list[0] , Ie_list[-1], Jee_list[0] , Jee_list[-1]]) 
plt.grid(False) 
plt.colorbar()
plt.ylabel('$J_{EE}$ (au)') 
plt.xlabel('$I_E$ (au)') 


plt.show() 
