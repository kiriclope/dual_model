import sys, os, importlib
from importlib import reload 
from scipy.signal import savgol_filter 

import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd 
from joblib import Parallel, delayed

import params as gv 
importlib.reload(sys.modules['params']) 
import progressbar as pgb 

from write import replace_global
from get_m1 import *
from utils import *

J0_list = np.arange(0.005, .041, .001) 
Jee_list = np.arange(0.0, 4.1, .1) 

gv.IF_STP = 1
gv.IF_SPEC = 1
gv.IF_INI_COND = 0 
gv.init_param() 

def parloop(J0, Jee, path=gv.path, folder=gv.folder, verbose=0) : 

    new_folder = folder + '_J0_%.3f_Jee_%.2f' % (J0, Jee) 
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
                
        if verbose : 
            print('m1', m1) 
            
    except : 
        m0 = np.zeros(2) * np.nan 
        m1 = np.zeros(2) * np.nan 
    
    return m0, m1 

parloop(0.005, 1.0, verbose=1) 

with pgb.tqdm_joblib( pgb.tqdm(desc='computing m0 and m1', total=int(Jee_list.shape[0] * J0_list.shape[0]) ) ) as progress_bar: 
    
    m0_list, m1_list = zip( *Parallel(n_jobs=-2)(delayed(parloop)(J0, Jee, verbose=0) 
                                                 for Jee in Jee_list 
                                                 for J0 in J0_list ) ) 

m0_list = np.array(m0_list).T 
m1_list = np.array(m1_list).T 

m0_list = m0_list.reshape(gv.n_pop, Jee_list.shape[0], J0_list.shape[0])
m1_list = m1_list.reshape(gv.n_pop, Jee_list.shape[0], J0_list.shape[0]) 

extent=[J0_list[0]*1000 , J0_list[-1]*1000, Jee_list[0] , Jee_list[-1]] 

figtitle = 'm0_m1_J0_Jee_' + gv.folder 

fig = plt.figure(figtitle, figsize=(1.25*1.618*1.5*2, 1.618*1.25*2)) 
ax = fig.add_subplot(1,2,1) 
    
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
im=ax.imshow(m0_list[0], cmap='jet', vmin=0, vmax=20, origin='lower', extent=extent, aspect='auto') 
fig.colorbar(im, cax=cax, orientation='vertical') 
ax.grid(False)
ax.set_ylabel('$J_{EE}$ (au)') 
ax.set_xlabel('$\\nu_{ext}$ (Hz)') 

ax = fig.add_subplot(1,2,2) 
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
im = ax.imshow(m1_list[0], cmap='jet', vmin=0, vmax=10, origin='lower', extent=extent, aspect='auto')
fig.colorbar(im, cax=cax, orientation='vertical')

ax.grid(False) 
ax.set_ylabel('$J_{EE}$ (au)') 
ax.set_xlabel('$\\nu_{ext}$ (Hz)') 

plt.show() 
