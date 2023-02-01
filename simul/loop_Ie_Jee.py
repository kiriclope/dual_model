import sys, os, importlib
from importlib import reload 
from scipy.signal import savgol_filter 
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

Ie_list = np.arange(700, 810, 10) 
Jee_list = np.arange(900, 1010, 10) 

gv.IF_STP = 1
gv.IF_SPEC = 0

gv.init_param() 

def heatmap(x, clim=None):
    if clim is not None:
        im=ax.imshow(x, cmap='viridis', vmin=clim[0], vmax=clim[1], origin='lower',
                     extent=[Ie_list[0] , Ie_list[-1], Jee_list[0] , Jee_list[-1]]) 
    else:
        im=ax.imshow(x, cmap='viridis', vmin=0, origin='lower',
                     extent=[Ie_list[0] , Ie_list[-1], Jee_list[0] , Jee_list[-1]])

    plt.ylabel('$J_{EE}$ (au)')
    plt.xlabel('$I_E$ (au)') 
    plt.grid(False)
        
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.grid(False)

def parloop(Ie, Jee, path=gv.path, folder=gv.folder, bins=None, verbose=0) : 

    new_folder = folder + '_Ie_%.2f_Jee_%.2f' % (Ie, Jee) 
    new_path = path.replace(folder, new_folder) 
    
    if verbose: 
        print(new_path) 

    # if 0==0:
    try :
        
        time, rates = get_time_rates(MAP=0, path=new_path) 
        if(verbose):
            print('time', time.shape, 'rates', rates.shape)
        if bins is not None:
            if bins>0:
                avg_rates = np.nanmean(rates[:bins], axis=0) # avg over time
            else:
                avg_rates = np.nanmean(rates[bins:], axis=0) # avg over time
        else:
            avg_rates = np.nanmean(rates, axis=0) # avg over time
        
        m0 = np.nanmean(avg_rates, axis=-1) # mean over neurons 
        
        if verbose: 
            print('m0', m0, '<m0>', avg_rates.shape) 
        
        m1, phi, smooth_rates = get_avg_m1_phi_smooth_rates(avg_rates) 
        
        # m1, phi, smooth_rates, m1_time = get_m1_phi_smooth_rates(rates) 
        # m1 = np.mean(m1, axis=-1) 
        m1_time = m1
        
        if verbose : 
            print('m1', m1) 
            
    except : 
        m0 = np.zeros(2) * np.nan 
        m1 = np.zeros(2) * np.nan 
        m1_time = np.zeros(2) * np.nan 
    
    return m0, m1 , m1_time

parloop(2, 1.0, verbose=1) 

bins=4

with pgb.tqdm_joblib( pgb.tqdm(desc='computing m0 and m1', total=int(Jee_list.shape[0] * Ie_list.shape[0]) ) ) as progress_bar: 
    
    m0_list, m1_list, m1_time_list = zip( *Parallel(n_jobs=-2)(delayed(parloop)(Ie, Jee, bins=bins) 
                                                               for Jee in Jee_list 
                                                               for Ie in Ie_list ) ) 

m0_list = np.array(m0_list).T 
m1_list = np.array(m1_list).T 
m1_time_list = np.array(m1_time_list).T 

m0_list = m0_list.reshape(gv.n_pop, Jee_list.shape[0], Ie_list.shape[0])
m1_list = m1_list.reshape(gv.n_pop, Jee_list.shape[0], Ie_list.shape[0]) 
m1_time_list = m1_time_list.reshape(gv.n_pop, Jee_list.shape[0], Ie_list.shape[0]) 

if bins==4:
    figtitle = 'm0_m1_Ie_Jee_' + gv.folder + '_down'
else:
    figtitle = 'm0_m1_Ie_Jee_' + gv.folder + '_up'

fig = plt.figure(figtitle, figsize=(1.618*2*3, 1.618*2))

ax = fig.add_subplot(1,3,1)
ax.set_title('$m_0$')
heatmap(m0_list[0])

ax = fig.add_subplot(1,3,2) 
ax.set_title('$m_1$')
heatmap(m1_list[0])

ax = fig.add_subplot(1,3,3)
ax.set_title('$m_1 / m_0$')
heatmap(m1_list[0]/m0_list[0], clim=[0,1])
        
plt.show() 
