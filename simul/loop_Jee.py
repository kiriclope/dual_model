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
from rank_utils import *
from plot_settings import *
SetPlotParams()

gv.IF_INI_COND=0
gv.IF_TRIALS=0

gv.RANK=1
gv.MAP=0

gv.IF_STP=1
gv.init_param() 

NORM_M1=0 

if(gv.IF_STP): 
    Jee_list = np.arange(1.0, 11, 1) 
    # Jee_list = np.arange(1., 2.1, .1) 
    Jee_list = np.arange(0.5, 5.5, .5) 
else: 
    Jee_list = np.arange(0.5, 5.5, .5) 
    Jee_list = np.arange(0., 1.1, .1) 
    
ini_list = np.arange(0, 5+1, 1) 

def parloop(Jee, ini, path=gv.path, folder=gv.folder, con_path=gv.con_path, MAP=gv.MAP, verbose=0) : 
    
    new_folder = folder + '_Jee_%.3f' % Jee 
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
        
        m1, phi, _ = get_m1_phi_smooth_rates(rates.copy()) 
        m1 = np.nanmean(m1, axis=-1) 
        phi = np.nanmean(phi, axis=-1) 
        
        if verbose:
            print('m1', m1, 'phi', phi) 
                            
    except :
        m0 = np.zeros(2) * np.nan 
        m1 = np.zeros(2) * np.nan 
        phi = np.zeros(2) * np.nan 
    
    return m0, m1, phi

parloop(1,0, verbose=1) 

with pgb.tqdm_joblib( pgb.tqdm(desc='computing m0 and m1', total=int(Jee_list.shape[0] * ini_list.shape[0]) ) ) as progress_bar: 
    
    m0_list, m1_list, phi_list = zip( *Parallel(n_jobs=-1, backend="multiprocessing")(delayed(parloop)( Jee, ini, gv.path, gv.folder, MAP=0) 
                                                                                      for Jee in Jee_list 
                                                                                      for ini in ini_list ) ) 

if gv.RANK==2:
    with pgb.tqdm_joblib( pgb.tqdm(desc='computing m0 and m1', total=int(Jee_list.shape[0] * ini_list.shape[0]) ) ) as progress_bar: 
        
        _, m11_list, phi1_list = zip( *Parallel(n_jobs=-1, backend="multiprocessing")(delayed(parloop)( Jee, ini, gv.path, gv.folder, MAP=1) 
                                                                                      for Jee in Jee_list 
                                                                                      for ini in ini_list ) ) 
    
m0_list = np.array(m0_list).T 
m1_list = np.array(m1_list).T 
phi_list = np.array(phi_list).T

m0_list = m0_list.reshape(gv.n_pop, Jee_list.shape[0], ini_list.shape[0]) 
m1_list = m1_list.reshape(gv.n_pop, Jee_list.shape[0], ini_list.shape[0]) 
phi_list = phi_list.reshape(gv.n_pop, Jee_list.shape[0], ini_list.shape[0]) 
    
mean_m0 = np.nanmean(m0_list,axis=-1) 
mean_m1 = np.nanmean(m1_list,axis=-1) 
mean_phi = np.nanmean(phi_list,axis=-1) 

std_m0 = np.nanstd(m0_list,axis=-1) 
std_m1 = np.nanstd(m1_list,axis=-1) 
std_phi = np.nanstd(phi_list,axis=-1) 

print('m0', mean_m0) 
print('m1', mean_m1) 

if(gv.RANK==2):
    m11_list = np.array(m11_list).T 
    phi1_list = np.array(phi1_list).T 

    m11_list = m11_list.reshape(gv.n_pop, Jee_list.shape[0], ini_list.shape[0]) 
    phi1_list = phi1_list.reshape(gv.n_pop, Jee_list.shape[0], ini_list.shape[0]) 

    mean_m11 = np.nanmean(m11_list,axis=-1) 
    mean_phi1 = np.nanmean(phi1_list,axis=-1) 

    std_m11 = np.nanstd(m11_list,axis=-1) 
    std_phi1 = np.nanstd(phi1_list,axis=-1) 
    
figname = 'm0_m1_phi_Jee_' + gv.folder

if gv.RANK==1:
    fig, axis = plt.subplots(1, 3, figsize=(1.25*1.618*1.5*3, 1.618*1.5), num=figname) 
if gv.RANK==2:
    fig, (axis, axis1) = plt.subplots(2, 3, figsize=(1.25*1.618*1.5*3, 1.618*1.5*2), num=figname) 
    
axis[0].set(ylim=(0, np.ceil( np.nanmax(m0_list)*12 ) /10 ) ) 
axis[0].set_xlabel('$J_{EE}$ (a.u.)') 
axis[0].set_ylabel('Population Rates (Hz)') 
        
axis[1].set(ylim=(0, np.ceil( np.nanmax(m1_list)*12 ) /10 ) )     
axis[1].set_xlabel('$J_{EE}$ (a.u.)') 
axis[1].set_ylabel('Amplitude (Hz)') 

axis[2].set(ylim=(0, np.pi ) )
axis[2].set_xlabel('$J_{EE}$ (a.u.)')
axis[2].set_ylabel('Phase (rad)') 
axis[2].set_yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]) 
axis[2].set_yticklabels(['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$']) 

if(gv.RANK==2):
    axis1[1].set(ylim=(0, np.ceil( np.nanmax(m11_list)*12 ) /10 ) )     
    axis1[1].set_xlabel('$J_{EE}$ (a.u.)') 
    axis1[1].set_ylabel('Amplitude (Hz)') 

    axis1[2].set(ylim=(0, np.pi ) )
    axis1[2].set_xlabel('$J_{EE}$ (a.u.)')
    axis1[2].set_ylabel('Phase (rad)') 
    axis1[2].set_yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]) 
    axis1[2].set_yticklabels(['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$']) 
    
for i_pop in range(gv.n_pop):
    axis[0].plot(Jee_list, mean_m0[i_pop], '-o', color = gv.pal[i_pop]) 
    axis[0].plot(Jee_list, m0_list[i_pop,:], 'x', color = gv.pal[i_pop], alpha=0.2) 
    # axis[0].fill_between(Jee_list, mean_m0[i_pop]-std_m0[i_pop], mean_m0[i_pop]+std_m0[i_pop], alpha=.1, color=gv.pal[i_pop])
    
if(NORM_M1):
    for i_pop in range(gv.n_pop):
        axis[1].plot(Jee_list, mean_m1[i_pop]/mean_m0[i_pop], '-o', color = gv.pal[i_pop], label=gv.label[i_pop]) 
        # axis[1].fill_between(Jee_list, (mean_m1[i_pop]-std_m1[i_pop])/mean_m0[i_pop], (mean_m1[i_pop]+std_m1[i_pop])/mean_m0[i_pop], alpha=.1, color=gv.pal[i_pop]) 

    for i in range(m1_list.shape[1]):
        axis[1].plot(Jee_list[i] * np.ones(m1_list[0,i].shape[0]) , m1_list[0,i]/mean_m0[0,i], 'x', color = gv.pal[0], alpha=1) 
else:
    for i_pop in range(gv.n_pop):
        axis[1].plot(Jee_list, mean_m1[i_pop], '-o', color = gv.pal[i_pop], label=gv.label[i_pop]) 
        axis[1].fill_between(Jee_list, mean_m1[i_pop]-std_m1[i_pop], mean_m1[i_pop]+std_m1[i_pop], alpha=.1, color=gv.pal[i_pop]) 
        # axis[1].plot(Jee_list, m1_list[0,:], 'x', color = gv.pal[0], alpha=1) 
        

# for i_pop in range(gv.n_pop):
#     axis[2].plot(Jee_list, mean_phi[i_pop], '-o', color = gv.pal[i_pop], label=gv.label[i_pop]) 
#     axis[2].fill_between(Jee_list, mean_phi[i_pop]-std_phi[i_pop], mean_phi[i_pop]+std_phi[i_pop], alpha=.1, color=gv.pal[i_pop]) 

axis[2].plot(Jee_list, phi_list[0,:], 'x', color = gv.pal[0], alpha=0.5) 
# axis[2].plot(Jee_list, phi_list[0,:], 'x', color = gv.pal[0], alpha=0.5) 

if(gv.RANK==2):
    for i_pop in range(gv.n_pop):
        axis1[1].plot(Jee_list, mean_m11[i_pop], '-o', color = gv.pal[i_pop], label=gv.label[i_pop]) 
        axis1[1].fill_between(Jee_list, mean_m11[i_pop]-std_m11[i_pop], mean_m11[i_pop]+std_m11[i_pop], alpha=.1, color=gv.pal[i_pop]) 
        # axis1[1].plot(Jee_list, m11_list[0,:], 'x', color = gv.pal[0], alpha=1) 

    axis1[2].plot(Jee_list, phi1_list[0,:], 'x', color = gv.pal[0], alpha=0.5) 

plt.show() 
