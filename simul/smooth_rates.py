import sys, os, importlib
from importlib import reload
from scipy.signal import savgol_filter 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import params as gv 
importlib.reload(sys.modules['params']) 

from get_m1 import *
from utils import *

gv.folder = 'christos_off'
gv.IF_INI_COND = 0
gv.IF_TRIALS = 0
gv.init_param()
path = gv.path

def get_smooth_rates(path):
    smooth_trial = []
    for i_trial in range(1, 2+1): 
        smooth_ini = []
        for i_ini in range(1, 4 +1):
            gv.path = path
            gv.path += '/trial_%d' % i_trial ; 
            gv.path += '/ini_cond_%d' % i_ini ; 
            print(gv.path) 

            try:
                time, rates = get_time_rates(path=gv.path) 
                # pop_rates = np.mean(rates[8:12, 0, : gv.n_size[0]], axis=0) 
                pop_rates = np.mean(rates[12:16, 0, : gv.n_size[0]], axis=0) 
                
                smooth_rates = circular_convolution(pop_rates, int(pop_rates.shape[-1]*.1) ) 
                phi = compute_phi(smooth_rates)
                
                print('phi', phi.shape, 'smooth_rates', smooth_rates.shape) 
                
                smooth_ini.append(np.roll(smooth_rates, int( (phi/np.pi-0.5)*gv.n_size[0]) ) ) 
                print('phi', phi * 180 / np.pi, 'phi_ext', (1-i_trial / gv.N_TRIALS) * 180)
            except:
                smooth_ini.append(np.nan*np.zeros(32000)) 
                print('error')
                pass
        smooth_trial.append(smooth_ini) 
    
    smooth_trial = np.asarray(smooth_trial) 
    
    return smooth_trial 

    

smooth = get_smooth_rates(path)
print('smooth', smooth.shape)

smooth_avg = np.nanmean(np.vstack(smooth), axis=0) 
smooth_std = np.nanstd(np.vstack(smooth), axis=0, ddof=1) / np.sqrt(np.vstack(smooth).shape[0])

theta = np.linspace(-180, 180, gv.n_size[0]) 

figname = 'off_on_' + 'smooth_stim'

plt.figure(figname)
plt.plot(theta, smooth_avg, 'b')
plt.fill_between(theta, smooth_avg - smooth_std, smooth_avg + smooth_std, 'b', alpha=.5)
plt.xlabel('Prefered location (°)')
plt.ylabel('Rates (Hz)')
plt.xticks([-180, -90, 0, 90, 180])

path  = path.replace('christos_off', 'on_2')

smooth_on = get_smooth_rates(path)
print('smooth_on', smooth_on.shape)

smooth_on_avg = np.nanmean(np.vstack(smooth_on), axis=0) 
smooth_on_std = np.nanstd(np.vstack(smooth_on), axis=0, ddof=1) / np.sqrt(np.vstack(smooth).shape[0])

plt.plot(theta, smooth_on_avg, 'r')
plt.fill_between(theta, smooth_on_avg - smooth_on_std, smooth_on_avg + smooth_on_std, 'r', alpha=.5)

plt.savefig(figname + '.svg', dpi=300)
