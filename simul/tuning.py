import sys, importlib

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
from joblib import Parallel, delayed
import seaborn as sns

import progressbar as pgb 
from get_m1 import * 
from utils import * 
from write import *

importlib.reload(sys.modules['params']) 
importlib.reload(sys.modules['get_m1']) 

gv.IF_INI_COND = 0 
gv.IF_TRIALS = 0 
gv.N_TRIALS = 10 
gv.N_INI = 25
gv.IF_CHRISTOS = 0 
gv.FIX_CON = 0
gv.CON_SEED = 3

gv.folder = 'christos_off'
gv.init_param()

path = gv.path

DELAY = 1

def parloop(path, i_trial, i_ini, N_TRIALS=gv.N_TRIALS, A_CUE=gv.A_CUE, EPS_CUE=gv.EPS_CUE, A_DIST=gv.A_DIST, EPS_DIST=gv.EPS_DIST, DELAY=DELAY):

    phi_cue = i_trial / N_TRIALS 
    
    path += '/christos'
    path += '/cue_A_%.2f_eps_%.2f_phi_%.3f' % (A_CUE, EPS_CUE, phi_cue)
    path += '/dist_A_%.2f_eps_%.2f_phi_%.3f' % (A_DIST, EPS_DIST, 0.25+phi_cue)
    
    path += '/trial_%d' % i_trial ; 
    path += '/ini_cond_%d' % i_ini ; 

    # print(path)
    
    try:
        _, rates = get_time_rates(path=path)

        if DELAY:
            delay_rates = np.nanmean(rates[int(5.05/0.05):int(6.05/0.05)], axis=0) # over time
        else:
            delay_rates = np.nanmean(rates[int(2.05/0.05):int(3.05/0.05)], axis=0) # over time
        
        pop_rates = delay_rates[0] 
        pop_rates = pop_rates[~np.isnan(pop_rates)] 
        
        smooth_rates = circular_convolution(pop_rates, int(pop_rates.shape[0]*.01) ) 
        m1, phi = decode_bump(smooth_rates) 
        
        smooth_rates = np.roll(smooth_rates, int((phi/np.pi - 0.5 ) * 32000)) 
        
    except:
        smooth_rates = np.nan*np.zeros(32000)
        print('error:', path, 'not found')
        pass

    return smooth_rates

with pgb.tqdm_joblib( pgb.tqdm(desc='phi off', total= gv.N_INI*gv.N_TRIALS) ) as progress_bar: 
    
    smooth_off = Parallel(n_jobs=-64)(delayed(parloop)(path, i_trial, i_ini, A_CUE=.25, EPS_CUE=.25) 
                                      for i_ini in range(1, gv.N_INI+1) 
                                      for i_trial in range(1, gv.N_TRIALS+1) )
        

smooth_off = np.asarray(smooth_off)    
print(smooth_off.shape)
smooth_off = smooth_off.reshape(gv.N_INI, gv.N_TRIALS, gv.n_size[0]) 

print(smooth_off.shape) 

path  = path.replace(gv.folder, 'christos_on')

with pgb.tqdm_joblib( pgb.tqdm(desc='phi on', total= gv.N_INI *gv.N_TRIALS) ) as progress_bar: 
    
    smooth_on = Parallel(n_jobs=-64)(delayed(parloop)(path, i_trial, i_ini, A_CUE=.25, EPS_CUE=.25) 
                                     for i_ini in range(1, gv.N_INI+1) 
                                     for i_trial in range(1, gv.N_TRIALS+1) )
        
smooth_on = np.asarray(smooth_on).reshape(gv.N_INI, gv.N_TRIALS, gv.n_size[0]) 

theta = np.linspace(-180, 180, gv.n_size[0])
smooth_off_avg = np.nanmean(smooth_off, axis=0) # avg over ini 
smooth_off_avg = np.nanmean(smooth_off_avg, axis=0) # avg over trials

smooth_on_avg = np.nanmean(smooth_on, axis=0) 
smooth_on_avg = np.nanmean(smooth_on_avg, axis=0) 

smooth_off_std = np.nanstd( np.vstack(smooth_off), axis=0) 
smooth_on_std = np.nanstd( np.vstack(smooth_on), axis=0) 

if DELAY:
    figname = gv.folder + '_on_' + 'tuning_delay'
else:
    figname = gv.folder + '_on_' + 'tuning_stim'
    
plt.figure(figname, figsize=(2.427, 1.5))

pal = [sns.color_palette('tab10')[0],
       sns.color_palette('tab10')[1]]

plt.plot(theta, smooth_off_avg, color=pal[0])
plt.fill_between(theta, smooth_off_avg - smooth_off_std, smooth_off_avg + smooth_off_std,
                 color=pal[0], alpha=.25) 

plt.plot(theta, smooth_on_avg, color=pal[1])
plt.fill_between(theta, smooth_on_avg - smooth_on_std, smooth_on_avg + smooth_on_std,
                 color=pal[1], alpha=.25)

if DELAY:
    plt.title('Delay')
else:
    plt.title('Stimulation')
    
plt.xlabel('Prefered Location (°)')
plt.xticks([-180, -90, 0, 90, 180])
plt.ylabel('Rates (Hz)') 
plt.ylim([0, 25])
plt.savefig(figname + '.svg', dpi=300)
