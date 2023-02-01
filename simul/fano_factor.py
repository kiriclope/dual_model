import sys, importlib

import scipy
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat

from joblib import Parallel, delayed 
import progressbar as pgb 

from get_m1 import * 
from utils import * 
from write import *

importlib.reload(sys.modules['params'])
importlib.reload(sys.modules['get_m1']) 

gv.IF_INI_COND = 0
gv.IF_TRIALS = 0

gv.N_INI = 100
gv.folder = 'quench_off'
gv.init_param()

path_bu = gv.path

def parloop_spkC(neurons_id, i_neuron): 

    idx_neuron = neurons_id==i_neuron
    spike_count = np.sum(idx_neuron) 
    
    return spike_count 

def get_spike_count(path, bins=[3000, 4000], n_size=gv.n_size):
    
    raw_spike_times = pd.read_csv(path + '/spike_times.dat', sep='\s+').to_numpy() 
    # print('data', raw_spike_times.shape)     
    neurons_id = raw_spike_times[:,0] 
    # print('neurons_id', neurons_id.shape, neurons_id[500:505]) 
    
    spike_times = raw_spike_times[:,1] 
    # print('spike_times', spike_times.shape, spike_times[500:505]) 
    
    idx = np.logical_and(spike_times>bins[0], spike_times<bins[1]) 
    neurons_id = neurons_id[idx] # time in ms 
    
    n_neurons = n_size[0]
    
    # with pgb.tqdm_joblib( pgb.tqdm(desc='spike_count', total=n_neurons) ) as progress_bar: 
    spike_counts = Parallel(n_jobs=-1, backend='multiprocessing')(
        delayed(parloop_spkC)(neurons_id, i_neuron)
        for i_neuron in range(n_neurons)
    ) 
    
    return spike_counts

def parloop_ff(i_ini, path):
    path += '/ini_cond_%d' % i_ini ; 

    spike_count_bl = get_spike_count(path,  bins=[0, 2000])
    spike_count_stim = get_spike_count(path,  bins=[2000, 3000])
    spike_count_delay = get_spike_count(path,  bins=[4000, 5000])

    return spike_count_bl, spike_count_stim, spike_count_delay
    
def get_fano_factor(path):
    spike_count_inis = []
    # spike_count_bl = [] 
    # spike_count_stim = []
    # spike_count_delay = []

    with pgb.tqdm_joblib( pgb.tqdm(desc='spike_count', total=gv.N_INI) ) as progress_bar:
        spike_count_bl, spike_count_stim, spike_count_delay = zip(*Parallel(n_jobs=-1,
                                                                            backend='multiprocessing')(
            delayed(parloop_ff)(i_ini, path) 
            for i_ini in range(1, gv.N_INI+1) )
        )
    # for i_ini in range(1, gv.N_INI + 1):
    #     path = path_bu
    #     path += '/ini_cond_%d' % i_ini ; 
    #     # gv.path += '/trial_%d' % i_ini ; 
    #     print(path)
        
    #     spike_count_bl.append(get_spike_count(path,  bins=[0, 2000])) 
    #     spike_count_stim.append(get_spike_count(path,  bins=[2000, 3000])) 
    #     spike_count_delay.append(get_spike_count(path,  bins=[3000, 4000])) 
    
    spike_count_inis = np.asarray([spike_count_bl, spike_count_stim, spike_count_delay])
    spike_count_inis = np.asarray(spike_count_inis)
    print('spike_count', spike_count_inis.shape, spike_count_inis[0, :10]) 
    
    mean_spike_count = np.nanmean(spike_count_inis, axis=1) # over inis 
    var_spike_count = np.nanvar(spike_count_inis, axis=1) 
    
    fano_factor = var_spike_count / mean_spike_count 
    
    print('fano_factor', fano_factor.shape) 
    
    return fano_factor, spike_count_inis 

fano_off, spike_off = get_fano_factor(path_bu)
path_bu = path_bu.replace('quench_off', 'quench_on') # change dirname 
fano_on, spike_on = get_fano_factor(path_bu)

# figname = 'off_on_' + 'fano_factor_hist'

# plt.figure(figname)
# plt.hist(fano_off, histtype='step', color='b', lw=2) 
# plt.hist(fano_on, histtype='step', color='r', lw=2) 
# plt.xlabel('Fano Factor') 
# plt.ylabel('Count')

# plt.savefig(figname + '.svg', dpi=300)

figname = 'off_on_' + 'fano_factor_bar'
plt.figure(figname, figsize=(5.663, 3.5))
mean = [np.nanmean(fano_off, axis=1), np.nanmean(fano_on, axis=1)]
# sem = [np.nanstd(fano_off, axis=1, ddof=1)/np.sqrt(np.sum(~np.isnan(fano_off), axis=1)),
#        np.nanstd(fano_on, axis=1, ddof=1)/np.sqrt(np.sum(~np.isnan(fano_on), axis=1))]

# sem = [np.nanstd(fano_off, axis=1, ddof=1),
#        np.nanstd(fano_on, axis=1, ddof=1)]

sem = [scipy.stats.sem(fano_off, axis=1, ddof=1, nan_policy='omit'),
       scipy.stats.sem(fano_on, axis=1, ddof=1, nan_policy='omit')]

pal = [sns.color_palette('colorblind')[0],
       sns.color_palette('colorblind')[2]]

plt.bar([0,3,6] , mean[0], 1, yerr=sem[0], color=pal[0])
plt.bar([1,4,7] , mean[1], 1, yerr=sem[1], color=pal[1])

plt.ylabel('Fano Factor')
plt.xticks([0.5,3.5,6.5], ['Baseline', 'Cue', 'Delay'])

plt.savefig(figname + '.svg', dpi=300)
