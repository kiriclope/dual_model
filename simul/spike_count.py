import sys, os, importlib 
from importlib import reload 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from joblib import Parallel, delayed 
import progressbar as pgb 

import params as gv 
importlib.reload(sys.modules['params']) 

gv.init_param()
    
# raw_spike_times = pd.read_csv('/homecentral/alexandre.mahrach/bistable_mongilo/src/data/spikes.txt', sep='\s+').to_numpy() 
raw_spike_times = pd.read_csv(gv.path + '/spike_times.dat', sep='\s+').to_numpy() 
print('data', raw_spike_times.shape) 

neurons_id = raw_spike_times[:,0] 
print('neurons_id', neurons_id.shape, neurons_id[500:505]) 

spike_times = raw_spike_times[:,1] 
print('spike_times', spike_times.shape, spike_times[500:505]) 

def parloop_spkC(neurons_id, i_neuron): 

    idx_neuron = neurons_id==i_neuron
    print(idx_neuron)
    spike_count = np.sum(idx_neuron)
    
    return spike_count 

n_neurons = gv.n_neurons * 10000 

spike_count = parloop_spkC(neurons_id, 0)
print(spike_count)

# with pgb.tqdm_joblib( pgb.tqdm(desc='spike_count', total=n_neurons) ) as progress_bar: 
#     spike_count = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(parloop_spkC)(neurons_id, i_neuron) for i_neuron in range(n_neurons) ) 

# spike_count = np.array(spike_count) 
# print('spike_count', spike_count.shape, spike_count[:10]) 

# figtitle = 'spike_count' 
# fig = plt.figure(figtitle, figsize=(1.25*1.618*1.5, 1.618*1.25)) 

# plt.hist(spike_count[:gv.n_size[0]], histtype='step', ls='-', color='r') 
# plt.hist(spike_count[gv.n_size[0]:], histtype='step', ls='-', color='b') 

# plt.xlabel('CV') 
# plt.ylabel('Count') 
