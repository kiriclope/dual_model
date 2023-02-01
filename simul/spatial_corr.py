import sys, os, importlib 
from importlib import reload 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from nancorrmp.nancorrmp import NaNCorrMp
from statsmodels.tsa.stattools import acf
from joblib import Parallel, delayed 
import progressbar as pgb 

import params as gv 
importlib.reload(sys.modules['params']) 

from numpy.fft import fft, ifft

gv.init_param()

def periodic_corr(x, y):
    return ifft(fft(x) * fft(y).conj()).real

raw_spike_times = pd.read_csv(gv.path + '/spike_times.dat', sep='\s+').to_numpy() 
print('data', raw_spike_times.shape) 

neurons_id = raw_spike_times[:,0] 
spike_times = raw_spike_times[:,1] 

# idx = np.logical_and(spike_times>0, spike_times<2000) 
# neurons_id = neurons_id[idx] # time in ms
# spike_times = spike_times[idx] # time in ms 

idx = np.logical_and(neurons_id>0, neurons_id<32000) 
neurons_id = neurons_id[idx] # time in ms
spike_times = spike_times[idx] # time in ms 

print('neurons_id', neurons_id.shape, neurons_id[:5]) 
print('spike_times', spike_times.shape, spike_times[:5]) 

AC=[]
neurons_t = []
dum = spike_times[0]
counter=0

df = pd.DataFrame(data = np.vstack([spike_times, neurons_id]).T, columns=['time', 'id'])
# print(df)
# def autocorr(x, lags):
#     xcorr = np.correlate(x - x.mean(), x - x.mean(), 'full')  # Compute the autocorrelation
#     xcorr = xcorr[xcorr.size//2:] #/ xcorr.max()               # Convert to correlation coefficients
#     return xcorr[:lags+1]                                     # Return only requested lags


def autocorr(y, lag):
    acf_list = [] 
    mu = y.mean() 
    # print('mu', mu) 
    
    for i_lag in range(0, lag):
        acf_list.append(sum((y - mu).iloc[i_lag:] * (y.shift(i_lag) - mu).iloc[i_lag:]) / sum((y - mu) ** 2))

    
    return np.array(acf_list)

mat = np.zeros((int(2/.1), 32000)) 

AC = np.zeros((100, 41)) * np.nan 
t=0.1
counter=0
while t<2:
    print('np spks', df[df.time==t].shape[0])
    
    times_t = df[df.time==t].time.to_numpy()
    neurons_t = df[df.time==t].id.to_numpy()
    
    print('t', t, 'spike times', times_t, 'neurons', neurons_t)
    t = round(t + .1, 1)
    
    # neuron_df = pd.DataFrame(neurons_t)
    mat[counter, neurons_t.astype(int)] = 1
    # corr = acf(neuron_df, 100)
    # AC[counter, :corr.shape[0]] = corr 

    counter = counter+1 
    
# plt.figure(figsize=(5.663, 3.5))
# plt.plot(np.nanmean(AC,0))

# AC = np.zeros((100, 41)) * np.nan
# neuron = 0 
# while neuron<32000:
    
#     times_t = df[df.id==neuron].time.to_numpy()
#     neurons_t = df[df.id==neuron].id.to_numpy()
#     # print('neuron', neuron, 'spike times', times_t, 'neurons', neurons_t)

#     try:
#         neuron_df = pd.DataFrame(times_t) 
#         corr = acf(neuron_df, 100)
#         # print(corr.shape)
#         AC[neuron, :corr.shape[0]] = corr
#     except:
#         pass
#     neuron = neuron+1

# # AC = np.array(AC)

# plt.figure(figsize=(5.663, 3.5))
# plt.plot(np.nanmean(AC, axis=0))
