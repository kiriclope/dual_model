import sys, os, importlib
from importlib import reload
from scipy.signal import savgol_filter 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import params as gv 
importlib.reload(sys.modules['params']) 

from get_m1 import *
    
gv.init_param()
    
filter_rates = pd.read_csv(gv.path + 'filter_rates.dat', sep='\s+').to_numpy() 
# print(filter_rates.shape)

time = filter_rates[:,0] / 1000
# print('time', time.shape)

rates = np.delete(filter_rates, [0], axis=1)
# print('rates', rates.shape) 

if gv.n_pop!=1:
    n_neurons = int(rates.shape[1]/2)
    rates = np.reshape(rates, (rates.shape[0], 2, n_neurons))
else: 
    n_neurons = rates.shape[0] 
    
# print('rates', rates.shape)

mean_rates = np.mean(rates, axis=-1) 

# print(mean_rates.shape) 

avg_mean_rates = np.mean(mean_rates, axis=0) 
print('avg_mean_rates', avg_mean_rates) 

figtitle = 'spatial_profile_' + gv.folder 
fig = plt.figure(figtitle, figsize=(1.25*1.618*1.5*2, 1.618*1.25)) 
# ax = fig.add_subplot(1,3,1)

# plt.plot(time, mean_rates[:,0], lw=2, color='r') 
# plt.plot(time, mean_rates[:,1], lw=2, color='b')

# plt.title('$m_E=$%.2f, $m_I=$%.2f' % (avg_mean_rates[0], avg_mean_rates[1]) ) 
# plt.xlabel('Time (s)') 
# plt.ylabel('Rates (Hz)') 
# plt.ylim([0, 100]) 

ax = fig.add_subplot(1,2,1) 
theta = np.linspace(0, np.pi, gv.n_size)

smooth_rates = rates 
# smooth_rates = circular_convolution(rates, int( rates.shape[0]*.01 ), axis=0 ) 
smooth_rates = circular_convolution(smooth_rates, int(rates.shape[-1]*.1) ) 
m1 = compute_m1(smooth_rates)
m1 = circular_convolution(m1, int( m1.shape[0]*.01 ), axis=0 ) 

phi = compute_phi(smooth_rates) 
phi = circular_convolution(phi, int( phi.shape[0]*.01 ), axis=0 )

print(m1.shape)

avg_m1 = np.mean( m1, axis=0) 
print('<m1>', avg_m1) 

for i_pop in range(gv.n_pop): 
    plt.plot(time, m1[:,i_pop], color=gv.pal[i_pop]) 

plt.title('$m^1_E=$%.2f, $m^1_I=$%.2f' % (avg_m1[0], avg_m1[1]) ) 
plt.xlabel('Time (s)') 
plt.ylabel('$m_1$ (Hz)') 
# plt.ylim([-0.1, 1]) 

ax = fig.add_subplot(1,2,2) 
# for i_pop in range(gv.n_pop): 
plt.plot(time, phi[:,0], color=gv.pal[0], alpha=0.25) 

plt.title('$m^1_E=$%.2f, $m^1_I=$%.2f' % (avg_m1[0], avg_m1[1]) ) 
plt.xlabel('Time (s)') 
plt.ylabel('$\phi$ (rad)') 
plt.yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
           [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$']) 

# avg_rates = np.mean(rates, axis=0) # avg over time 

# smooth_avg_rates = circular_convolution(avg_rates, int(rates.shape[-1] * .1 ) ) 
# m1 = compute_m1(smooth_avg_rates) 
# # print(m1.shape) 
# print('<m1>', m1) 

# ax = fig.add_subplot(1,3,3) 
# for i_pop in range(gv.n_pop): 
#     plt.plot(theta, smooth_avg_rates[i_pop], color=gv.pal[i_pop] ) 

# plt.title('$m^1_E=$%.2f, $m^1_I=$%.2f' % (m1[0], m1[1]) ) 
# plt.xlabel('$\\theta$ (rad)') 

# plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
#            ['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{2}$', r'$\pi$'])

# plt.ylabel('Rates (Hz)') 

plt.show()
