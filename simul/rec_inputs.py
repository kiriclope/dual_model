import sys, os, importlib
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy

import params as gv 
importlib.reload(sys.modules['params']) 

gv.init_param()

from balance_inputs_dist import inputs_dist, vec_Phi
from utils import *
from get_m1 import *

from plot_settings import SetPlotParams
SetPlotParams(5, 2) 

time, inputs = get_time_inputs(path=gv.path) 

print('time', time.shape)
print('inputs', inputs.shape) 

i_pop=0
# bins = [int(0/gv.T_WINDOW), int(2/gv.T_WINDOW)] 
bins = [int(3/gv.T_WINDOW), -1] 

pop_inputs = np.mean(inputs[bins[0]:bins[1], :, :gv.n_size[i_pop]], axis=1) 
print('pop_inputs', pop_inputs.shape) 

m1, phi = decode_bump(pop_inputs)
smooth_inputs = circular_convolution(pop_inputs, int(pop_inputs.shape[1]*.1) ) 
print('smooth', np.array(smooth_inputs).shape)


figtitle = 'net_inputs_tuning'
fig = plt.figure(figtitle, figsize=(5.663*3, 3.5)) 

####################################
# m1 dist
####################################

ax = fig.add_subplot(int('131'))

plt.hist(m1, histtype='step', color=gv.pal[0], lw=2, density=1)
plt.vlines(np.mean(m1[1:]), 0, 1, lw=2, colors=gv.pal[0])
plt.xlabel('Net Input Amplitude') 

####################################
# phi dist
####################################
ax = fig.add_subplot(int('132')) 

phi = 2*phi
phi -= np.pi 

phi -= np.mean(phi) 
# phi *= 180 / np.pi 

plt.hist(phi, histtype='step', color=gv.pal[0], lw=2, density=1)
plt.xlabel('Net Input Diffusion') 

bins_fit = np.linspace(-.1, .1, 100) 
mu, sigma = scipy.stats.norm.fit(phi) 
fit = scipy.stats.norm.pdf(bins_fit, mu, sigma) 
plt.plot(bins_fit, fit, color=gv.pal[0], lw=2) 
# plt.xticks([-180, -90, 0, 90, 180]) 

###################################
# tuning curves 
###################################
ax = fig.add_subplot(int('133')) 

theta = np.linspace(-180, 180, gv.n_size[i_pop]) 

for i in range(1,10):
    plt.plot(theta, smooth_inputs[i,:].T - np.mean(smooth_inputs[i,:].T) , color=gv.pal[0]) 

plt.xlabel('Prefered Location (°)')
plt.xticks([-180, -90, 0, 90, 180])
plt.ylabel('Net Inputs') 

plt.show()
