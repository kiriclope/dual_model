import sys, os, importlib
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import params as gv 
importlib.reload(sys.modules['params']) 

gv.init_param()

from balance_inputs_dist import inputs_dist, vec_Phi
from utils import *
from get_m1 import compute_m1

rates_off = []
n_trials = 25

gv.folder = 'quench_noise_0.75_off'
for trial in range(1, n_trials+1):
    gv.TRIAL_ID = trial 
    gv.init_param() 
    time, rates = get_time_rates(path=gv.path) 
    rates_off.append(np.mean(rates[20:40], axis=0))

rates_off = np.asarray(rates_off)
rates_off = np.swapaxes(rates_off, 0, -1)
print(rates_off.shape)

m0_off = np.mean(rates_off, axis=-1) # average over trials 
m1_off = compute_m1(rates_off) 

print(m0_off.shape, m1_off.shape)
m1_m0_off = m1_off[:,0]/m0_off[:,0]

plt.hist(m1_m0_off)
