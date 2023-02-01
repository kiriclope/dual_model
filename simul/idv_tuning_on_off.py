import sys, os, importlib
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import params as gv 
importlib.reload(sys.modules['params']) 

from balance_inputs_dist import inputs_dist, vec_Phi
from utils import *
from get_m1 import decode_bump

def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    
rates_off = []
n_trials = 10
bins =  [int(4/gv.T_WINDOW), int(5/gv.T_WINDOW)] 

gv.folder = 'quench_noise_0.75_off'
gv.IF_INI_COND = 1 
gv.IF_TRIALS = 1

gv.A_CUE = 0.25

for trial in range(1, gv.N_TRIALS+1):
    gv.TRIAL_ID = trial
    gv.PHI_CUE = trial / gv.N_TRIALS 
    gv.PHI_DIST = 1 - trial / gv.N_TRIALS 
    gv.init_param()
    print(gv.path)
    time, rates = get_time_rates(path=gv.path) 
    rates_off.append(np.nanmean(rates[bins[0]:bins[1]], axis=0))

rates_off = np.asarray(rates_off)
rates_off = np.swapaxes(rates_off, 0, -1)
print(rates_off.shape)

m0_off = np.nanmean(rates_off, axis=-1) # average over trials 
m1_off, _ = decode_bump(rates_off)

print(m0_off.shape, m1_off.shape)
m1_m0_off = m1_off[:,0]/m0_off[:,0]

print(m1_m0_off.shape)

rates_on = []
gv.folder = 'quench_on'
for trial in range(1, gv.N_TRIALS+1):
    gv.TRIAL_ID = trial 
    gv.PHI_CUE = trial / gv.N_TRIALS 
    gv.PHI_DIST = 1 - trial / gv.N_TRIALS 
    gv.init_param() 
    time, rates = get_time_rates(path=gv.path) 
    rates_on.append(np.nanmean(rates[bins[0]:bins[1]], axis=0)) 

rates_on = np.asarray(rates_on)
rates_on = np.swapaxes(rates_on, 0, -1)
print(rates_on.shape)

m0_on = np.nanmean(rates_on, axis=-1)
m1_on, _ = decode_bump(rates_on)
print(m0_on.shape, m1_on.shape)

m1_m0_on = m1_on[:,0]/m0_on[:,0]
print(m1_m0_on.shape)

m1_m0_off[np.argwhere(m0_off<1)] = np.nan
m1_m0_on[np.argwhere(m0_on<1)] = np.nan

figname = 'osi_hist_off_on'
plt.figure(figname, figsize=(5.663, 3.5))
plt.hist(m1_m0_off/2, histtype='step', color='b', lw=2, alpha=.25, bins='auto')
plt.hist(m1_m0_on/2, histtype='step', color='r', lw=2, alpha=.25, bins='auto') 

# fig = plt.figure('tuning_on_off', figsize=(1.618*1.5, 1.618*1.5)) 
# plt.scatter(m1_m0_off, m1_m0_on)
# plt.xlabel('m1/m0 off')
# plt.ylabel('m1/m0 on')

# plt.plot([0,2],[0,2], 'k--') 
# # definitions for the axes
# left, width = 0.1, 0.65
# bottom, height = 0.1, 0.65
# spacing = 0.005

# rect_scatter = [left, bottom, width, height]
# rect_histx = [left, bottom + height + spacing, width, 0.2]
# rect_histy = [left + width + spacing, bottom, 0.2, height]

# # start with a square Figure
# fig = plt.figure(figsize=(8, 8))

# ax = fig.add_axes(rect_scatter)
# ax_histx = fig.add_axes(rect_histx, sharex=ax)
# ax_histy = fig.add_axes(rect_histy, sharey=ax)

# # use the previously defined function
# scatter_hist(m1_m0_off, m1_m0_on, ax, ax_histx, ax_histy)
