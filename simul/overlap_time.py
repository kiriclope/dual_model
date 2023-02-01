import sys, os, importlib
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt

import params as gv
importlib.reload(sys.modules['params'])
from utils import *
from plot_settings import *
SetPlotParams()

from shuffle import shuffle_stat

gv.init_param()

time, rates = get_time_rates(path=gv.path)
print('rates',rates.shape)

figtitle = 'rates_overlap_time_'
fig = plt.figure(figtitle, figsize=(5.663 * gv.RANK, 3.5))
ax = fig.add_subplot(1, gv.RANK, 1)

overlap = get_overlap(rates, n_size=gv.n_size, ksi_path=gv.ksi_path, MAP=0)
plt.plot(time, overlap[0], '-', color=gv.pal[0] , alpha=gv.alpha )
plt.xlabel('Time (s)')
plt.ylabel('Sample Overlap')

if(gv.IF_ADD_VLINES):
    add_vlines()

plt.xticks([0, 2, 4, 6, 8, 10, 12])
plt.xlim([0, 12])

print('rates', rates.shape)
print('overlap', overlap.shape)

ksi_path = gv.ksi_path
n_size= gv.n_size

# overlap_shuffle = shuffle_stat(rates, lambda x: get_overlap(x, n_size, ksi_path, 0)  )
# print('overlap shuffle', overlap_shuffle.shape)

# mean_shuffle = np.nanmean(overlap_shuffle[:,0], axis=0)
# perc_shuffle = np.nanpercentile(overlap_shuffle[:,0], [2.5, 97.5], axis=0)
# print('mean', mean_shuffle.shape, 'perc', perc_shuffle.shape)

# plt.plot(time, mean_shuffle, '--', color=gv.pal[0] )
# plt.fill_between(time, perc_shuffle[0], perc_shuffle[1], alpha=0.1)

if(gv.RANK==2):
    overlap_1 = get_overlap(rates, n_size=gv.n_size, ksi_path=gv.ksi_path, MAP=1)
    ax = fig.add_subplot(1,gv.RANK,2)
    plt.plot(time, overlap_1[0], '-',  color=gv.pal[0], alpha=gv.alpha )
    plt.xlabel('Time (s)')
    plt.ylabel('Distractor Overlap')
    if(gv.IF_ADD_VLINES):
        add_vlines()
    plt.xticks([0, 2, 4, 6, 8, 10, 12])
    plt.xlim([0, 12])

    # overlap_shuffle = shuffle_stat(rates, lambda x: get_overlap(x, n_size, ksi_path, 1)  )
    # print('overlap shuffle', overlap_shuffle.shape)

    # mean_shuffle = np.nanmean(overlap_shuffle[:,0], axis=0)
    # perc_shuffle = np.nanpercentile(overlap_shuffle[:,0], [2.5, 97.5], axis=0)
    # print('mean', mean_shuffle.shape, 'perc', perc_shuffle.shape)

    # plt.plot(time, mean_shuffle, '--', color=gv.pal[0] )
    # plt.fill_between(time, perc_shuffle[0], perc_shuffle[1], alpha=0.1)
plt.savefig('overlap_Go_first.svg', dpi=300, format='svg')
