import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy

gv.init_param()

from balance_inputs_dist import inputs_dist, vec_Phi
from utils import *
from get_m1 import *

from plot_settings import SetPlotParams
SetPlotParams(5, 2)

time, ff_inputs = get_time_ff_inputs(path=gv.path)
mean_ff_inputs = np.nanmean(ff_inputs, axis=-1)

print('time', time.shape)
print('ff_inputs', ff_inputs.shape)

i_pop=0

pop_inputs = ff_inputs[1:,i_pop,:gv.n_size[i_pop]]
print('pop_inputs', pop_inputs.shape)

m1, phi = decode_bump(pop_inputs)
smooth_ff_inputs = circular_convolution(pop_inputs, int(pop_inputs.shape[1]*.05) )
print('smooth', np.array(smooth_ff_inputs).shape)


figtitle = 'ff_inputs'
fig = plt.figure(figtitle, figsize=(5.663*3, 3.5))

####################################
# m1 dist
####################################

ax = fig.add_subplot(int('131'))

plt.hist(m1, histtype='step', color=gv.pal[0], lw=2, density=1)
plt.vlines(np.mean(m1[1:]), 0, 1, lw=2, colors=gv.pal[0])
plt.xlabel('FF Amplitude')

####################################
# phi dist
####################################
ax = fig.add_subplot(int('132'))

phi = 2*phi
phi -= np.pi
# phi *= 180 / np.pi

plt.hist(phi, histtype='step', color=gv.pal[0], lw=2, density=1)
plt.xlabel('FF Phase')

bins_fit = np.linspace(-np.pi, np.pi, 100)
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
    plt.plot(theta, smooth_ff_inputs[i,:].T - np.mean(smooth_ff_inputs[i,:].T) , color=gv.pal[0])

plt.xlabel('Prefered Location (°)')
plt.xticks([-180, -90, 0, 90, 180])
plt.ylabel('FF Inputs')

plt.show()
