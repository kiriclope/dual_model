import sys, os, importlib
from importlib import reload
from scipy.signal import savgol_filter 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import params as gv 
# importlib.reload(sys.modules['params']) 

from get_m1 import *
from utils import *
from rank_utils import * 

gv.IF_INI_COND = 0 
gv.IF_LOOP_J0 = 0 
gv.IF_HYSTERESIS = 1 
gv.HYST_JEE = int(sys.argv[1]) 
gv.HYST_M0 = int(sys.argv[2]) 

gv.init_param() 
print(gv.path)

time, rates = get_time_rates(path=gv.path, con_path=gv.con_path) 
m0 = np.nanmean(rates, axis=-1).T
m1, phi, smooth_rates =  get_m1_phi_smooth_rates(rates) 

if(gv.RANK==2): 
    time, rates_perm = get_time_rates(path=gv.path, con_path=gv.con_path, MAP=1) 
    m1_perm, phi_perm, smooth_rates_perm =  get_m1_phi_smooth_rates(rates_perm) 

if(gv.HYST_JEE!=0):
    time *= 1000
else:
    time *= 1000*1000 

print(time) 
print(m0[0]) 

figtitle = 'hysteresis_' + gv.folder + '_kappa_%.2f' % gv.KAPPA 

if plt.fignum_exists(1)==False: 
    print('create fig') 
    axis = np.zeros( (1, gv.RANK+1) ) 
    fig, axis = plt.subplots(1, gv.RANK+1, figsize=(1.25*1.618*1.5*(gv.RANK+1), 1.618*1.25), num=figtitle) 
else:
    axis = plt.gcf().get_axes() 

for i_pop in range(gv.n_pop): 
    axis[0].plot(time, m0[i_pop], '-o', color=gv.pal[i_pop]) 
    # axis[0].plot(time, np.flip(m0[i_pop]), lw=2, color=gv.pal[i_pop]) 

if(gv.HYST_JEE!=0):
    axis[0].set_xlabel('$J_{EE}$')
if(gv.HYST_M0!=0):
    axis[0].set_xlabel('$\\nu_{ext}$ (Hz)')
    
axis[0].set_ylabel('$\\nu^{(0)}$ (Hz)') 

for i_pop in range(gv.n_pop): 
    axis[1].plot(time, m1[i_pop], '-o', color=gv.pal[i_pop]) 
    # axis[1].plot(time, np.flip(m1[i_pop]), '-o', color=gv.pal[i_pop]) 

if(gv.RANK==2):
    axis[1].set_title('Map 0') 
    
if(gv.HYST_JEE!=0):
    axis[1].set_xlabel('$J_{EE}$')
if(gv.HYST_M0!=0):
    axis[1].set_xlabel('$\\nu_{ext}$ (Hz)')

axis[1].set_ylabel('$\\nu^{(1)}_0$ (Hz)') 

if(gv.RANK==2):
    
    for i_pop in range(gv.n_pop): 
        axis[2].plot(time, m1_perm[i_pop], '-', color=gv.pal[i_pop]) 

    axis[2].set_title('Map 1')
    if(gv.HYST_JEE!=0):
        axis[2].set_xlabel('$J_{EE}$') 
    if(gv.HYST_M0!=0): 
        axis[2].set_xlabel('$\\nu_{ext}$ (Hz)') 
        
    axis[2].set_ylabel('$\\nu^{(1)}_1$ (Hz)') 

plt.show()

# filename = gv.fig_path + figtitle + ".svg" 
# plt.savefig(filename,format='svg', dpi=300) 
# print('save fig to', filename) 
