import pandas as pd
import matplotlib
from matplotlib.offsetbox import AnchoredText
from matplotlib.animation import FuncAnimation, PillowWriter

import sys, os, importlib
from importlib import reload
from scipy.signal import savgol_filter 
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDrawingArea

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import params as gv 
importlib.reload(sys.modules['params']) 

from get_m1 import *
from utils import *
from rank_utils import *

gv.init_param()

alpha = [1, 0.1]
    
if(gv.MAP==0): 
    time, rates = get_time_rates(path=gv.path, con_path=gv.con_path) 
    m1, phi, smooth_rates, m1_osc = get_m1_phi_smooth_rates(rates,osc=1) 

# print(m1_osc)

if(gv.MAP==1): 
    time, rates_perm =  get_time_rates(MAP=1, path=gv.path, con_path=gv.con_path) 
    m1, phi, smooth_rates = get_m1_phi_smooth_rates(rates_perm) 

idx_phi_1 = np.where(phi[0]<=np.pi/2) 
idx_phi_2 = np.where(phi[0]>np.pi/2)

phi_1 = phi[0].copy()
phi_1[idx_phi_2] *= np.nan 

# phi_1 = pd.Series(phi_1)
# phi_1.fillna(method='ffill', limit=0)

phi_2 = phi[0].copy()
phi_2[idx_phi_1] *= np.nan 
# phi_2 = pd.Series(phi_2)
# phi_2.fillna(method='ffill', limit=0)
    
if(gv.IF_DPA): 
    figname = 'smooth_m1_phi_DPA_' + gv.folder + '_MAP_%d'  % gv.MAP 
    
    if(gv.PHI_EXT==3*np.pi/2): 
        figtitle = 'DPA, ODOR A, MAP %d' % gv.MAP 
    if(gv.PHI_EXT==np.pi/2): 
        figtitle = 'DPA, ODOR B, MAP %d' % gv.MAP 
        
elif(gv.IF_DUAL): 
    figname = 'smooth_m1_phi_DUAL_' + gv.folder + '_MAP_%d'  % gv.MAP

    if(gv.PHI_EXT==3*np.pi/2): 
        figtitle = 'Dual Go, ODOR A, MAP %d' % gv.MAP 
    if(gv.PHI_EXT==np.pi/2): 
        figtitle = 'Dual Go, ODOR B, MAP %d' % gv.MAP 
    
elif(gv.IF_DRT): 
    figname = 'smooth_m1_phi_DRT_' + gv.folder + '_MAP_%d'  % gv.MAP 
    figtitle = 'DRT Go, MAP %d' % gv.MAP 
else:
    figname = 'smooth_m1_phi_tasks_' + gv.folder + '_MAP_%d'  % gv.MAP 
    figtitle = 'DRT Go, MAP %d' % gv.MAP 
    
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(1.25*1.618*1.5*3, 1.618*1.5), num=figname) 

fig.suptitle(figtitle) 
ax1.set(xlim=(0, np.pi), ylim=(0, np.ceil( np.nanmax(smooth_rates)*1.2 ) ) )

if(gv.MAP==0): 
    ax1.set_xlabel('$\\theta$ (rad)')
    if(gv.model=='lif'):
        ax1.set_ylabel('Rates($\\theta$) (Hz)')
        
if(gv.MAP==1): 
    ax1.set_xlabel('$\phi$ (rad)') 
    if(gv.model=='lif'):
        ax1.set_ylabel('Rates($\phi$) (Hz)') 

ax1.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]) 
ax1.set_xticklabels(['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$']) 
    
ax2.set(xlim=(0, 14), ylim=(0, np.ceil( np.nanmax(m1) * 12 )/10 ) ) 
ax2.set_xlabel('Time (s)')
ax2.set_xticks([0, 2, 4, 6, 8, 10, 12, 14]) 
    
ax2.set_ylabel('Amplitude (Hz)') 
    
add_vlines_axis(ax2) 

ax3.set(xlim=(0, 14), ylim=(0, np.pi)) 
ax3.set_xlabel('Time (s)')

ax3.set_ylabel('Phase (rad)')
    
ax3.set_xticks([0, 2, 4, 6, 8, 10, 12, 14]) 
# ax3.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]) 
# ax3.set_yticklabels(['$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2 \pi$']) 

ax3.set_yticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]) 
ax3.set_yticklabels(['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\frac{3\pi}{4}$', r'$\pi$']) 

add_vlines_axis(ax3)    

for i_pop in range(gv.n_pop): 
    theta = np.linspace(0, np.pi, gv.n_size[i_pop]) 
    if(i_pop==0):
        smooth_line_E = ax1.plot(theta, smooth_rates[i_pop, 0,:gv.n_size[i_pop]], '-', color=gv.pal[i_pop])[0]
    else:
        smooth_line_I = ax1.plot(theta, smooth_rates[i_pop, 0,:gv.n_size[i_pop]], '-', color=gv.pal[i_pop])[0]

for i_pop in range(gv.n_pop):
    if(i_pop==0):    
        m1_line_E = ax2.plot(time[0], m1[i_pop,0], '-', color=gv.pal[i_pop])[0] 
    else:
        m1_line_I = ax2.plot(time[0], m1[i_pop,0], '-', color=gv.pal[i_pop])[0] 
        
for i_pop in range(gv.n_pop): 
    if(i_pop==0):    
        phi_line_E = ax3.plot(time[0], phi[i_pop,0], '-', color=gv.pal[i_pop], alpha=alpha[i_pop])[0] 
        # phi_line_E_1 = ax3.plot(time[0], phi[i_pop,0], '-', color=gv.pal[i_pop], alpha=alpha[i_pop])[0] 
    else:
        phi_line_I = ax3.plot(time[0], phi[i_pop,0], '-', color=gv.pal[i_pop], alpha=alpha[i_pop])[0]

# # function that draws each frame of the animation 
def animate(i):

    theta = np.linspace(0, np.pi, gv.n_size[0])
    smooth_line_E.set_xdata(theta)
    smooth_line_E.set_ydata(smooth_rates[0, i,:gv.n_size[0]])
    
    theta = np.linspace(0, np.pi, gv.n_size[1])
    smooth_line_I.set_xdata(theta)     
    smooth_line_I.set_ydata(smooth_rates[1, i,:gv.n_size[1]]) 
    
    m1_line_E.set_xdata(time[:i]) 
    m1_line_E.set_ydata(m1[0,:i]) 

    m1_line_I.set_xdata(time[:i]) 
    m1_line_I.set_ydata(m1[1,:i]) 
    
    phi_line_E.set_xdata(time[:i]) 
    phi_line_E.set_ydata(phi[0,:i]) 
    
    # phi_line_E_1.set_xdata(time[:i]) 
    # phi_line_E_1.set_ydata(phi_2[:i]) 
    
    phi_line_I.set_xdata(time[:i]) 
    phi_line_I.set_ydata(phi[1,:i]) 
     
# # run the animation
anim = FuncAnimation(fig, animate, frames=time.shape[0], interval=100, repeat=True, cache_frame_data=False) 
plt.draw()
plt.show()

# filename = gv.fig_path + "/" + figname + ".gif" 

# print("saving to:", filename) 
# writergif = PillowWriter(fps=60) 
# anim.save(filename, writer=writergif, dpi=150) 

# plt.close('all') 
