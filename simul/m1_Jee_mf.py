import sys, os, importlib
import numpy as np 
import matplotlib.pyplot as plt 
import random as rand 

import params as gv 
importlib.reload(sys.modules['params']) 
# from mean_field_spec import get_m0_m1_mf
from mean_field_spec_stp import get_m0_m1_mf

gv.init_param()

Jee_list = np.linspace(0.5,1.5,10) 

m0 = np.zeros( (Jee_list.shape[0], gv.n_pop) ) 
m1 = np.zeros( (Jee_list.shape[0], gv.n_pop) ) 
x0 = np.array([rand.random() for i in range(0,3*gv.n_pop)] ) 

for i_Jee in range(Jee_list.shape[0]): 
    gv.JEE = Jee_list[i_Jee]
    gv.JEE2 = Jee_list[i_Jee] * Jee_list[i_Jee]     
    m0[i_Jee], m1[i_Jee], x0 = get_m0_m1_mf(kappa=gv.KAPPA, J=gv.J, J2=gv.J2, K=gv.K, ext_inputs=gv.ext_inputs, n_pop=gv.n_pop, JEE=gv.JEE, JEE2=gv.JEE2, verbose=1) 

m0 = m0.T
m1 = m1.T

figname = 'm0_m1_Jee_mf_' + gv.folder 

if plt.fignum_exists(1)==False: 
    print('create fig') 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(1.25*1.618*1.5*2, 1.618*1.5), num=figname) 
else:
    (ax1, ax2) = plt.gcf().get_axes() 

ax1.set(xlim=(Jee_list[0], Jee_list[-1]), ylim=(0, np.ceil( np.nanmax(m0)*1.2 ) ) )
ax1.set_xlabel('$J_{EE}$')
ax1.set_ylabel('$\\nu^{(0)}$')

ax2.set(xlim=(Jee_list[0], Jee_list[-1]), ylim=(0, np.ceil( np.nanmax(m1)*1.2 ) ) )
ax2.set_xlabel('$J_{EE}$')
ax2.set_ylabel('$\\nu^{(1)}$')

for i_pop in range(2):
    ax1.plot(Jee_list, m0[i_pop], 'o', color=gv.pal[i_pop])
    ax2.plot(Jee_list, m1[i_pop], 'o', color=gv.pal[i_pop]) 
