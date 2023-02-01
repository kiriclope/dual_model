import sys, os, importlib
import numpy as np 
import matplotlib.pyplot as plt 
import random as rand 

import params as gv 
importlib.reload(sys.modules['params']) 

from mean_field_spec_rank_2_alt import get_m0_m1_mf

gv.init_param()

kappa_list = np.arange(21) 

m00 = np.zeros( (kappa_list.shape[0], gv.n_pop) ) 
m10 = np.zeros( (kappa_list.shape[0], gv.n_pop) ) 
m01 = np.zeros( (kappa_list.shape[0], gv.n_pop) ) 

x0 = np.array([rand.random() for i in range(0, 4*gv.n_pop)] ) 

for i_kappa in range(kappa_list.shape[0]):
    gv.KAPPA = kappa_list[i_kappa]
    m00[i_kappa], m10[i_kappa], m01[i_kappa], x0 = get_m0_m1_mf(kappa=gv.KAPPA, J=gv.J, J2=gv.J2, K=gv.K, ext_inputs=gv.ext_inputs, n_pop=gv.n_pop, verbose=1) 

figname = 'm0_m1_kappa_mf_' + gv.folder 

if plt.fignum_exists(1)==False: 
    print('create fig') 
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(1.25*1.618*1.5*3, 1.618*1.5), num=figname) 
else:
    (ax1, ax2, ax3) = plt.gcf().get_axes() 

ax1.set(xlim=(kappa_list[0], kappa_list[-1]), ylim=(0, np.ceil( np.nanmax(m00)*1.2 ) ) )
ax1.set_xlabel('$\kappa$')
ax1.set_ylabel('$\\nu^{(0)}$')

ax2.set(xlim=(kappa_list[0], kappa_list[-1]), ylim=(0, np.ceil( np.nanmax(m10)*1.2 ) ) )
ax2.set_xlabel('$\kappa$')
ax2.set_ylabel('$\\nu^{(10)}$')

ax3.set(xlim=(kappa_list[0], kappa_list[-1]), ylim=(0, np.ceil( np.nanmax(m01)*1.2 ) ) )
ax3.set_xlabel('$\kappa$')
ax3.set_ylabel('$\\nu^{(01)}$')

ax1.plot(kappa_list, m00, 'o') 
ax2.plot(kappa_list, m10, 'o') 
ax3.plot(kappa_list, m01, 'o') 
