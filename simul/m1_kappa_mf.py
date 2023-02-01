import sys, os, importlib
import numpy as np 
import matplotlib.pyplot as plt 
import random as rand 

import params as gv 
importlib.reload(sys.modules['params']) 
from mean_field_spec import get_m0_m1_mf
# from mean_field_spec_stp import get_m0_m1_mf

gv.init_param()

kappa_list = np.arange(51)[20:] 

m0 = np.zeros( (kappa_list.shape[0], gv.n_pop) ) 
m1 = np.zeros( (kappa_list.shape[0], gv.n_pop) ) 
x0 = np.array([rand.random() for i in range(0,3*gv.n_pop)] ) 

J = gv.J
J2 = J*J
# J = gv.J.reshape(gv.n_pop,gv.n_pop)
# J2 = J*J
# print(J[0][0])

for i_kappa in range(kappa_list.shape[0]): 
    gv.KAPPA = kappa_list[i_kappa] 
    # m0[i_kappa], m1[i_kappa], x0 = get_m0_m1_mf(kappa=gv.KAPPA, J=J, J2=J2, K=gv.K, ext_inputs=gv.ext_inputs, n_pop=gv.n_pop, JEE=gv.JEE, JEE2=gv.JEE2, verbose=1) 

    m0[i_kappa], m1[i_kappa], x0 = get_m0_m1_mf(kappa=gv.KAPPA, J=J, J2=J2, K=gv.K, ext_inputs=gv.ext_inputs, n_pop=gv.n_pop, verbose=1) 
    
figname = 'm0_m1_kappa_mf_' + gv.folder 

if plt.fignum_exists(1)==False: 
    print('create fig') 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(1.25*1.618*1.5*2, 1.618*1.5), num=figname) 
else:
    (ax1, ax2) = plt.gcf().get_axes() 

ax1.set(xlim=(kappa_list[0], kappa_list[-1]), ylim=(0, np.ceil( np.nanmax(m0)*1.2 ) ) )
ax1.set_xlabel('$\kappa$')
ax1.set_ylabel('$\\nu^{(0)}$')

ax2.set(xlim=(kappa_list[0], kappa_list[-1]), ylim=(0, np.ceil( np.nanmax(m1)*1.2 ) ) )
ax2.set_xlabel('$\kappa$')
ax2.set_ylabel('$\\nu^{(1)}$')

ax1.plot(kappa_list, m0, 'o')
ax2.plot(kappa_list, m1, 'o') 
