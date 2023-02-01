import pandas as pd
import numpy as np
# from nancorrmp.nancorrmp import NaNCorrMp
from statsmodels.tsa.stattools import acf

import params as gv 

gv.init_param()
from utils import *
from get_m1 import *

time, inputs = get_time_inputs(path=gv.path) 
print('time', time.shape, 'inputs', inputs.shape) 
try:
    _, ff_inputs = get_time_ff_inputs(path=gv.path) 
except:
    ff_inputs = np.sqrt(gv.K) * gv.ext_inputs 

E_inputs = np.nanmean( ff_inputs[:int(2.05/0.05), 0, :32000] + inputs[:int(2.05/0.05), 0, :32000] , axis=0) 
I_inputs = np.nanmean( inputs[:int(2.05/0.05), 1, :32000], axis=0)

# E_inputs = ff_inputs[:, 0, :32000] + inputs[:, 0, :32000] 
# I_inputs = inputs[:, 1, :32000]

net_inputs =  E_inputs + I_inputs

smooth = circular_convolution(E_inputs, 3200)
mean_corr = acf(smooth, nlags=32000) 

# net_corr = []
# for i in range(int(2.05/0.05)):
#     net_corr.append( acf(smooth[i], nlags=32000) )
    
# mean_corr = np.nanmean(net_corr, axis=0)

figtitle = 'noise_corr' 
fig = plt.figure(figtitle, figsize=(5.663, 3.5) ) 

plt.plot(mean_corr[3200:], color=gv.pal[0])
plt.xlabel('lag')
plt.ylabel('noise corr.')
