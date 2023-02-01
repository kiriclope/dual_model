import sys, os, importlib
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import params as gv 
importlib.reload(sys.modules['params']) 

from balance_inputs_dist import inputs_dist, vec_Phi
from utils import *

gv.IF_INI_COND=1
gv.INI_COND_ID=0
gv.init_param()

time, rates = get_time_rates(path=gv.path) 
mean_rates = np.nanmean(rates, axis=0) 

gv.INI_COND_ID=1 
gv.init_param() 

time, rates_1 = get_time_rates(path=gv.path) 
mean_rates_1 = np.nanmean(rates_1, axis=0) 

figtitle = 'rates_ini' + gv.folder 
fig = plt.figure(figtitle, figsize=(1.25*1.618*1.5, 1.618*1.25)) 

plt.scatter(mean_rates, mean_rates_1, alpha=0.2) 
plt.xlabel('Rates i') 
plt.ylabel('Rates j') 
plt.show()
