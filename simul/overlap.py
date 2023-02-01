import sys, os, importlib
from importlib import reload

import numpy as np
import matplotlib.pyplot as plt

import params as gv
importlib.reload(sys.modules['params'])
from utils import *

gv.init_param()

ksi = open_binary(gv.con_path, 'ksi', np.dtype("double") )    

plt.hist(ksi) 
plt.xlabel('ksi') 
plt.ylabel('Count') 
