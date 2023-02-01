import sys, importlib

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat 
from get_m1 import * 
from utils import * 
from write import *

importlib.reload(sys.modules['params'])
importlib.reload(sys.modules['get_m1']) 

gv.IF_INI_COND = 0
gv.IF_TRIALS = 0

gv.N_TRIALS = 100
gv.init_param()

path = gv.path

def get_diffusion(path):
    phi_trial = []
    for i_trial in range(1, gv.N_TRIALS + 1):
        gv.path = path
        gv.path += '/trial_%d' % i_trial ; 
        print(gv.path)
        # if 0==0:
        try:
            time, rates = get_time_rates(path=gv.path) 
            _, phi = decode_bump(rates[:,0]) 
            # phi_ini.append( phi[0] - (1.0 - i_trial/gv.N_TRIALS) * np.pi )                
            
            print('phi', phi.shape)
            if(phi.shape[0]!=160):
                phi = np.nan*np.zeros(160) 
            Dphi = ( phi - (gv.PHI_DIST) * np.pi )
            
            phi_trial.append(phi) 
            print('phi', phi[int(3/gv.T_WINDOW)] * 180 / np.pi,
                  'phi_dist', gv.PHI_DIST * 180,
                  'Dphi', Dphi[int(3/gv.T_WINDOW)] * 180 / np.pi) 
        except:
            phi_trial.append(np.nan*np.zeros(160)) 
            print('error') 
            pass
                
    phi_trial = np.asarray(phi_trial) 
    print('phi_trial', phi_trial.shape) 
    
    return phi_trial * 180 / np.pi 

phi_off = get_diffusion(path)
path = path.replace('off', 'on') # change dirname 
phi_on = get_diffusion(path) 

Dphi_off = phi_off - gv.PHI_DIST * 180 
# Dphi_off = phi_off - (1- gv.PHI_DIST) * 180
Dphi_off[Dphi_off>90] -= 180 
Dphi_off[Dphi_off<-90] += 180 

Dphi_off[Dphi_off>15/2] -= np.nan
Dphi_off[Dphi_off<-15/2] += np.nan

bins =  [int(4/gv.T_WINDOW), int(5/gv.T_WINDOW)] 
drift_off_avg = stat.circmean(Dphi_off[..., bins[0]:bins[1]], high=90, low=-90, axis=-1, nan_policy='omit') 
# drift_off_avg = stat.circmean(Dphi_off[..., 25:29], high=90, low=-90, axis=-1, nan_policy='omit') 

Dphi_on = phi_on - (gv.PHI_DIST)* 180 
# Dphi_on = phi_on - (1-gv.PHI_DIST)* 180
Dphi_on[Dphi_on>90] -= 180 
Dphi_on[Dphi_on<-90] += 180 

Dphi_on[Dphi_on>15/2] -= np.nan
Dphi_on[Dphi_on<-15/2] += np.nan

drift_on_avg = stat.circmean(Dphi_on[..., bins[0]:bins[1]], high=90, low=-90, axis=-1, nan_policy='omit') 
# drift_on_avg = stat.circmean(Dphi_on[..., 25:29], high=90, low=-90, axis=-1, nan_policy='omit') 

figname = gv.folder + '_on_' + 'error_hist'
# figname = 'first_off_on_' + 'error_hist'
# figname = 'sec_off_on_' + 'error_hist'

plt.figure(figname)
plt.hist(2*drift_off_avg, histtype='step', color='b') 
plt.hist(2*drift_on_avg, histtype='step', color='r') 
plt.xlabel('Angular Deviation (deg)') 
plt.ylabel('Count')

plt.savefig(figname + '.svg', dpi=300)
# plt.boxplot([drift_off, drift_on], patch_artist=True, labels=['off', 'on'], showfliers=False, notch=True)
# plt.ylabel('Error (deg)')
