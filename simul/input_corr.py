import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from statsmodels.tsa.stattools import acf

import params as gv

import progressbar as pgb
from utils import get_time_inputs, get_time_ff_inputs

gv.IF_INI_COND = 0
gv.IF_TRIALS = 0
gv.N_TRIALS = 10
gv.N_INI = 25
gv.IF_CHRISTOS = 0
gv.FIX_CON = 0
gv.CON_SEED = 3

gv.folder = 'christos_off'
gv.init_param()

path = gv.path

def parloop(path, i_trial, i_ini, N_TRIALS=gv.N_TRIALS, A_CUE=gv.A_CUE, EPS_CUE=gv.EPS_CUE, A_DIST=gv.A_DIST, EPS_DIST=gv.EPS_DIST, verbose=0):

    phi_cue = i_trial / N_TRIALS

    path += '/christos'
    path += '/cue_A_%.2f_eps_%.2f_phi_%.3f' % (A_CUE, EPS_CUE, phi_cue)
    path += '/dist_A_%.2f_eps_%.2f_phi_%.3f' % (A_DIST, EPS_DIST, 0.25 + phi_cue)

    path += '/trial_%d' % i_trial ;
    path += '/ini_cond_%d' % i_ini ;

    if(verbose):
        print(path)

    try:
        _, inputs = get_time_inputs(path=path)
        _, ff_inputs = get_time_ff_inputs(path=path)

        if(verbose):
            print(inputs.shape)
    except:
        pass

    return inputs, ff_inputs

# parloop(path, 1, 1, verbose=1)

with pgb.tqdm_joblib(pgb.tqdm(desc='phi off', total= gv.N_INI*gv.N_TRIALS)) as progress_bar:

    inputs_off , ff_off= zip(*Parallel(n_jobs=-64)(delayed(parloop)(path, i_trial, i_ini, A_CUE=.25, EPS_CUE=.25)
                                                   for i_ini in range(1, gv.N_INI+1)
                                                   for i_trial in range(1, gv.N_TRIALS+1))
                      )

ff_inputs = np.array(ff_off)
print(ff_inputs.shape)
ff_inputs = ff_inputs.reshape(25, 10, 161, 2, 32000)
ff_inputs = ff_inputs[..., 0, :]

inputs = np.array(inputs_off)
inputs_off = inputs.reshape(25, 10, 161, 2, 40000)

net_inputs = inputs_off[..., 0, :32000] + inputs_off[..., 1, :32000] + ff_inputs

u_off = np.nanmean(net_inputs, axis=0)  # average over ini
mean_off = np.nanmean(u_off, axis=-1)  # average over neurons

du_off = u_off - mean_off[..., np.newaxis]

# du_off_bl = np.nanmean(du_off[:, 0:int(2/0.05), :], axis=1)
du_off_bl = du_off[:, int(1/0.05), :]

ac_off = []
for i in range(10):
    ac_off.append(acf(du_off_bl[i], nlags=32000))

theta = np.linspace(0, 2*np.pi, 32000)
plt.plot(theta[1:], np.mean(ac_off, 0)[1:])

plt.xlabel('Preferred location (°)')
plt.ylabel('Cross Correlation')
plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],[0, 90, 180, 270, 360])
