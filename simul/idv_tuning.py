import sys
from importlib import reload

import numpy as np

# import matplotlib.pyplot as plt

import params as gv
from utils import get_time_rates
from get_m1 import decode_bump

reload(sys.modules["params"])


rates_off = []
N_TRIALS = 10

gv.folder = "christos_off"

gv.IF_CHRISTOS = 0
gv.IF_INI_COND = 0
gv.IF_TRIALS = 0

gv.init_param()
print(gv.path)

for i_trial in range(1, N_TRIALS + 1):

    path = gv.path

    phi_cue = i_trial / N_TRIALS

    path += "/christos"
    path += "/cue_A_%.2f_eps_%.2f_phi_%.3f" % (gv.A_CUE, gv.EPS_CUE, phi_cue)
    path += "/dist_A_%.2f_eps_%.2f_phi_%.3f" % (gv.A_DIST, gv.EPS_DIST, 0.25 + phi_cue)

    path += "/trial_%d" % i_trial
    path += "/ini_cond_%d" % 1

    time, rates = get_time_rates(path=path)
    rates_off.append(rates)

rates_off = np.asarray(rates_off)
rates_off = np.swapaxes(rates_off, 0, -1)

rates_off = np.mean(rates_off[:, int(3 / 0.05) : int(5 / 0.05), 0], 1)
print(rates_off.shape)

m0_off = np.mean(rates_off, -1)
m1_off, _ = decode_bump(rates_off)

gv.folder = "christos_on"
gv.init_param()
print(gv.path)

rates_on = []
for i_trial in range(1, N_TRIALS + 1):

    path = gv.path

    phi_cue = i_trial / N_TRIALS

    path += "/christos"
    path += "/cue_A_%.2f_eps_%.2f_phi_%.3f" % (gv.A_CUE, gv.EPS_CUE, phi_cue)
    path += "/dist_A_%.2f_eps_%.2f_phi_%.3f" % (gv.A_DIST, gv.EPS_DIST, 0.25 + phi_cue)

    path += "/trial_%d" % i_trial
    path += "/ini_cond_%d" % 1

    time, rates = get_time_rates(path=path)
    rates_on.append(rates)

rates_on = np.asarray(rates_on)
rates_on = np.swapaxes(rates_on, 0, -1)

rates_on = np.mean(rates_on[:, int(3 / 0.05) : int(5 / 0.05), 0], 1)
print(rates_on.shape)

m0_on = np.mean(rates_on, -1)
m1_on, _ = decode_bump(rates_on)
