import os
from common.params import *

global KAPPA
KAPPA = 0.25

global KAPPA_EXT, PHI_EXT, PHI_DIST
KAPPA_EXT = 3.0
PHI_EXT = 0.25

global A_CUE, EPS_CUE, A_DIST, EPS_DIST, PHI_CUE, PHI_DIST
PHI_CUE = 0.250
PHI_DIST = 1 - PHI_CUE

A_CUE = 0.250
EPS_CUE = 0.250

A_DIST = 0.000
EPS_DIST = 0.250

global T_SAMPLE_ON, T_SAMPLE_OFF
T_SAMPLE_ON = 2
T_SAMPLE_OFF = 3

global T_DIST_ON, T_DIST_OFF
T_DIST_ON = 4.5 # 4.5
T_DIST_OFF = 5.5 # 5.5

global T_ERASE_ON, T_ERASE_OFF
T_ERASE_ON = 5 # 4.5
T_ERASE_OFF = 6 # 5.5

path += '/christos'
path += '/cue_A_%.2f_eps_%.2f_phi_%.3f' % (A_CUE, EPS_CUE, PHI_CUE)
path += '/dist_A_%.2f_eps_%.2f_phi_%.3f' % (A_DIST, EPS_DIST, PHI_DIST)

fig_path += '/christos'
fig_path += '/cue_A_%.2f_eps_%.2f_phi_%.3f' % (A_CUE, EPS_CUE, PHI_CUE)
fig_path += '/dist_A_%.2f_eps_%.2f_phi_%.3f' % (A_DIST, EPS_DIST, PHI_DIST)

if not os.path.isdir(fig_path):
    os.makedirs(fig_path)
