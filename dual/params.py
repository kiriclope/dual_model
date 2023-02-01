import os
import importlib
import common.params

importlib.reload(common.params)

from common.params import *

global T_SAMPLE_ON, T_SAMPLE_OFF
T_SAMPLE_ON = 2
T_SAMPLE_OFF = 3

global T_DIST_ON, T_DIST_OFF
T_DIST_ON = 4.5  # 4.5
T_DIST_OFF = 5.5  # 5.5

global T_CUE_ON, T_CUE_OFF
T_CUE_ON = 6.5
T_CUE_OFF = 7.5

global T_TEST_ON, T_TEST_OFF
T_TEST_ON = 9
T_TEST_OFF = 10

global RANK, MAP
RANK = 2
MAP = 0

global KAPPA, KAPPA_1
KAPPA = 4.7
KAPPA_1 = KAPPA  # 8 # 12
global KAPPA_EXT
KAPPA_EXT = 3.0

global SEED_KSI
SEED_KSI = 2

global TASK, SAMPLE, DISTRACTOR
TASK = None  # 'DPA', DualGo, DualNoGo
SAMPLE = 0
DISTRACTOR = 0

path += "/low_rank/kappa_%.2f" % KAPPA
fig_path += "/low_rank/kappa_%.2f" % KAPPA
con_path += "/low_rank/kappa_%.2f" % KAPPA

if RANK == 2:
    path += "_kappa_1_%.2f" % KAPPA_1
    fig_path += "_kappa_1_%.2f" % KAPPA_1
    con_path += "_kappa_1_%.2f" % KAPPA_1

path += "/seed_%d" % SEED_KSI

ksi_path = "/home/leon/bebopalula/cpp/model/connectivity/%dpop" % n_pop
ksi_path += "/NE_%d_NI_%d" % (n_size[0] / 1000, n_size[1] / 1000)
ksi_path += "/low_rank/rank_%d/seed_ksi_%d" % (RANK, SEED_KSI)

if TASK is not None:
    path += "/%s/kappa_%.2f" % (TASK, KAPPA_EXT)
    fig_path += "/%s/kappa_%.2f" % (TASK, KAPPA_EXT)

    if SAMPLE:
        path += "/sample_B"
    else:
        path += "/sample_A"

    if DISTRACTOR:
        path += "/NoGo"
    else:
        path += "/Go"


global IF_TRIALS, TRIAL_ID, N_TRIALS
IF_TRIALS = 0
TRIAL_ID = 2
N_TRIALS = 10

global IF_INI_COND, INI_COND_ID, N_INI
IF_INI_COND = 0
INI_COND_ID = 5
N_INI = 10

if IF_TRIALS:
    path += "/trial_%d" % TRIAL_ID
    con_path += "/trial_%d" % TRIAL_ID
    fig_path += "/trial_%d" % TRIAL_ID

if IF_INI_COND:
    path += "/ini_cond_%d" % INI_COND_ID
    fig_path += "/ini_cond_%d" % INI_COND_ID

global IF_ADD_VLINES
IF_ADD_VLINES = 1

if not os.path.isdir(fig_path):
    os.makedirs(fig_path)
