from sys import modules
from importlib import reload

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from scipy.stats import circmean

import progressbar as pgb
import params as gv

from utils import get_time_rates
from get_m1 import decode_bump

reload(modules["params"])
reload(modules["get_m1"])

gv.IF_INI_COND = 0
gv.IF_TRIALS = 0
gv.N_TRIALS = 10
gv.N_INI = 25
gv.IF_CHRISTOS = 0
gv.FIX_CON = 0
gv.CON_SEED = 3


def get_diff(phase, THRESH=30):

    mean_phase = circmean(phase, high=2 * np.pi, low=0, nan_policy="omit", axis=0)
    # average over initial conditions
    diff = phase - mean_phase[np.newaxis, :]
    diff *= 180 / np.pi

    diff[diff > 90] -= 180
    diff[diff < -90] += 180

    diff[np.abs(diff) > THRESH] = np.nan

    return diff


def parloop(
    path,
    i_trial,
    i_ini,
    N_TRIALS=gv.N_TRIALS,
    A_CUE=gv.A_CUE,
    EPS_CUE=gv.EPS_CUE,
    A_DIST=gv.A_DIST,
    EPS_DIST=gv.EPS_DIST,
    verbose=0,
):

    phi_cue = i_trial / N_TRIALS

    path += "/christos"
    path += "/cue_A_%.2f_eps_%.2f_phi_%.3f" % (A_CUE, EPS_CUE, phi_cue)
    path += "/dist_A_%.2f_eps_%.2f_phi_%.3f" % (A_DIST, EPS_DIST, 0.25 + phi_cue)

    path += "/trial_%d" % i_trial
    path += "/ini_cond_%d" % i_ini

    if verbose:
        print("path", path)

    try:
        _, rates = get_time_rates(path=path)
    except:
        rates = np.nan * np.zeros((161, 32000))
        pass

    if verbose:
        print("rates", rates.shape)

    return rates[:, 0]  # only excitatory


def get_rates(path, n_ini, n_trials):

    with pgb.tqdm_joblib(
        pgb.tqdm(desc="rates", total=n_ini * n_trials)
    ) as progress_bar:

        rates = Parallel(n_jobs=-64)(
            delayed(parloop)(path, i_trial, i_ini, A_CUE=0.25, EPS_CUE=0.25)
            for i_ini in range(1, n_ini + 1)
            for i_trial in range(1, n_trials + 1)
        )

    rates = np.asarray(rates).reshape(gv.N_INI, gv.N_TRIALS, 161, 32000)
    # print(rates_off.shape)

    return rates


rng = np.random.default_rng()

gv.folder = "christos_off"
gv.init_param()
path = gv.path

parloop(path, 1, 1, verbose=1)

rates_off = get_rates(path, gv.N_INI, gv.N_TRIALS)
rates_perm = rates_off.copy()
# rates_perm = rng.permuted(rates_off, axis=0)
_, phi_off = decode_bump(rates_perm)
diff_off = get_diff(2.0 * phi_off)

gv.folder = "christos_on"
gv.init_param()
path = gv.path

rates_on = get_rates(path, gv.N_INI, gv.N_TRIALS)
rates_perm = rates_on.copy()
# rates_perm = rng.permuted(rates_on, axis=0)
_, phi_on = decode_bump(rates_perm)
diff_on = get_diff(2.0 * phi_on)

bins = [int(5.05 / gv.T_WINDOW), int(5.55 / gv.T_WINDOW)]

diff_off_avg = diff_off[..., bins[1]]  # average
diff_off_stack = np.hstack(diff_off_avg)  # stack trials and ini together

diff_on_avg = diff_on[..., bins[1]]  # average
diff_on_stack = np.hstack(diff_on_avg)  # stack trials and ini together

pal = [sns.color_palette("tab10")[0], sns.color_palette("tab10")[1]]

figname = gv.folder + "_on_" + "diff_hist"
plt.figure(figname, figsize=(2.427, 1.5))

_, bins_off, _ = plt.hist(
    diff_off_stack, histtype="step", color=pal[0], density=1, alpha=0.5, bins="auto"
)

_, bins_on, _ = plt.hist(
    diff_on_stack, histtype="step", color=pal[1], density=1, alpha=0.5, bins="auto"
)

plt.xlabel("Diffusion (°)")
plt.ylabel("Density")
# plt.xlim([-2,2])
plt.savefig(figname + ".svg", dpi=300)
