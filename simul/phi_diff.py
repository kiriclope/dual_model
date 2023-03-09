import sys, importlib

import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.stats as stat
from joblib import Parallel, delayed
import seaborn as sns

import progressbar as pgb
import params as gv
from get_m1 import *
from utils import *
from write import *

importlib.reload(sys.modules["params"])
importlib.reload(sys.modules["get_m1"])

gv.IF_INI_COND = 0
gv.IF_TRIALS = 0
gv.N_TRIALS = 10
gv.N_INI = 25
gv.IF_CHRISTOS = 0
gv.FIX_CON = 0
gv.CON_SEED = 3

gv.folder = "christos_off"
gv.init_param()

path = gv.path


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
        print(path)

    try:
        _, rates = get_time_rates(path=path)
        m1, phi = decode_bump(rates[:, 0])

        if verbose:
            print(phi.shape)

        if phi.shape[0] != 161:
            print("error:", path, "wrong shape")
            m1 = np.nan * np.zeros(161)
            phi = np.nan * np.zeros(161)
    except:
        m1 = np.nan * np.zeros(161)
        phi = np.nan * np.zeros(161)
        print("error:", path, "not found")
        pass

    return m1, phi


# CUE .25 EPS .2 works
parloop(path, 1, 1, verbose=1)

with pgb.tqdm_joblib(
    pgb.tqdm(desc="phi off", total=gv.N_INI * gv.N_TRIALS)
) as progress_bar:

    m1_off, phi_off = zip(
        *Parallel(n_jobs=-64)(
            delayed(parloop)(path, i_trial, i_ini, A_CUE=0.25, EPS_CUE=0.25)
            for i_ini in range(1, gv.N_INI + 1)
            for i_trial in range(1, gv.N_TRIALS + 1)
        )
    )

m1_off = np.asarray(m1_off).reshape(gv.N_INI, gv.N_TRIALS, 161)
phi_off = np.asarray(phi_off).reshape(gv.N_INI, gv.N_TRIALS, 161) * 180 / np.pi

print(phi_off.shape)
print(phi_off[0, :, int(3 / gv.T_WINDOW)])

path = path.replace(gv.folder, "christos_on")

with pgb.tqdm_joblib(
    pgb.tqdm(desc="phi on", total=gv.N_INI * gv.N_TRIALS)
) as progress_bar:

    m1_on, phi_on = zip(
        *Parallel(n_jobs=-64)(
            delayed(parloop)(path, i_trial, i_ini, A_CUE=0.25, EPS_CUE=0.25)
            for i_ini in range(1, gv.N_INI + 1)
            for i_trial in range(1, gv.N_TRIALS + 1)
        )
    )

m1_on = np.asarray(m1_on).reshape(gv.N_INI, gv.N_TRIALS, 161)
phi_on = np.asarray(phi_on).reshape(gv.N_INI, gv.N_TRIALS, 161) * 180 / np.pi

print(phi_on.shape)

bins = [int(5.05 / gv.T_WINDOW), int(5.55 / gv.T_WINDOW)]
THRESH = 7.5

pal = [sns.color_palette("tab10")[0], sns.color_palette("tab10")[1]]

# ###########################
# ## Circular rep
# ###########################
phi_cues = np.linspace(0.1, 1, gv.N_TRIALS) * 180

figname = gv.folder + "_on_" + "circular"
plt.figure(figname, figsize=(1.5, 1.5))

m1_off_avg = m1_off[..., bins[0]]
phi_off_avg = phi_off[..., bins[0]]

# m1_off_avg = np.nanmean(m1_off[..., bins[0]:bins[1]], axis=-1)
# phi_off_avg = np.nanmean(phi_off[..., bins[0]:bins[1]], axis=-1)

x_off = np.cos(2 * phi_off_avg / 180 * np.pi)
y_off = np.sin(2 * phi_off_avg / 180 * np.pi)

plt.plot(m1_off_avg * x_off, m1_off_avg * y_off, "x", color=pal[0], ms=1)

m1_on_avg = m1_on[..., bins[0]]
phi_on_avg = phi_on[..., bins[0]]

# m1_on_avg = np.nanmean(m1_on[..., bins[0]:bins[1]], axis=-1)
# phi_on_avg = np.nanmean(phi_on[..., bins[0]:bins[1]], axis=-1)

x_on = np.cos(2 * phi_on_avg / 180 * np.pi)
y_on = np.sin(2 * phi_on_avg / 180 * np.pi)

plt.plot(m1_on_avg * x_on, m1_on_avg * y_on, "x", color=pal[1], ms=1)
plt.axis("off")

x_cues = np.cos(2 * phi_cues / 180 * np.pi)
y_cues = np.sin(2 * phi_cues / 180 * np.pi)

m1_cues = gv.A_CUE * np.sqrt(0.8 * gv.K)

plt.plot(m1_cues * x_cues, m1_cues * y_cues, "k+", ms=1)

plt.savefig(figname + ".svg", dpi=300)


###########################
## DRIFT
###########################
# Dphi_off = np.abs(phi_off - ( 180 - phi_cues[np.newaxis,:,np.newaxis] ))
phi_off_0 = phi_off[..., int(3.05 / gv.T_WINDOW)]
Dphi_off = phi_off - phi_off_0[..., np.newaxis]

Dphi_off[Dphi_off > 90] -= 180
Dphi_off[Dphi_off < -90] += 180

# Dphi_off[np.abs(Dphi_off)>THRESH] = np.nan
Dphi_off = np.abs(Dphi_off)

# drift_off_avg = Dphi_off[..., bins[1]]
# drift_off_avg = np.nanmean(Dphi_off[..., bins[0]:bins[1]], axis=-1)
drift_off_avg = stat.circmean(
    Dphi_off[..., bins[0] : bins[1]], high=90, low=-90, axis=-1, nan_policy="omit"
)
drift_off_stack = np.hstack(drift_off_avg)  # stack trials and ini together

# Dphi_on = np.abs(phi_on - ( 180 - phi_cues[np.newaxis,:,np.newaxis] ) )
phi_on_0 = phi_on[..., int(3.05 / gv.T_WINDOW)]
Dphi_on = phi_on - phi_on_0[..., np.newaxis]

Dphi_on[Dphi_on > 90] -= 180
Dphi_on[Dphi_on < -90] += 180

# Dphi_on[np.abs(Dphi_on)>THRESH] = np.nan
Dphi_on = np.abs(Dphi_on)

# drift_on_avg = Dphi_on[..., bins[1]]
drift_on_avg = stat.circmean(
    Dphi_on[..., bins[0] : bins[1]], high=90, low=-90, axis=-1, nan_policy="omit"
)
drift_on_stack = np.hstack(drift_on_avg)  # stack trials and ini together

figname = gv.folder + "_on_" + "error_hist"

plt.figure(figname, figsize=(2.427, 1.5))
plt.hist(2 * drift_off_stack, histtype="step", color=pal[0], density=1)
plt.hist(2 * drift_on_stack, histtype="step", color=pal[1], density=1)

plt.xlabel("Drift (°)")
plt.ylabel("Density")

plt.savefig(figname + ".svg", dpi=300)

figname = "off_on_" + "drift_bar"
plt.figure(figname, figsize=(2.427, 1.5))
mean = [np.nanmean(2 * drift_off_stack), np.nanmean(2 * drift_on_stack)]
sem = [
    scipy.stats.sem(2 * drift_off_stack, ddof=1, nan_policy="omit"),
    scipy.stats.sem(2 * drift_on_stack, ddof=1, nan_policy="omit"),
]

# mean = [np.nanvar(2*drift_off_stack), np.nanvar(2*drift_on_stack)]

plt.bar([3], mean[0], 1, yerr=sem[0], color=pal[0])
plt.bar([4], mean[1], 1, yerr=sem[1], color=pal[1])

plt.ylabel("Absolute Error")
plt.xticks([3, 4], ["Off", "On"])
plt.xlim([0, 8])
plt.savefig(figname + ".svg", dpi=300)

###########################
## DIFFUSION

phi_off[np.abs(Dphi_off) > THRESH] = np.nan
phi_on[np.abs(Dphi_on) > THRESH] = np.nan

mean_phi_off = stat.circmean(
    phi_off, high=180, low=0, nan_policy="omit", axis=0
)  # average over initial conditions
diff_off = phi_off - mean_phi_off[np.newaxis, :]
print("diff_off", diff_off.shape)

diff_off[diff_off > 90] -= 180
diff_off[diff_off < -90] += 180

diff_off_avg = diff_off[..., bins[1]]
# diff_off_avg = np.nanmean(diff_off[..., bins[0]:bins[1]], axis=-1) # average over time
diff_off_stack = np.hstack(diff_off_avg)  # stack trials and ini together

mean_phi_on = stat.circmean(
    phi_on, high=180, low=0, nan_policy="omit", axis=0
)  # average over initial conditions
diff_on = phi_on - mean_phi_on[np.newaxis, :]
print("diff_on", diff_on.shape)

# diff_on[diff_on>90] -= 180
# diff_on[diff_on<-90] += 180

diff_on_avg = diff_on[..., bins[1]]  # average
# diff_on_avg = np.nanmean(diff_on[..., bins[0]:bins[1]], axis=-1) # average
diff_on_stack = np.hstack(diff_on_avg)  # stack trials and ini together

diff_off_stack = diff_off_stack[np.abs(diff_off_stack) < THRESH]
diff_on_stack = diff_on_stack[np.abs(diff_on_stack) < THRESH]

figname = gv.folder + "_on_" + "diff_hist"
plt.figure(figname, figsize=(2.427, 1.5))

_, bins_off, _ = plt.hist(
    2 * diff_off_stack, histtype="step", color=pal[0], density=1, alpha=0.5, bins="auto"
)
_, bins_on, _ = plt.hist(
    2 * diff_on_stack, histtype="step", color=pal[1], density=1, alpha=0.5, bins="auto"
)

# _, bins_off = np.histogram(2*diff_off_stack, density=1, bins='auto')
# _, bins_on = np.histogram(2*diff_on_stack, density=1, bins='auto')

bins_fit = np.linspace(-2 * THRESH, 2 * THRESH, 100)
mu_off, sigma_off = scipy.stats.norm.fit(2 * diff_off_stack)
fit_off = scipy.stats.norm.pdf(bins_fit, mu_off, sigma_off)
plt.plot(bins_fit, fit_off, color=pal[0])

mu_on, sigma_on = scipy.stats.norm.fit(2 * diff_on_stack)
fit_on = scipy.stats.norm.pdf(bins_fit, mu_on, sigma_on)
plt.plot(bins_fit, fit_on, color=pal[1])

plt.xlabel("Diffusion (°)")
plt.ylabel("Density")
# plt.xlim([-2,2])
plt.savefig(figname + ".svg", dpi=300)


figname = "off_on_" + "diff_bar"
plt.figure(figname, figsize=(2.427, 1.5))

abs_diff_off = np.abs(2 * diff_off_stack)
abs_diff_on = np.abs(2 * diff_on_stack)

mean = [np.nanmean(abs_diff_off), np.nanmean(abs_diff_on)]
sem = [
    scipy.stats.sem(abs_diff_off, ddof=1, nan_policy="omit"),
    scipy.stats.sem(abs_diff_on, ddof=1, nan_policy="omit"),
]

# mean = [np.nanvar(2*diff_off_stack), np.nanvar(2*diff_on_stack)]

plt.bar([3], mean[0], 1, yerr=sem[0], color=pal[0])
plt.bar([4], mean[1], 1, yerr=sem[1], color=pal[1])

plt.ylabel("Diffusion Error (°)")
plt.xticks([3, 4], ["Off", "On"])
plt.xlim([0, 8])
plt.savefig(figname + ".svg", dpi=300)

# def fill_btw(mean, std, color):
#     if mean.shape==0:
#         plt.fill_between(mean, mean-std, mean+std, color=color, alpha=.1)
#     else:
#         for i in range(mean.shape[1]):
#             plt.fill_between(np.linspace(0,8,161), mean[:,i]-std[:,i], mean[:,i]+std[:,i],
#                              color=color, alpha=.1)

# figname = gv.folder + '_on_' + 'phi_time'
# plt.figure(figname, figsize=(5.663, 3.5))

# plt.plot(np.linspace(0,8,161), np.nanmean(phi_off, axis=0).T, 'b')
# plt.plot(np.linspace(0,8,161), np.nanmean(phi_on, axis=0).T, 'r')


# fill_btw( np.nanmean(phi_off, axis=0).T, np.nanstd(phi_off, axis=0).T, color='b')
# fill_btw( np.nanmean(phi_on, axis=0).T, np.nanstd(phi_on, axis=0).T, color='r')

# plt.xlabel('Time (s)')
# plt.ylabel('Phase (°)')
