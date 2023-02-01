import importlib
import numpy as np
import matplotlib.pyplot as plt

import dual.params as gv

importlib.reload(gv)
from dual.get_overlap import get_overlap_trials


def carteToPolar(x, y):
    radius = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)

    return radius, theta * 180 / np.pi


def plot_kappa_plane(path=gv.path, ksi_path=gv.ksi_path, n_trials=100):

    print(path)
    print(ksi_path)
    trial_list = np.arange(1, n_trials + 1, 1)
    overlap = get_overlap_trials(trial_list, MAP=0, path=path, ksi_path=ksi_path)
    overlap_1 = get_overlap_trials(trial_list, MAP=1, path=path, ksi_path=ksi_path)

    plt.figure("overlaps_plane_" + gv.folder)
    plt.plot(overlap, overlap_1, "o")
    plt.xlabel("Sample Overlap")
    plt.ylabel("Dist. Overlap")

    plot_phase_dist(overlap, overlap_1)


def plot_phase_dist(overlap, overlap_1):

    radius, theta = carteToPolar(overlap, overlap_1)

    plt.figure("overlaps_phases_" + gv.folder)
    plt.hist(theta % 180, histtype="step", density=1, bins="auto")
    plt.xlim([0, 180])
    plt.xticks([0, 45, 90, 135, 180])

    plt.xlabel("Overlaps Pref. Dir. (°)")
    plt.ylabel("Density")


#
