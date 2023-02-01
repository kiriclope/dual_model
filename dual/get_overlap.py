import importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

import dual.params as gv

importlib.reload(gv)

from common.utils import get_time_rates
import common.progressbar as pgb


def add_vlines():
    plt.axvspan(gv.T_SAMPLE_ON, gv.T_SAMPLE_OFF, alpha=0.1, color="b")
    plt.axvspan(gv.T_DIST_ON, gv.T_DIST_OFF, alpha=0.1, color="b")
    plt.axvspan(gv.T_TEST_ON, gv.T_TEST_OFF, alpha=0.1, color="b")
    plt.axvspan(gv.T_CUE_ON, gv.T_CUE_OFF, alpha=0.1, color="g")


def open_binary(path, file_name, dtype):
    try:
        with open(path + "/" + file_name + ".dat", "rb") as file:
            data = np.fromfile(file, dtype)
    except EOFError:
        pass

    file.close()

    return data


def get_overlap(path=gv.path, ksi_path=gv.ksi_path, MAP=0, IF_SHUFFLE=0):

    ksi_name = "ksi" if MAP == 0 else "ksi_1"
    ksi = open_binary(ksi_path, ksi_name, np.dtype("float32"))

    if IF_SHUFFLE:
        np.random.seed(None)
        rng = np.random.default_rng()
        rng.shuffle(ksi)

    time, rates = get_time_rates(path=path)

    E_rates = rates[:, 0]
    overlap = np.dot(ksi, E_rates.T) / E_rates.shape[-1]

    my_array = np.vstack((time, overlap))
    df = pd.DataFrame(my_array.T, columns=["time", "overlap"])

    return df


def parloop_trials(trial, MAP, path=gv.path, ksi_path=gv.ksi_path):

    # print(path, ksi_path)

    ipath = path
    ipath += "/trial_%d" % trial

    try:
        overlap = get_overlap(ipath, ksi_path, MAP)
        avg_overlap = overlap[-2:]["overlap"].mean()
    except:
        avg_overlap = np.nan

    return avg_overlap


def get_overlap_trials(trial_list, MAP, path=gv.path, ksi_path=gv.ksi_path):

    # print(path, ksi_path)
    with pgb.tqdm_joblib(pgb.tqdm(desc="overlap trial", total=len(trial_list))):
        overlaps = Parallel(n_jobs=-64)(
            delayed(parloop_trials)(trial, MAP, path, ksi_path) for trial in trial_list
        )

    return np.array(overlaps)


def parloop_kappas(kappa, trial_list, MAP, path=gv.path, ksi_path=gv.ksi_path):
    ipath = path.split("/low_rank")[0]
    iseed = path.split("/seed_")[-1]

    ipath += "/low_rank/kappa_%.2f" % kappa
    ipath += "_kappa_1_%.2f" % kappa
    ipath += "/seed_" + iseed

    overlap_trials = get_overlap_trials(trial_list, MAP, ipath, ksi_path)

    return overlap_trials


def get_overlap_kappas(kappa_list, trial_list, MAP, path=gv.path, ksi_path=gv.ksi_path):

    with pgb.tqdm_joblib(pgb.tqdm(desc="overlap kappa", total=len(kappa_list))):
        overlaps = Parallel(n_jobs=-64)(
            delayed(parloop_kappas)(kappa, trial_list, MAP, path, ksi_path)
            for kappa in kappa_list
        )

    return np.array(overlaps)
