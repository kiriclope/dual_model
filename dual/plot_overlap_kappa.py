import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

import dual.params as gv
from dual.get_overlap import get_overlap
import common.progressbar as pgb


def overlap_signed(x, sign=1, th=0.15):
    x_signed = x.copy()

    for kappa in range(x.shape[0]):
        for trial in range(x.shape[1]):

            if x_signed[kappa, trial] > -th and sign == -1:
                x_signed[kappa, trial] = np.nan
            if x_signed[kappa, trial] <= th and sign == 1:
                x_signed[kappa, trial] = np.nan
            elif np.abs(x_signed[kappa, trial]) > th and sign == 0:
                x_signed[kappa, trial] = np.nan

    return np.nanmean(x_signed, axis=-1)


def set_plot():
    figname = "rates_overlap_kappa_" + gv.folder
    fig, axis = plt.subplots(
        1, gv.RANK, figsize=(0.75 * 2.427 * (gv.RANK), 0.75 * 1.5), num=figname
    )

    # axis[0].set(ylim=(0, np.ceil(np.nanmax(rates_list) * 12) / 10))
    # axis[0].set_xlabel("$var(\\xi)$")
    # axis[0].set_ylabel("Rates (Hz)")

    axis[0].set_xlabel("$var(\\xi^{Sample})$")
    axis[0].set_ylabel("Sample Overlap (a.u.)")

    if gv.RANK == 2:
        axis[1].set_xlabel("$var(\\xi^{Dist})$")
        axis[1].set_ylabel("Dist. Overlap (a.u.)")

    return fig, axis


def plot_func(kappa, overlap, axis):

    overlap_plus = overlap_signed(overlap)
    overlap_minus = overlap_signed(overlap, sign=-1)
    overlap_zero = overlap_signed(overlap, sign=0)

    axis.plot(kappa, overlap, "o", color=gv.pal[0], alpha=1)

    axis.plot(kappa, overlap_plus, "-o", color=gv.pal[0], alpha=1)
    axis.plot(kappa, overlap_minus, "-o", color=gv.pal[0], alpha=1)
    axis.plot(kappa, overlap_zero, "-o", color=gv.pal[0], alpha=1)


def parloop(
    kappa,
    trial,
    path=gv.path,
    ksi_path=gv.ksi_path,
    RANK=gv.RANK,
    ksi_seed=gv.SEED_KSI,
    verbose=0,
):

    ksi_path = ksi_path.split("/low_rank")[0]
    path = path.split("/low_rank")[0]

    ksi_path += "/low_rank/rank_%d/seed_ksi_%d" % (RANK, ksi_seed)

    path += "/low_rank/kappa_%.2f" % kappa
    if RANK == 2:
        path += "_kappa_1_%.2f" % kappa

    path += "/seed_%d" % ksi_seed
    path += "/trial_%d" % trial

    if verbose:
        print(path)
        print(ksi_path)

    # if 0 == 0:
    try:
        _, overlap = get_overlap(path=path, ksi_path=ksi_path, MAP=0)
        overlap = np.nanmean(overlap[-2:], axis=-1)
    except:
        overlap = np.nan
        print("error ")

    if RANK == 2:
        try:
            _, overlap_1 = get_overlap(path=path, ksi_path=ksi_path, MAP=1)
            overlap_1 = np.nanmean(overlap_1[-2:], axis=-1)
        except:
            overlap_1 = np.nan
            print("error 1")

    if verbose:
        print("overlap", overlap)

    return overlap, overlap_1


if __name__ == "__main__":
    kappa_list = np.arange(4.7, 4.8, 0.1)
    trial_list = np.arange(1, 101, 1)

    print("kappa", kappa_list)
    print("trials", trial_list)

    parloop(kappa_list[0], trial_list[0], verbose=1)

    with pgb.tqdm_joblib(
        pgb.tqdm(
            desc="computing rates and overlap",
            total=int(kappa_list.shape[0] * trial_list.shape[0]),
        )
    ) as progress_bar:
        overlap_list, overlap_1_list = zip(
            *Parallel(n_jobs=-64)(
                delayed(parloop)(kappa, trial)
                for kappa in kappa_list
                for trial in trial_list
            )
        )

    # rates_list = np.array(rates_list).T
    overlap_list = np.array(overlap_list).T
    overlap_1_list = np.array(overlap_1_list).T

    print(overlap_list.shape)
    print(overlap_1_list.shape)

    # rates_list = rates_list.reshape(gv.n_pop, kappa_list.shape[0], trial_list.shape[0])
    overlap_list = overlap_list.reshape(kappa_list.shape[0], trial_list.shape[0])
    overlap_1_list = overlap_1_list.reshape(kappa_list.shape[0], trial_list.shape[0])

    print(overlap_list.shape)
    print(overlap_1_list.shape)

    print(
        # "rates",
        # rates_list.shape,
        "overlap",
        overlap_list.shape,
        "overlap_1",
        overlap_1_list.shape,
    )
    # print(rates_list)

    # mean_rates = np.nanmean(rates_list, axis=-1)
    # mean_overlap = np.nanmean(overlap_list, axis=-1)
    # mean_overlap_1 = np.nanmean(overlap_1_list, axis=-1)

    # std_rates = np.nanstd(rates_list, axis=-1)
    # std_overlap = np.nanstd(overlap_list, axis=-1)
    # std_overlap_1 = np.nanstd(overlap_list, axis=-1)

    # print("rates", mean_rates)
    # print("overlap", mean_overlap)

    fig, axis = set_plot()

    # for i_pop in range(gv.n_pop - 1):
    #     axis[0].plot(kappa_list, mean_rates[i_pop], "-o", color=gv.pal[i_pop])
    #     axis[0].plot(
    #         kappa_list, rates_list[i_pop, :], "x", color=gv.pal[i_pop], alpha=0.2
    #     )
    #     axis[0].fill_between(
    #         kappa_list,
    #         mean_rates[i_pop] - std_rates[i_pop],
    #         mean_rates[i_pop] + std_rates[i_pop],
    #         alpha=0.1,
    #         color=gv.pal[i_pop],
    #     )

    plot_func(kappa_list, overlap_list, axis[0])
    plot_func(kappa_list, overlap_1_list, axis[1])

    plt.show()
