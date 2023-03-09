import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stat

import params as gv

gv.init_param()

pal = [sns.color_palette("tab10")[0], sns.color_palette("tab10")[1]]


def plot_hist_fit(figname, X, pal, THRESH=30):

    plt.figure(figname)

    _, bins, _ = plt.hist(
        X, histtype="step", color=pal, density=1, alpha=0.5, bins="auto"
    )

    bins_fit = np.linspace(-THRESH, THRESH, 1000)
    mu_, sigma_ = stat.norm.fit(X)
    fit_ = stat.norm.pdf(bins_fit, mu_, sigma_)
    plt.plot(bins_fit, fit_, color=pal)
    plt.ylabel("Density")


def add_pval(pvalue):

    if pvalue < 0.001:
        plt.text(3.25, y + 0.1, "***")
    elif pvalue < 0.01:
        plt.text(3.4, y + 0.1, "**")
    elif pvalue < 0.05:
        plt.text(3.45, y + 0.1, "*")
    else:
        plt.text(3.25, y + 0.2, "n.s.")


def get_drift_diff(
    monkey, condition, task, THRESH=30, CUT_OFF=[0, 45], IF_CORRECT=True
):

    df = get_df(monkey, condition, task, IF_CORRECT)

    drift = []
    diff = []

    theta_end, theta_stim, theta_cue = get_phases(df)
    thetas_out, thetas_in = get_theta_loc(
        theta_end, theta_stim, theta_cue, CUT_OFF, task=task
    )

    for i_stim in range(thetas_in.shape[0]):
        drift.append(get_drift(thetas_out[i_stim], thetas_in[i_stim], THRESH))
        diff.append(
            get_diff(thetas_out[i_stim], thetas_in[i_stim], THRESH, drift=drift[-1])
        )

    drift = np.hstack(drift)
    diff = np.hstack(diff)

    print("drift", drift.shape, "diff", diff.shape)

    return drift, diff


def classToStim(df):

    first_X = [
        12,
        12,
        12,
        12,
        12,
        -12,
        -12,
        -12,
        -12,
        -12,
        12,
        12,
        12,
        12,
        np.nan,
        -12,
        -12,
        -12,
        -12,
        np.nan,
    ]
    first_Y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.nan, 0, 0, 0, 0, np.nan]

    second_X = [
        -12,
        0,
        8.48528137423857,
        12,
        np.nan,
        12,
        0,
        -8.48528137423857,
        -12,
        np.nan,
        -12,
        0,
        8.48528137423857,
        12,
        12,
        12,
        0,
        -8.48528137423857,
        -12,
        -12,
    ]
    second_Y = [
        0,
        12,
        8.48528137423857,
        0,
        np.nan,
        0,
        12,
        8.48528137423857,
        0,
        np.nan,
        0,
        12,
        8.48528137423857,
        0,
        0,
        0,
        12,
        8.48528137423857,
        0,
        0,
    ]

    for i_class in range(20):
        df.loc[df["class"] == i_class + 1, "FirstStiX"] = first_X[i_class]
        df.loc[df["class"] == i_class + 1, "FirstStiY"] = first_Y[i_class]

        df.loc[df["class"] == i_class + 1, "SecondStiX"] = second_X[i_class]
        df.loc[df["class"] == i_class + 1, "SecondStiY"] = second_Y[i_class]

    return df


def carteToPolar(x, y):
    radius = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)

    return radius, theta


def get_theta_loc(theta_end, theta_stim, theta_cue, CUT_OFF=[45, 135], task="first"):

    stim = [0, np.pi]
    # stim = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    cue = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]

    thetas_out = []
    thetas_in = []

    print("get theta at each location")

    for i in range(len(stim)):
        idx_stim = np.where(theta_stim == stim[i])[0].tolist()

        for j in range(len(cue)):
            idx_cue = np.where(theta_cue == cue[j])[0].tolist()

            delta = np.abs(stim[i] - cue[j]) * 180 / np.pi

            # if( delta == CUT_OFF[0] or delta == CUT_OFF[1]):
            if delta >= CUT_OFF[0] and delta <= CUT_OFF[1]:
                idx = list(set(idx_stim) & set(idx_cue))

                print(
                    "stim",
                    stim[i] * 180 / np.pi,
                    "cue",
                    cue[j] * 180 / np.pi,
                    "delta",
                    delta,
                    "idx",
                    len(idx),
                )
                # print('idx_stim', len(idx_stim), 'idx_cue', len(idx_cue), 'idx', len(idx))
                thetas_out.append(theta_end.iloc[idx].to_numpy())

                if task == "first":
                    thetas_in.append(theta_stim.iloc[idx].to_numpy())
                else:
                    thetas_in.append(theta_cue.iloc[idx].to_numpy())

    return np.array(thetas_out), np.array(thetas_in)


def circular_plot(x, y, figname, color="b"):
    plt.figure(figname)
    plt.plot(x, y, "x", ms=5, color=color)
    plt.axis("off")


def correct_sessions(df, task="first", IF_CORRECT=True):
    correct_df = df
    # # print('df', df.shape)

    if IF_CORRECT:
        correct_df = df[df.State_code == 7]
    # print('correct_df', correct_df.shape)

    correct_df = correct_df[correct_df.latency != 0]
    # print('latency', correct_df.shape)
    if task == "first":
        correct_df = correct_df[correct_df["class"] <= 10]
    else:
        correct_df = correct_df[correct_df["class"] > 10]

    # print('correct_first_df', correct_first_df.shape)

    return correct_df


def import_session(df, n_session, on=0, monkey=0):

    if monkey:
        if on:
            if n_session >= 39 and n_session <= 47:
                filename = "HECbeh_odrstim%d" % n_session  # 39-47
            elif n_session >= 18 and n_session <= 29:
                filename = "HECstim0%d_1" % n_session  # 18-29

        else:
            if n_session <= 15 and n_session >= 2:
                filename = "HECbeh_odrstim_%d" % n_session  # 2-15
            elif n_session >= 25 and n_session <= 45:
                filename = "HECbeh_odr%d" % n_session  # 25 to 45
    else:
        if on:
            filename = "'GRUstim0%d_1'" % n_session  # 28-44
        else:
            if n_session <= 10:
                filename = "'GRUbeh_odrstim_%d'" % n_session  # 1-10
            else:
                filename = "'GRUbeh_odr%d'" % (n_session - 2)  # 9-17

    print("filename", filename, np.sum(df.filename == filename))

    if np.sum(df.filename == filename) > 0:
        return df[df.filename == filename]
    else:
        return np.nan


def get_X_Y(df):

    X_end = df.endpointX
    Y_end = df.endpointY

    X_stim = df.FirstStiX
    Y_stim = df.FirstStiY

    X_cue = df.SecondStiX
    Y_cue = df.SecondStiY

    return X_end, Y_end, X_stim, Y_stim, X_cue, Y_cue


def get_phases(df):

    X_end = df.endpointX
    Y_end = df.endpointY

    X_stim = df.FirstStiX
    Y_stim = df.FirstStiY

    X_cue = df.SecondStiX
    Y_cue = df.SecondStiY

    _, theta_end = carteToPolar(X_end, Y_end)
    # print('theta', theta_end.shape)

    _, theta_stim = carteToPolar(X_stim, Y_stim)
    # print('theta_stim', theta_end.shape)

    _, theta_cue = carteToPolar(X_cue, Y_cue)

    return theta_end, theta_stim, theta_cue


def get_drift(theta_out, theta_in, THRESH=30):

    drift = theta_out - theta_in

    # print(drift*180/np.pi)

    drift[drift > np.pi] -= 2 * np.pi
    drift[drift < -np.pi] += 2 * np.pi
    drift *= 180 / np.pi

    drift = drift[np.abs(drift) < THRESH]
    # drift = np.abs(drift)

    print(
        "DRIFT: stim/cue",
        np.nanmean(theta_in) * 180 / np.pi,
        "mean drift",
        np.nanmean(drift),
    )

    return drift


def get_diff(theta_out, theta_in, THRESH=30, drift=None):

    # average over trials
    mean_theta = stat.circmean(
        theta_out, nan_policy="omit", axis=0, high=np.pi, low=-np.pi
    )

    diff = theta_out - mean_theta
    # diff = stat.circvar(theta_out, nan_policy="omit", axis=0, high=np.pi, low=-np.pi)

    diff[diff >= np.pi] -= 2 * np.pi
    diff[diff <= -np.pi] += 2 * np.pi

    diff *= 180 / np.pi

    print(
        "DIFF: stim/cue",
        np.nanmean(theta_in) * 180 / np.pi,
        "mean_theta",
        mean_theta * 180 / np.pi,
        "mean diff",
        np.nanmean(diff),
    )

    # if drift is not None:
    # diff = diff[drift < THRESH]
    diff = diff[np.abs(diff) < THRESH]

    # diff = np.abs(diff)
    return diff


def get_df(monkey, condition, task, IF_CORRECT=True):

    if condition == "on":
        if monkey == 1:
            df = pd.read_excel(
                "./data/StimulationEyeEndPoint2.xlsx",
                engine="openpyxl",
                sheet_name="Inf_HecStim",
            )
            df = classToStim(df)
        else:
            df = pd.read_excel(
                "./data/StimulationEyeEndPoint2.xlsx",
                engine="openpyxl",
                sheet_name="Inf_GruStim",
            )
    else:
        if monkey == 1:
            df = pd.read_excel(
                "./data/StimulationEyeEndPoint2.xlsx",
                engine="openpyxl",
                sheet_name="Inf_HecNoStim",
            )
            df = classToStim(df)
        else:
            df = pd.read_excel(
                "./data/StimulationEyeEndPoint2.xlsx",
                engine="openpyxl",
                sheet_name="Inf_GruNoStim",
            )

    df["filename"] = df["filename"].astype("string")
    df_correct = correct_sessions(df, task=task, IF_CORRECT=IF_CORRECT)

    return df_correct


if __name__ == "__main__":

    THRESH = 30
    IF_CORRECT = True
    CUT_OFF = [45, 45]

    task = "first"

    condition = "off"
    monkey = 0
    drift_0, diff_0 = get_drift_diff(monkey, condition, task, THRESH, CUT_OFF)
    monkey = 1
    drift_1, diff_1 = get_drift_diff(monkey, condition, task, THRESH, CUT_OFF)

    drift_off = np.hstack((drift_0, drift_1))
    diff_off = np.hstack((diff_0, diff_1))

    condition = "on"
    monkey = 0
    drift_0, diff_0 = get_drift_diff(monkey, condition, task, THRESH, CUT_OFF)
    monkey = 1
    drift_1, diff_1 = get_drift_diff(monkey, condition, task, THRESH, CUT_OFF)

    drift_on = np.hstack((drift_0, drift_1))
    diff_on = np.hstack((diff_0, diff_1))

    # Drift
    figname = task + "_exp_" + "error_hist"
    plot_hist_fit(figname, drift_off, pal[0], THRESH=THRESH)
    plot_hist_fit(figname, drift_on, pal[1], THRESH=THRESH)
    plt.xlabel("Systematic Bias (°)")
    plt.savefig(figname + ".svg", dpi=300)

    figname = task + "_exp_" + "drift_bar"

    plt.figure(figname)

    mean_off = np.sqrt(np.nanmean(drift_off**2))  #
    # mean_off = np.nanmean(np.abs(drift_off))
    sem_off = stat.sem(drift_off, ddof=1, nan_policy="omit")

    mean_on = np.sqrt(np.nanmean(drift_on**2))
    # mean_on = np.nanmean(np.abs(drift_on))
    sem_on = stat.sem(drift_on, ddof=1, nan_policy="omit")

    plt.bar([3], mean_off, 1, yerr=sem_off, color=pal[0])
    plt.bar([4], mean_on, 1, yerr=sem_on, color=pal[1])

    plt.ylabel("Systematic Error (°)")
    plt.xticks([3, 4], ["Off", "On"])
    plt.xlim([0, 8])

    y = np.max([mean_off + sem_off + 1, mean_on + sem_on + 1])
    plt.hlines(y, 3, 4)

    stats = stat.ttest_ind(drift_off, drift_on, nan_policy="omit", equal_var=False)
    add_pval(stats.pvalue)
    print("pval", stats.pvalue)

    plt.savefig(figname + ".svg", dpi=300)

    figname = task + "_exp_" + "diff_hist"
    plot_hist_fit(figname, diff_off, pal[0], THRESH=THRESH)
    plot_hist_fit(figname, diff_on, pal[1], THRESH=THRESH)
    plt.xlabel("Angular Deviation (°)")
    plt.savefig(figname + ".svg", dpi=300)

    figname = task + "_exp_" + "diff_bar"
    plt.figure(figname)

    mean_off = np.nanmean(np.abs(diff_off))
    var_off = np.nanvar(diff_off)
    sem_off = stat.sem(np.abs(diff_off), ddof=1, nan_policy="omit")

    mean_on = np.nanmean(np.abs(diff_on))
    var_on = np.nanvar(diff_on)
    sem_on = stat.sem(np.abs(diff_on), ddof=1, nan_policy="omit")

    plt.bar([3], mean_off, 1, yerr=sem_off, color=pal[0])
    plt.bar([4], mean_on, 1, yerr=sem_on, color=pal[1])

    # plt.bar([3] , var_off, 1, yerr=sem_off, color=pal[0])
    # plt.bar([4] , var_on, 1, yerr=sem_on, color=pal[1])

    plt.ylabel("Angular Error (°)")
    plt.xticks([3, 4], ["Off", "On"])
    plt.xlim([0, 8])

    y = np.max([mean_off + sem_off + 1, mean_on + sem_on + 1])
    plt.hlines(y, 3, 4)

    stats = stat.ttest_ind(
        np.abs(diff_off), np.abs(diff_on), nan_policy="omit", equal_var=False
    )

    add_pval(stats.pvalue)
    print("pval", stats.pvalue)
    plt.savefig(figname + ".svg", dpi=300)
