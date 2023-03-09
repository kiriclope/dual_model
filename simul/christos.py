import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stat

import params as gv

gv.init_param()

pal = [sns.color_palette("tab10")[0], sns.color_palette("tab10")[1]]


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


def correct_sessions(df, task="first"):
    correct_df = df
    # # print('df', df.shape)

    # correct_df = df[df.State_code == 7]
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

    # drift = drift[np.abs(drift) < THRESH]
    drift = np.abs(drift)

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

    if drift is not None:
        diff = diff[drift < THRESH]
    # diff = np.abs(diff)
    return diff


if __name__ == "__main__":

    IF_SESSION = 0
    task = "first"
    monkey = 2
    THRESH = 30
    CUT_OFF = [0, 45]

    if monkey == 2:
        df = pd.read_excel(
            "./data/StimulationEyeEndPoint2.xlsx",
            engine="openpyxl",
            sheet_name="Inf_HecNoStim",
        )
        df = classToStim(df)
        df2 = pd.read_excel(
            "./data/StimulationEyeEndPoint2.xlsx",
            engine="openpyxl",
            sheet_name="Inf_GruNoStim",
        )
        df = df.append(df2)
    elif monkey == 1:
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

    df_correct = correct_sessions(df, task=task)

    X_end, Y_end, X_stim, Y_stim, X_cue, Y_cue = get_X_Y(df_correct)
    figname = "monkey_" + str(monkey) + "_" + task + "_off_on_" + "circular"
    circular_plot(X_end, Y_end, figname, pal[0])
    circular_plot(X_stim, Y_stim, figname, "k")
    circular_plot(X_cue, Y_cue, figname, "k")

    if IF_SESSION:

        drift_session = []
        diff_session = []

        for n_session in range(1, 50):
            drift_loc = []
            diff_loc = []

            try:
                df_session = import_session(df_correct, n_session, on=0, monkey=monkey)

                theta_end, theta_stim, theta_cue = get_phases(df_session)
                thetas_out, thetas_in = get_theta_loc(
                    theta_end, theta_stim, theta_cue, CUT_OFF, task=task
                )

                for i_stim in range(thetas_in.shape[0]):
                    drift_loc.append(
                        np.nanmean(
                            get_drift(thetas_out[i_stim], thetas_in[i_stim], THRESH)
                        )
                    )
                    diff_loc.append(
                        np.nanmean(
                            get_diff(thetas_out[i_stim], thetas_in[i_stim], THRESH)
                        )
                    )

                drift = np.nanmean(drift_loc)
                diff = np.nanmean(diff_loc)
            except:
                drift = np.nan
                diff = np.nan

            drift_session.append(drift)
            diff_session.append(diff)

        drift_off = np.hstack(drift_session)
        diff_off = np.hstack(diff_session)

    else:

        drift_off = []
        diff_off = []

        theta_end, theta_stim, theta_cue = get_phases(df_correct)
        thetas_out, thetas_in = get_theta_loc(
            theta_end, theta_stim, theta_cue, CUT_OFF, task=task
        )

        for i_stim in range(thetas_in.shape[0]):
            drift_off.append(get_drift(thetas_out[i_stim], thetas_in[i_stim], THRESH))
            diff_off.append(
                get_diff(
                    thetas_out[i_stim], thetas_in[i_stim], THRESH, drift=drift_off[-1]
                )
            )

        drift_off = np.hstack(drift_off)
        diff_off = np.hstack(diff_off)

    print("drift_off", drift_off.shape, "diff_off", diff_off.shape)

    # on condition
    if monkey == 2:
        df = pd.read_excel(
            "./data/StimulationEyeEndPoint2.xlsx",
            engine="openpyxl",
            sheet_name="Inf_HecStim",
        )
        df = classToStim(df)
        df2 = pd.read_excel(
            "./data/StimulationEyeEndPoint2.xlsx",
            engine="openpyxl",
            sheet_name="Inf_GruStim",
        )
        df = df.append(df2)
    elif monkey == 1:
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
    # df = pd.read_excel('./data/StimulationEyeEndPointFilenameRawData.xlsx',
    #                    engine='openpyxl', sheet_name='Inf_gruStim')

    df["filename"] = df["filename"].astype("string")

    df_correct = correct_sessions(df, task=task)

    drift_session = []
    diff_session = []

    X_end, Y_end, X_stim, Y_stim, X_cue, Y_cue = get_X_Y(df_correct)

    circular_plot(X_end, Y_end, figname, pal[1])
    circular_plot(X_stim, Y_stim, figname, "k")
    circular_plot(X_cue, Y_cue, figname, "k")
    plt.savefig(figname + ".svg", dpi=300)

    if IF_SESSION:

        drift_session = []
        diff_session = []

        for n_session in range(1, 50):
            drift_loc = []
            diff_loc = []

            try:
                df_session = import_session(df_correct, n_session, on=1, monkey=monkey)
                theta_end, theta_stim, theta_cue = get_phases(df_session)
                thetas_out, thetas_in = get_theta_loc(
                    theta_end, theta_stim, theta_cue, CUT_OFF, task=task
                )

                for i_stim in range(thetas_in.shape[0]):
                    drift_loc.append(
                        np.nanmean(
                            get_drift(thetas_out[i_stim], thetas_in[i_stim], THRESH)
                        )
                    )
                    diff_loc.append(
                        np.nanmean(
                            get_diff(thetas_out[i_stim], thetas_in[i_stim], THRESH)
                        )
                    )

                drift = np.nanmean(drift_loc)
                diff = np.nanmean(diff_loc)
            except:
                drift = np.nan
                diff = np.nan

            drift_session.append(drift)
            diff_session.append(diff)

        drift_on = np.hstack(drift_session)
        diff_on = np.hstack(diff_session)

    else:
        drift_on = []
        diff_on = []

        theta_end, theta_stim, theta_cue = get_phases(df_correct)
        thetas_out, thetas_in = get_theta_loc(
            theta_end, theta_stim, theta_cue, CUT_OFF, task=task
        )

        for i_stim in range(thetas_in.shape[0]):
            drift_on.append(get_drift(thetas_out[i_stim], thetas_in[i_stim], THRESH))
            diff_on.append(
                get_diff(
                    thetas_out[i_stim], thetas_in[i_stim], THRESH, drift=drift_on[-1]
                )
            )

        drift_on = np.hstack(drift_on)
        diff_on = np.hstack(diff_on)

    print("drift_on", drift_on.shape, "diff_on", diff_on.shape)

    figname = "monkey_" + str(monkey) + "_" + task + "_exp_" + "error_hist"
    plt.figure(figname)
    plt.hist(drift_off, histtype="step", color=pal[0], density=1, bins="auto")
    plt.hist(drift_on, histtype="step", color=pal[1], density=1, bins="auto")
    plt.xlabel("Absolute Error")
    plt.ylabel("Density")
    plt.savefig(figname + ".svg", dpi=300)

    figname = "monkey_" + str(monkey) + "_" + task + "_exp_" + "drift_bar"
    plt.figure(figname)

    # mean_off = np.sqrt( np.nanmean(drift_off**2) )
    mean_off = np.nanmean(np.abs(drift_off))
    var_off = np.nanvar(drift_off)
    sem_off = stat.sem(drift_off, ddof=1, nan_policy="omit")

    # mean_on = np.sqrt( np.nanmean(drift_on**2) )
    mean_on = np.nanmean(np.abs(drift_on))
    var_on = np.nanvar(drift_on)
    sem_on = stat.sem(drift_on, ddof=1, nan_policy="omit")

    plt.bar([3], mean_off, 1, yerr=sem_off, color=pal[0])
    plt.bar([4], mean_on, 1, yerr=sem_on, color=pal[1])

    # plt.bar([3] , var_off, 1, yerr=sem_off, color=pal[0])
    # plt.bar([4] , var_on, 1, yerr=sem_on, color=pal[1])

    plt.ylabel("Absolute Error")
    plt.xticks([3, 4], ["Off", "On"])
    plt.xlim([0, 8])

    y = np.max([mean_off + sem_off + 1, mean_on + sem_on + 1])
    plt.hlines(y, 3, 4)

    stats = stat.ttest_ind(drift_off, drift_on, nan_policy="omit", equal_var=False)
    print("pval", stats.pvalue)

    if stats.pvalue < 0.001:
        plt.text(3.25, y + 0.1, "***")
    elif stats.pvalue < 0.01:
        plt.text(3.4, y + 0.1, "**")
    elif stats.pvalue < 0.05:
        plt.text(3.45, y + 0.1, "*")
    else:
        plt.text(3.25, y + 0.2, "n.s.")

    plt.savefig(figname + ".svg", dpi=300)

    figname = "monkey_" + str(monkey) + "_" + task + "_exp_" + "diff_hist"
    plt.figure(figname)

    _, bins, _ = plt.hist(
        diff_off, histtype="step", color=pal[0], density=1, alpha=0.5, bins="auto"
    )

    _, bins, _ = plt.hist(
        diff_on, histtype="step", color=pal[1], density=1, alpha=0.5, bins="auto"
    )

    bins_fit = np.linspace(-THRESH, THRESH, 1000)
    mu_off, sigma_off = stat.norm.fit(diff_off)
    fit_off = stat.norm.pdf(bins_fit, mu_off, sigma_off)
    plt.plot(bins_fit, fit_off, color=pal[0])

    mu_on, sigma_on = stat.norm.fit(diff_on)
    fit_on = stat.norm.pdf(bins_fit, mu_on, sigma_on)
    plt.plot(bins_fit, fit_on, color=pal[1])

    plt.xlabel("Angular Deviation (°)")
    plt.ylabel("Density")
    plt.savefig(figname + ".svg", dpi=300)

    figname = "monkey_" + str(monkey) + "_" + task + "_exp_" + "diff_bar"
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

    plt.ylabel("Angular Error")
    plt.xticks([3, 4], ["Off", "On"])
    plt.xlim([0, 8])

    y = np.max([mean_off + sem_off + 1, mean_on + sem_on + 1])
    plt.hlines(y, 3, 4)

    # stats = stat.ttest_ind(var_off, var_on, nan_policy='omit', equal_var=False)
    stats = stat.ttest_ind(
        np.abs(diff_off), np.abs(diff_on), nan_policy="omit", equal_var=False
    )
    print("pval", stats.pvalue)

    # if stats.pvalue < .001 :
    #     plt.text(3.25, y+.1, '***', fontsize=14)
    # elif stats.pvalue < .01 :
    # plt.text(3.4, y+.1, '**', fontsize=14)
    # elif stats.pvalue < .05 :
    plt.text(3.45, y + 0.1, "*")
    # else:
    #     plt.text(3.25, y+.2, 'n.s.')

    plt.savefig(figname + ".svg", dpi=300)

    # if task=='first':
    #     data_off = np.array([drift_off, np.zeros(drift_off.shape) , np.zeros(drift_off.shape) , np.zeros(drift_off.shape) ]).T
    #     df_off = pd.DataFrame(data = data_off, columns=['error', 'stim_state', 'error_type', 'task'])

    #     df_all = df_off

    #     data_off = np.array([np.abs(diff_off), np.zeros(drift_off.shape) , np.ones(drift_off.shape) , np.zeros(drift_off.shape) ]).T
    #     df_off = pd.DataFrame(data = data_off, columns=['error', 'stim_state', 'error_type', 'task'])

    #     df_all = df_all.append(df_off)

    #     data_on = np.array([drift_on, np.ones(drift_on.shape) , np.zeros(drift_on.shape) , np.zeros(drift_on.shape) ]).T
    #     df_on = pd.DataFrame(data = data_on, columns=['error', 'stim_state', 'error_type', 'task'])

    #     df_all = df_all.append(df_on)

    #     data_on = np.array([np.abs(diff_on), np.ones(drift_on.shape) , np.ones(drift_on.shape) , np.zeros(drift_on.shape) ]).T
    #     df_on = pd.DataFrame(data = data_on, columns=['error', 'stim_state', 'error_type', 'task'])

    #     df_all = df_all.append(df_on)

    # else:

    #     data_off = np.array([drift_off, np.zeros(drift_off.shape) , np.zeros(drift_off.shape) , np.ones(drift_off.shape) ]).T
    #     df_off = pd.DataFrame(data = data_off, columns=['error', 'stim_state', 'error_type', 'task'])

    #     df_all_2 = df_off

    #     data_off = np.array([np.abs(diff_off), np.zeros(drift_off.shape) , np.ones(drift_off.shape) , np.ones(drift_off.shape) ]).T
    #     df_off = pd.DataFrame(data = data_off, columns=['error', 'stim_state', 'error_type', 'task'])

    #     df_all_2 = df_all_2.append(df_off)

    #     data_on = np.array([drift_on, np.ones(drift_on.shape) , np.zeros(drift_on.shape) , np.ones(drift_on.shape) ]).T
    #     df_on = pd.DataFrame(data = data_on, columns=['error', 'stim_state', 'error_type', 'task'])

    #     df_all_2 = df_all_2.append(df_on)

    #     data_on = np.array([np.abs(diff_on), np.ones(drift_on.shape) , np.ones(drift_on.shape) , np.ones(drift_on.shape) ]).T
    #     df_on = pd.DataFrame(data = data_on, columns=['error', 'stim_state', 'error_type', 'task'])

    #     df_all_2 = df_all_2.append(df_on)

    # # results = smf.ols('error ~ stim_state * error_type * task', data=df).fit()

    # figname = 'glm' + '_off_on_'
    # plt.figure(figname, figsize=(5.663, 3.5))
    # ax = df_res["coef"][1:].plot(kind = "bar", legend = False, yerr=df_res["std err"][1:]) # no error bars added here
    # # X
    # ax.set_ylabel("a.u.")
    # # Y
    # ax.set_xlabel("")
    # ax.set_xticklabels(["$\\beta_{stim}$", "$\\beta_{error}$", "$\\beta_{stim*error}$",
    #                     "$\\beta_{task}$", "$\\beta_{stim*task}$", "$\\beta_{error*task}$",
    #                     "$\\beta_{stim*error*task}$"], rotation=45)
