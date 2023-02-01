import warnings
import os
import numpy as np
import seaborn as sns

global n_pop, n_neurons, K, m0, folder, model
model = "lif"  # lif, binary, rate
n_pop = 2
n_neurons = 4  # 8
K = 2000  # 4000
folder = "dual_col"
m0 = 0.001
nu0 = 0.05
global T_WINDOW
T_WINDOW = 0.05

global E_frac, n_frac, n_size
if n_pop == 2:
    E_frac = 0.8
    n_frac = [E_frac, np.round(1.0 - E_frac, 2)]
    n_size = [
        int(n_frac[0] * n_neurons * 10000.0),
        int(n_frac[1] * n_neurons * 10000.0),
    ]
else:
    E_frac = 1.0
    n_frac = [E_frac, np.round(1.0 - E_frac, 2)]
    n_size = [int(n_frac[0] * n_neurons * 10000.0)]

global IF_STRUCTURE, IF_SPEC, IF_RING
IF_SPEC = 0
IF_RING = 0
IF_GAUSS = 0

IF_STRUCTURE = IF_SPEC or IF_RING or IF_GAUSS

global SIGMA
SIGMA = [60, 60, 70, 60]

global FIX_SEED, MAP_SEED
FIX_SEED = 1
MAP_SEED = 1

global FIX_CON, CON_SEED
FIX_CON = 0
CON_SEED = 1

global IF_STP
IF_STP = 1
TAU_FAC = 400
TAU_REC = 200
USE = 0.03

global IF_TRIALS, TRIAL_ID, N_TRIALS
IF_TRIALS = 1
TRIAL_ID = 1
N_TRIALS = 10

global IF_INI_COND, INI_COND_ID, N_INI
IF_INI_COND = 0
INI_COND_ID = 5
N_INI = 10

global ext_inputs, J, J2, Tsyn, JEE, JEE2
ext_inputs = []
J = []
J2 = []
Tsyn = []
JEE = 0
JEE2 = 0

global alpha
alpha = 1

global pal
if folder.find("_on") != -1:
    pal = [sns.color_palette("tab10")[1], "b"]
else:
    pal = [
        sns.color_palette("tab10")[0],
        sns.color_palette("colorblind")[0],
        sns.color_palette("colorblind")[1],
        sns.color_palette("colorblind")[3],
    ]

# if IF_DPA:
#     pal = [sns.color_palette('bright')[3]]
# if IF_DUAL:
#     if DISTRACTOR:
#         # pal = [sns.color_palette('bright')[1]]
#         pal = ['k',sns.color_palette('muted')[1]]
#     else:
#         # pal = [sns.color_palette('bright')[0]]
#         pal = [sns.color_palette('muted')[0]]

global label
label = ["E", "I"]

global J0, I0
J0 = -1.0
I0 = 1.0

if n_pop == 1:
    folder = "I0_%.2f_J0_%.2f" % (I0, J0)

global path, con_path, fig_path
path = ""
con_path = ""
fig_path = ""


def read_params():
    if n_pop != 1:
        file_name = " ~/bebopalula/cpp/model/parameters/%dpop/%s.txt" % (n_pop, folder)
        # print("reading parameters from:", file_name)

        i = 0
        with open(
            file_name, "r"
        ) as file:  # Open file for read; needs absolute PATH !!!!
            for line in file:  # Read line-by-line
                line = (
                    line.strip().split()
                )  # Strip the leading/trailing whitespaces and newline
                line.pop(0)
                if i == 0:
                    ext_inputs = np.asarray([float(j) for j in line])
                    ext_inputs *= nu0
                    # print(ext_inputs)
                if i == 1:
                    J = np.asarray([float(j) for j in line])
                    J = J.reshape(n_pop, n_pop)
                    J2 = J * J
                    # print(J)
                if i == 2:
                    Tsyn = np.asarray([float(j) for j in line])
                    Tsyn = Tsyn.reshape(n_pop, n_pop)
                    # print(Tsyn)
                i = i + 1
    else:
        ext_inputs = I0
        J = J0
        J2 = J0 * J0
        Tsyn = 2


if n_pop != 1:
    file_name = "/home/leon/bebopalula/cpp/model/parameters/%dpop/%s.txt" % (
        n_pop,
        folder,
    )
    # print("reading parameters from:", file_name)

    i = 0
    with open(file_name, "r") as file:  # Open file for read
        for line in file:  # Read line-by-line
            line = (
                line.strip().split()
            )  # Strip the leading/trailing whitespaces and newline
            line.pop(0)
            if i == 0:
                ext_inputs = np.asarray([float(j) for j in line])
                ext_inputs *= nu0
                # print(ext_inputs)
            if i == 1:
                J = np.asarray([float(j) for j in line])
                J = J.reshape(n_pop, n_pop)
                J2 = J * J
                # print(J)
            if i == 2:
                Tsyn = np.asarray([float(j) for j in line])
                Tsyn = Tsyn.reshape(n_pop, n_pop)
                # print(Tsyn)
            i = i + 1
else:
    ext_inputs = I0
    J = J0
    J2 = J0 * J0
    Tsyn = 2

path = "/home/leon/bebopalula/cpp/model/simulations/%s/%dpop/%s" % (
    model,
    n_pop,
    folder,
)
fig_path = "/home/leon/bebopalula/python/model/simul/figures/%s/%dpop/%s" % (
    model,
    n_pop,
    folder,
)
con_path = "/home/leon/bebopalula/cpp/model/connectivity/%dpop" % n_pop

if K != np.inf:

    if n_pop == 1:
        path += "/N%d/K%d" % (n_size[0] / 1000, K)
        con_path += "/N%d/K%d" % (n_size[0] / 1000, K)
        fig_path += "/N%d/K%d" % (n_size[0] / 1000, K)
    else:
        path += "/NE_%d_NI_%d/K%d" % (n_size[0] / 1000, n_size[1] / 1000, K)
        fig_path += "/NE_%d_NI_%d/K%d" % (n_size[0] / 1000, n_size[1] / 1000, K)
        con_path += "/NE_%d_NI_%d/K%d" % (n_size[0] / 1000, n_size[1] / 1000, K)

else:
    path += "/N_inf/K_inf"
    con_path += "/N_inf/K_inf"
    fig_path += "/N_inf/K_inf"

if FIX_CON:
    path += "/seed_%d" % (CON_SEED)

if IF_STP:
    path += "/STP/Tf_%d_Tr_%d_U_%.2f" % (TAU_FAC, TAU_REC, USE)

if IF_STRUCTURE:
    if IF_SPEC:
        if RANK == 1:
            path += "/spec/kappa_%.2f" % KAPPA
            fig_path += "/spec/kappa_%.2f" % KAPPA
            con_path += "/spec/kappa_%.2f" % KAPPA
        elif RANK == 2:
            path += "/spec/kappa_%.2f_kappa_1_%.2f" % (KAPPA, KAPPA_1)
            fig_path += "/spec/kappa_%.2f_kappa_1_%.2f" % (KAPPA, KAPPA_1)
            con_path += "/spec/kappa_%.2f_kappa_1_%.2f" % (KAPPA, KAPPA_1)
            if FIX_SEED:
                path += "/seed_%d" % (MAP_SEED)
                fig_path += "/seed_%d" % (MAP_SEED)
                con_path += "/seed_%d" % (MAP_SEED)

    if IF_RING:
        path += "/ring/kappa_%.2f" % KAPPA
        con_path += "/ring/kappa_%.2f" % KAPPA
        fig_path += "/ring/kappa_%.2f" % KAPPA

    if IF_GAUSS:
        path += "/gauss/EE_%d_EI_%d_IE_%d_II_%d" % (
            SIGMA[0],
            SIGMA[1],
            SIGMA[2],
            SIGMA[3],
        )
        con_path += "/gauss/EE_%d_EI_%d_IE_%d_II_%d" % (
            SIGMA[0],
            SIGMA[1],
            SIGMA[2],
            SIGMA[3],
        )
        fig_path += "/gauss/EE_%d_EI_%d_IE_%d_II_%d" % (
            SIGMA[0],
            SIGMA[1],
            SIGMA[2],
            SIGMA[3],
        )

    if IF_TRIALS:
        path += "/trial_%d" % TRIAL_ID
        con_path += "/trial_%d" % TRIAL_ID
        fig_path += "/trial_%d" % TRIAL_ID

    if IF_INI_COND:
        path += "/ini_cond_%d" % INI_COND_ID
        fig_path += "/ini_cond_%d" % INI_COND_ID

    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)
