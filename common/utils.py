import numpy as np
import pandas as pd
import common.params as gv


def pad_along_axis(array: np.ndarray, target_length: int, axis: int = -1):

    pad_size = int(target_length)

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    pad_array = np.pad(
        np.array(array, dtype=float),
        pad_width=npad,
        mode="constant",
        constant_values=np.nan,
    )

    return pad_array


def get_time_rates(n_pop=gv.n_pop, n_size=gv.n_size, path=gv.path):

    filter_rates = pd.read_csv(
        path + "/filter_rates.dat", sep="\s+", header=None
    ).to_numpy()
    time = filter_rates[:, 0] / 1000
    rates = np.delete(filter_rates, [0], axis=1)

    if n_pop != 1:
        rates = pad_along_axis(rates, n_size[0] - n_size[1])
        n_neurons = int(rates.shape[1] / 2)
        rates = np.reshape(rates, (rates.shape[0], 2, n_neurons))
    else:
        n_neurons = rates.shape[1]
        rates = np.reshape(rates, (rates.shape[0], 1, n_neurons))

    return time, rates


def get_time_ff_inputs(n_pop=gv.n_pop, n_size=gv.n_size, path=gv.path):

    filter_ff_inputs = pd.read_csv(
        path + "/ff_inputs.dat", sep="\s+", header=None
    ).to_numpy()
    time = filter_ff_inputs[:, 0] / 1000
    ff_inputs = np.delete(filter_ff_inputs, [0], axis=1)

    if n_pop != 1:
        ff_inputs = pad_along_axis(ff_inputs, n_size[0] - n_size[1])
        n_neurons = int(ff_inputs.shape[1] / 2)
        ff_inputs = np.reshape(ff_inputs, (ff_inputs.shape[0], 2, n_neurons))
    else:
        n_neurons = ff_inputs.shape[1]
        ff_inputs = np.reshape(ff_inputs, (ff_inputs.shape[0], 1, n_neurons))

    return time, ff_inputs


def get_time_inputs(
    n_pop=gv.n_pop, n_size=gv.n_size, path=gv.path, con_path=gv.con_path
):

    filter_inputs = pd.read_csv(path + "/inputs.dat", sep="\s+", header=None).to_numpy()
    time = filter_inputs[:, 0] / 1000
    net_inputs = np.delete(filter_inputs, [0], axis=1)

    print("raw net inputs", net_inputs.shape, "n_size", n_size)

    if n_pop != 1:
        n_neurons = int(net_inputs.shape[1] / 2)
        net_inputs = np.reshape(net_inputs, (net_inputs.shape[0], 2, n_neurons))
    else:
        n_neurons = net_inputs.shape[1]
        net_inputs = np.reshape(net_inputs, (net_inputs.shape[0], 1, n_neurons))

    print("net inputs", net_inputs.shape, "n_neurons", n_neurons)

    return time, net_inputs


def open_binary(path, file_name, dtype):
    try:
        with open(path + "/" + file_name + ".dat", "rb") as file:
            data = np.fromfile(file, dtype)
    except EOFError:
        pass

    file.close()

    return data
