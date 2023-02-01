import sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import params as gv

def read_args(list):
    d=defaultdict(list)
    for k, v in ((k.lstrip('-'), v) for k,v in (a.split('=') for a in sys.argv[1:])):
        d[k].append(v)

    return d

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = -1):

    pad_size = int(target_length)

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    pad_array = np.pad( np.array(array, dtype=float), pad_width=npad, mode='constant', constant_values=np.nan)

    return pad_array

def add_vlines():

    plt.axvspan(gv.T_SAMPLE_ON, gv.T_SAMPLE_OFF, alpha=0.1, color='b')
    plt.axvspan(gv.T_DIST_ON, gv.T_DIST_OFF, alpha=0.1, color='b')
    plt.axvspan(gv.T_TEST_ON, gv.T_TEST_OFF, alpha=0.1, color='b')
    plt.axvspan(gv.T_CUE_ON, gv.T_CUE_OFF, alpha=0.1, color='g')
    # plt.axvspan(gv.T_ERASE_ON, gv.T_ERASE_OFF, alpha=0.1, color='g')

def add_vlines_axis(axis):

    axis.axvspan(gv.T_SAMPLE_ON, gv.T_SAMPLE_OFF, alpha=0.1, color='b')
    axis.axvspan(gv.T_DIST_ON, gv.T_DIST_OFF, alpha=0.1, color='b')
    axis.axvspan(gv.T_TEST_ON, gv.T_TEST_OFF, alpha=0.1, color='b')
    axis.axvspan(gv.T_CUE_ON, gv.T_CUE_OFF, alpha=0.1, color='g')
    # axis.axvspan(gv.T_ERASE_ON, gv.T_ERASE_OFF, alpha=0.1, color='g')

def get_time_rates(n_pop=gv.n_pop, n_size=gv.n_size, MAP=0, path=gv.path, con_path=gv.con_path):

    filter_rates = pd.read_csv(path + '/filter_rates.dat', sep='\s+', header=None).to_numpy()
    time = filter_rates[:,0] / 1000
    rates = np.delete(filter_rates, [0], axis=1)

    if(MAP):
        idx_perm = get_idx_perm(con_path)
        rates = rates[:, idx_perm]

    if n_pop!=1:
        rates = pad_along_axis(rates, n_size[0]-n_size[1])
        n_neurons = int(rates.shape[1]/2)
        rates = np.reshape(rates, (rates.shape[0], 2, n_neurons))

    else:
        n_neurons = rates.shape[1]
        rates = np.reshape(rates, (rates.shape[0], 1, n_neurons))

    return time, rates


def get_time_ff_inputs(n_pop=gv.n_pop, n_size=gv.n_size, MAP=0, path=gv.path, con_path=gv.con_path):

    filter_ff_inputs = pd.read_csv(path + '/ff_inputs.dat', sep='\s+', header=None).to_numpy()
    time = filter_ff_inputs[:,0] / 1000
    ff_inputs = np.delete(filter_ff_inputs, [0], axis=1)

    if(MAP):
        idx_perm = get_idx_perm(con_path)
        ff_inputs = ff_inputs[:, idx_perm]

    if n_pop!=1:
        ff_inputs = pad_along_axis(ff_inputs, n_size[0]-n_size[1])
        n_neurons = int(ff_inputs.shape[1]/2)
        ff_inputs = np.reshape(ff_inputs, (ff_inputs.shape[0], 2, n_neurons))

    else:
        n_neurons = ff_inputs.shape[1]
        ff_inputs = np.reshape(ff_inputs, (ff_inputs.shape[0], 1, n_neurons))

    return time, ff_inputs

def get_time_inputs(n_pop=gv.n_pop, n_size=gv.n_size, MAP=0, path=gv.path, con_path=gv.con_path):

    filter_inputs = pd.read_csv(path + '/inputs.dat', sep='\s+', header=None).to_numpy()
    time = filter_inputs[:,0] / 1000
    net_inputs = np.delete(filter_inputs, [0], axis=1)

    # print('raw net inputs', net_inputs.shape, 'n_size', n_size)

    # if(MAP):
    #     idx_perm = get_idx_perm(con_path)
    #     net_inputs = net_inputs[:, idx_perm]

    if n_pop!=1:
        n_neurons = int(net_inputs.shape[1]/2)
        net_inputs = np.reshape(net_inputs, (net_inputs.shape[0], 2, n_neurons))
    else:
        n_neurons = net_inputs.shape[1]
        net_inputs = np.reshape(net_inputs, (net_inputs.shape[0], 1, n_neurons))

    # print('net inputs', net_inputs.shape, 'n_neurons', n_neurons)

    return time, net_inputs

def open_binary(path, file_name, dtype):
    try:
        with open(path + '/' + file_name + '.dat', 'rb') as file:
            data = np.fromfile(file, dtype)
    except EOFError:
        pass

    file.close()

    return data

def get_idx_perm(path=gv.con_path):
    dtype = np.dtype("L")

    try:
        with open(path + '/idx_perm.dat', 'rb') as f:
            idx_perm = np.fromfile(f, dtype)
        f.close()

    except EOFError:
        pass

    # print('idx_perm', idx_perm.shape, idx_perm[:10], idx_perm[-10:])

    return idx_perm

def get_overlap(rates, n_size=gv.n_size, ksi_path=gv.ksi_path, MAP=gv.MAP, IF_SHUFFLE=0):

    overlap = np.zeros( (rates.shape[1], rates.shape[0]) ) * np.nan

    if(MAP==0):
        ksi = open_binary(ksi_path, 'ksi', np.dtype("float32") )
        # print('ksi', ksi.shape)
    else:
        ksi = open_binary(ksi_path, 'ksi_1', np.dtype("float32") )
        # print('ksi 1', ksi.shape)
    if IF_SHUFFLE:
        np.random.seed(None)
        rng = np.random.default_rng()
        rng.shuffle(ksi)

    for i_pop in range(rates.shape[1]) :
        pop_rates = rates[:, i_pop, : n_size[i_pop]]
        if(IF_SHUFFLE):
            np.random.shuffle(ksi)

        if i_pop==0:
            ksi_pop = ksi[:n_size[0]]
        else:
            ksi_pop = np.zeros(n_size[1])

        dum= np.dot(ksi_pop, pop_rates.T) / pop_rates.shape[-1]
        overlap[i_pop] = dum

    return overlap
