import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from joblib import Parallel, delayed

import common.progressbar as pgb
import common.params as gv


def parloop_CV(spike_times, neurons_id, i_neuron):

    try:
        idx_neuron = np.where(neurons_id==i_neuron)
        ISI = spike_times[idx_neuron][1:] - spike_times[idx_neuron][:-1]
        CV = np.std(ISI) / np.mean(ISI)
    except:
        CV = np.nan

    return CV


def parloop_CV2(spike_times, neurons_id, i_neuron):

    try:
        idx_neuron = np.where(neurons_id==i_neuron)
        ISI = spike_times[idx_neuron][1:] - spike_times[idx_neuron][:-1]

        CV2 = 2.0 * np.mean( np.absolute(ISI[1:] - ISI[:-1]) ) / np.mean( ISI[1:] + ISI[:-1] )

    except:
        CV2 = np.nan

    return CV2


if __name__ == '__main__':

    raw_spike_times = pd.read_csv(gv.path + '/spike_times.dat', sep='\s+').to_numpy()
    print('data', raw_spike_times.shape)

    neurons_id = raw_spike_times[:,0]
    print('neurons_id', neurons_id.shape, neurons_id[500:505])

    spike_times = raw_spike_times[:,1]
    print('spike_times', spike_times.shape, spike_times[500:505])

    idx = np.logical_and(spike_times>0, spike_times<2000)
    # idx = np.logical_and(spike_times>3000, spike_times<5000)
    neurons_id = neurons_id[idx] # time in ms

    # CV = np.zeros(gv.n_pop*gv.n_size)
    # for i_neuron in range(0, gv.n_pop*gv.n_size):

    #     idx_neuron = np.where(neurons_id==i_neuron)
    #     ISI = spike_times[idx_neuron][1:] - spike_times[idx_neuron][:-1]
    #     CV[i_neuron] = np.std(ISI) / np.mean(ISI)

    n_neurons = gv.n_neurons * 10000

    with pgb.tqdm_joblib( pgb.tqdm(desc='CV', total=n_neurons) ) as progress_bar:
        CV = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(parloop_CV)(spike_times, neurons_id, i_neuron) for i_neuron in range(n_neurons) )

    with pgb.tqdm_joblib( pgb.tqdm(desc='CV2', total=n_neurons) ) as progress_bar:
        CV2 = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(parloop_CV2)(spike_times, neurons_id, i_neuron) for i_neuron in range(n_neurons) )

    figtitle = 'spike_statistics'
    fig = plt.figure(figtitle, figsize=(5.663*2, 3.5))

    ax = fig.add_subplot(int('121'))
    plt.hist(CV[:gv.n_size[0]], histtype='step', ls='-', color=gv.pal[0],  lw=2)
    # plt.hist(CV[gv.n_size[0]:], histtype='step', ls='-', color='b')

    plt.xlabel('CV')
    plt.ylabel('Count')

    ax = fig.add_subplot(int('122'))
    plt.hist(CV2[:gv.n_size[0]], histtype='step', ls='-', color=gv.pal[0], lw=2)
    # plt.hist(CV2[gv.n_size[0]:], histtype='step', ls='-', color='b')

    plt.xlabel('CV$_2$')
    plt.ylabel('Count')

    plt.show()
