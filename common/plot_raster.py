import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import params as gv

if __name__ == '__main__':

    raw_spike_times = pd.read_csv(gv.path + '/spike_times.dat', sep='\s+').to_numpy()
    print('data', raw_spike_times.shape)

    neurons_id = raw_spike_times[:,0]
    print('neurons_id', neurons_id.shape, neurons_id[500:505])

    spike_times = raw_spike_times[:,1]/1000
    print('spike_times', spike_times.shape, spike_times[500:505])

    figname = 'raster_' + gv.folder
    fig = plt.figure(figname)

    excitatory_idx = np.where(neurons_id<gv.n_size[0])
    inhibitory_idx = np.where(neurons_id>=gv.n_size[0])

    plt.scatter(spike_times[excitatory_idx],
                neurons_id[excitatory_idx],
                marker='|',
                alpha=0.25, color='r')
    plt.scatter(spike_times[inhibitory_idx],
                neurons_id[inhibitory_idx],
                marker='|',
                alpha=0.25, color='b')

    plt.xlabel('Time (s)')
    plt.ylabel('Neuron #')
    plt.show()
