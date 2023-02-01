import numpy as np
import matplotlib.pyplot as plt

from common import params as gv
from common.utils import get_time_inputs, get_time_ff_inputs

if __name__ == '__main__':

    gv.init_param()

    time, inputs = get_time_inputs(path=gv.path)
    print('time', time.shape, 'inputs', inputs.shape)

    try:
        _, ff_inputs = get_time_ff_inputs(path=gv.path)
    except:
        ff_inputs = np.sqrt(gv.K) * gv.ext_inputs

    figname = 'inputs'
    fig = plt.figure(figname)

    bins = [int(5/.05), -1]

    for i_pop in range(gv.n_pop):
        ax = fig.add_subplot(int('22%d' % (i_pop+1)))

        # averaged over neurons
        E_inputs = np.nanmean(ff_inputs[bins[0]:bins[1], 0, :gv.n_size[0]], axis=1)
        E_inputs += np.nanmean(inputs[bins[0]:bins[1], 0, :gv.n_size[0]], axis=1)
        I_inputs = np.nanmean(inputs[bins[0]:bins[1], 1, :gv.n_size[0]], axis=1)

        # neuron = np.random.randint(gv.n_size[i_pop])

        # E_inputs = ff_inputs[:, 0, neuron] + inputs[:, 0, neuron]
        # I_inputs = inputs[:, 1, neuron]

        net_inputs =  E_inputs + I_inputs

        plt.plot(time[bins[0]:bins[1]], E_inputs, 'r', lw=1)
        plt.plot(time[bins[0]:bins[1]], net_inputs, 'k', lw=1)
        plt.plot(time[bins[0]:bins[1]], I_inputs, 'b', lw=1)

        plt.xlabel('Time (ms)')
        plt.ylabel('Inputs (mA)')

        ax = fig.add_subplot(int('22%d' %(i_pop+3)))

        # averaged over time
        E_inputs = np.nanmean(ff_inputs[bins[0]:bins[1], 0, :gv.n_size[0]], axis=0)
        E_inputs += np.nanmean(inputs[bins[0]:bins[1], 0, :gv.n_size[0]], axis=0)
        I_inputs = np.nanmean(inputs[bins[0]:bins[1], 1, :gv.n_size[0]], axis=0)

        net_inputs =  E_inputs + I_inputs

        plt.hist(E_inputs , color='r', histtype='step')
        plt.hist(I_inputs , color='b', histtype='step')
        plt.hist(net_inputs , color='k', histtype='step')

        # if i_pop==0:
        #     plt.hist( mf_inputs_E, color='k', ls='--', histtype='step')
        # else:
        #     plt.hist( mf_inputs_I, color='k', ls='--', histtype='step')

        plt.xlabel('Inputs (mA)')
        plt.ylabel('Count')

    plt.show()
