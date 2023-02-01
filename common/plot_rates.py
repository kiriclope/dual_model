import numpy as np
import matplotlib.pyplot as plt

from common import params as gv
from common.utils import get_time_rates

if __name__ == '__main__':

    time, rates = get_time_rates(path=gv.path)
    mean_rates = np.nanmean(rates, axis=-1)

    print('rates', np.nanmean(mean_rates, axis=0))

    figname = 'rates_' + gv.folder
    fig = plt.figure(figname)

    ax = fig.add_subplot(int('121'))
    for i_pop in range(gv.n_pop):
        plt.plot(time, mean_rates[:,i_pop], color=gv.pal[i_pop])
    plt.xlabel('Time (ms)')
    plt.ylabel('Rates (Hz)')

    ax = fig.add_subplot(int('122'))

    avg_rates = np.nanmean(rates, axis=0)
    for i_pop in range(gv.n_pop):
        plt.hist(avg_rates[i_pop], histtype='step')
    plt.xlabel('Rates (Hz)')
    plt.ylabel('Count')

    plt.show()
