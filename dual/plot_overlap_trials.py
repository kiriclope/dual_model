import numpy as np
import matplotlib.pyplot as plt

import dual.params as gv
from dual.get_overlap import get_overlap, add_vlines

if __name__ == '__main__':

    overlap_ini = []
    overlap_1_ini = []

    for i_ini in range(1, 10+1):

        path = gv.path
        path += '/ini_cond_%d' % i_ini

        print(path)
        time, overlap = get_overlap(path=path, MAP=0)
        _, overlap_1 = get_overlap(path=path, MAP=1)

        overlap_ini.append(overlap)
        overlap_1_ini.append(overlap_1)

    overlap_ini = np.array(overlap_ini)
    overlap_1_ini = np.array(overlap_1_ini)

    print('overlap_ini', overlap_ini.shape)

    mean_overlap = np.mean(overlap_ini, axis=0)[0]
    mean_overlap_1 = np.mean(overlap_1_ini, axis=0)[0]

    std_overlap = np.std(overlap_ini, axis=0)[0]
    std_overlap_1 = np.std(overlap_1_ini, axis=0)[0]

    figtitle = 'rates_overlap_time'
    fig = plt.figure(figtitle, figsize=(5.663 * gv.RANK, 3.5))
    ax = fig.add_subplot(1, gv.RANK, 1)

    plt.plot(time, mean_overlap, '-', color=gv.pal[0], alpha=gv.alpha)
    # plt.errorbar(time, mean_overlap, yerr=std_overlap, color=gv.pal[0], alpha=gv.alpha )
    plt.fill_between(time, mean_overlap-std_overlap, mean_overlap+std_overlap, color=gv.pal[0], alpha=0.25)

    plt.xlabel('Time (s)')
    plt.ylabel('Sample Overlap')
    add_vlines()

    ax = fig.add_subplot(1,gv.RANK,2)
    plt.plot(time, mean_overlap_1, '-',  color=gv.pal[0], alpha=gv.alpha)
    # plt.errorbar(time, mean_overlap_1, yerr=std_overlap_1, color=gv.pal[0], alpha=gv.alpha )
    plt.fill_between(time, mean_overlap_1-std_overlap_1, mean_overlap_1+std_overlap_1, color=gv.pal[0], alpha=0.25)
    plt.xlabel('Time (s)')
    plt.ylabel('Distractor Overlap')
    if gv.IF_ADD_VLINES:
        add_vlines()
    plt.xticks([0, 2, 4, 6, 8, 10, 12])
    plt.xlim([0, 12])

    THRESHOLD = 0.1
    errors = overlap_ini[..., 0, int(2/0.05):int(9/0.05)] < THRESHOLD
    score = np.mean(~np.any(errors, axis=1))
    print(score)
