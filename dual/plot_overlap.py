import numpy as np
import matplotlib.pyplot as plt

from dual.params import *
from dual.get_overlap import get_overlap, add_vlines


if __name__ == "__main__":

    figtitle = "rates_overlap_time_"
    fig = plt.figure(figtitle, figsize=(2.427 * 0.75 * RANK, 1.5 * 0.75))
    ax = fig.add_subplot(1, RANK, 1)

    time, overlap = get_overlap(path=path, MAP=0)
    plt.plot(time, overlap, "-", color=pal[0], alpha=alpha)
    plt.xlabel("Time (s)")
    plt.ylabel("Sample Overlap")

    if IF_ADD_VLINES:
        add_vlines()

    # plt.xticks([0, 2, 4, 6, 8, 10, 12])
    # plt.xlim([0, 12])

    # print('overlap', overlap.shape)

    if RANK == 2:
        time, overlap_1 = get_overlap(path=path, MAP=1)
        ax = fig.add_subplot(1, RANK, 2)
        plt.plot(time, overlap_1, "-", color=pal[0], alpha=alpha)
        plt.xlabel("Time (s)")
        plt.ylabel("Distractor Overlap")

        if IF_ADD_VLINES:
            add_vlines()

        # plt.xticks([0, 2, 4, 6, 8, 10, 12])
        # plt.xlim([0, 12])

    plt.show()

    # plt.savefig('overlap_Go_first.svg', dpi=300, format='svg')
