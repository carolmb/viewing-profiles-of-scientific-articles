import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from read_file import select_original_breakpoints


def plot_ave_curve(slopes, intervals, filename, color):
    all_curve_descr = []
    for curve in slopes:
        curve_descr = np.zeros(N - 1)
        for i in range(N - 1):
            if curve[i] > curve[i + 1]:
                curve_descr[i] = 1
        all_curve_descr.append(tuple(curve_descr))

    A, B = np.unique(all_curve_descr, return_counts=True, axis=0)
    idxs = np.argsort(B)
    A = A[idxs]
    B = B[idxs]
    total = sum(B)
    for a, b in zip(A, B):
        print("{0}".format(a) + " : {:2f}".format(100 * b / total))

    plt.figure(figsize=(3, 3))

    j = -1
    selected_slopes = []
    selected_intervals = []
    for i, sample in enumerate(all_curve_descr):
        if sample == tuple(A[j]):
            selected_slopes.append(slopes[i])
            selected_intervals.append(intervals[i])

    ave_slope = np.mean(selected_slopes, axis=0)
    print(ave_slope)
    ave_intervals = np.mean(selected_intervals, axis=0)
    ave_tan = np.tan(np.radians(ave_slope))
    x = [0]
    y = [0]
    for s, i in zip(ave_tan, ave_intervals):
        x.append(x[-1] + i)
        y.append(y[-1] + i * s)

    curve_descr = ['+' if a == 0 else '-' for a in A[j]]
    curve_descr = ' '.join(curve_descr)
    print(curve_descr)
    plt.plot(x, y, 'o', ls='-', label='%s (%.2f%%)' % (curve_descr, 100 * B[j]/total), alpha=0.7, c=color)

    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Views')

    plt.tight_layout()
    plt.savefig(filename)


if __name__ == '__main__':

    sources = ['clusters\\clusters\\clusters_ind_single_0.35_3.txt',
               'clusters\\clusters\\clusters_ind_single_0.56_5.txt']
    colors = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'tab:grey']
    letters = ['a', 'b', 'c', 'd', 'e', 'f']
    idx = 0

    # for N, source in zip([3, 5], sources):
    #     labels = np.loadtxt(source, dtype=np.int)
    #     slopes, intervals = select_original_breakpoints(N, 'segm/segmented_curves_filtered.txt')
    #     unique, counts = np.unique(labels, return_counts=True)
    #     unique = unique[counts >= 10]
    #     counts = counts[counts >= 10]
    #     unique_idxs = np.argsort(counts)[-3:]
    #     unique = unique[unique_idxs].tolist()
    #     # labels = [unique.index(l) if l in unique else -1 for l in labels]
    #
    #     for i, label in enumerate(unique):
    #         idxs = labels == label
    #         slopes_i = slopes[idxs]
    #         intervals_i = intervals[idxs]
    #
    #         print(label, '-> tamanho', len(slopes_i))
    #         filename = 'ave_curve_%d_intervals_%s.pdf' % (N, letters[idx])
    #         plot_ave_curve(slopes_i, intervals_i, filename, colors[i])
    #         idx += 1
    #
    for N in [3, 5]:
        slopes, intervals = select_original_breakpoints(N, 'segm/segmented_curves_filtered.txt')

        filename = 'ave_curve_%d_intervals.pdf' % N
        plot_ave_curve(slopes, intervals, filename, 'gray')


# fig, ax = plt.subplots(figsize=(len(A), 10))
#
# im = ax.imshow(np.concatenate((A, B.reshape(-1, 1)), axis=1), cmap=plt.get_cmap("PiYG", 7))
#
# labels = ["$\\alpha_i <= \\alpha_{i+1}$", "$\\alpha_i > \\alpha_{i+1}$"]
# for i in range(len(A)):
#     for j in range(N):
#         if N-1 == j:
#             text = ax.text(j, i, "{:2f}".format(100*B[i]/total),
#                            ha="center", va="center", color="w")
#         else:
#             text = ax.text(j, i, labels[int(A[i, j])],
#                        ha="center", va="center", color="w")
# ax.set_title("Curves")
# ax.xaxis.set_ticklabels([])
# ax.yaxis.set_ticklabels([])
# fig.tight_layout()
# plt.show()
