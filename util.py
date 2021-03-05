import sys
import getopt
import numpy as np
from read_file import select_original_breakpoints


def read_preprocessed_file(N, source, labels_source):
    # original = np.loadtxt(source)
    # slopes = original[:, :N]
    # intervals = original[:, N:-1]
    # labels = original[:, -1]
    # labels = labels.astype(int)
    # total_labels = len(set(labels))
    slopes, intervals = select_original_breakpoints(N, source)

    labels = np.loadtxt(labels_source, dtype=np.int)
    unique, counts = np.unique(labels, return_counts=True)
    unique = unique[counts >= 10]
    counts = counts[counts >= 10]
    unique_idxs = np.argsort(counts)[-3:]
    unique = unique[unique_idxs].tolist()
    labels = [unique.index(l) if l in unique else -1 for l in labels]
    total_labels = len(unique)

    labels_slopes, labels_intervals = [[] for _ in range(total_labels)], [[] for _ in range(total_labels)]
    for label, s, l in zip(labels, slopes, intervals):
        if label != -1:
            labels_slopes[label].append(s)
            labels_intervals[label].append(l)

    labels_slopes = [np.asarray(values) for values in labels_slopes]
    labels_intervals = [np.asarray(values) for values in labels_intervals]
    return labels_slopes, labels_intervals


def get_args_terminal():
    argv = sys.argv[1:]

    source = None
    output = None
    N = 5
    preprocessed = False
    try:
        opts, args = getopt.getopt(argv, "s:o:N:p")
    except getopt.GetoptError:
        print('usage: python example.py -s <source> -o <output> -N <n> -p')
        return None

    for opt, arg in opts:

        if opt == '-s':
            source = arg
        elif opt == '-o':
            output = arg
        elif opt == '-N':
            N = int(arg)
        elif opt == '-p':
            preprocessed = True

    return source, output, N, preprocessed
