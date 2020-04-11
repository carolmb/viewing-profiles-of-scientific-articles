import sys
import getopt
import numpy as np

def read_preprocessed_file(N,source):
    original = np.loadtxt(source)
    slopes = original[:,:N]
    intervals = original[:,N:-1]
    labels = original[:,-1]
    labels = labels.astype(int)
    total_labels = len(set(labels))

    labels_slopes,labels_intervals = [[] for _ in range(total_labels)],[[] for _ in range(total_labels)]
    for label,s,l in zip(labels,slopes,intervals):
        
        labels_slopes[label].append(s)
        labels_intervals[label].append(l)

    labels_slopes = [np.asarray(values) for values in labels_slopes]
    labels_intervals = [np.asarray(values) for values in labels_intervals]
    return labels_slopes,labels_intervals

def get_args_terminal():
    argv = sys.argv[1:]
    
    source = None
    output = None
    N = 5
    preprocessed = False
    try:
        opts,args = getopt.getopt(argv,"s:o:N:p")
    except getopt.GetoptError:
        print('usage: python example.py -s <source> -o <output> -N <n> -p')
        return None

    for opt,arg in opts:

        if opt == '-s':
            source = arg
        elif opt == '-o':
            output = arg
        elif opt == '-N':
            N = int(arg)
        elif opt == '-p':
            preprocessed = True

    return source,output,N,preprocessed