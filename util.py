import sys
import getopt
import numpy as np

def read_preprocessed_file(N,source):
    original = np.loadtxt(source)
    print(original[:4])
    slopes = original[:,:N]
    intervals = original[:,N:]
    return slopes,intervals

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