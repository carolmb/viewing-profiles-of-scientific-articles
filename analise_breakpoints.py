# -*- coding: utf-8 -*-
import sys
import getopt
import numpy as np
from collections import defaultdict

from data_freq_scatter import generate_freq_plots
from artificial_data import generate_artificial_data,get_i
#from data_hist_heat import generate_hist_plots
from read_file import select_original_breakpoints

def norm(xs):
	mmax = max(xs)
	mmin = min(xs)

	return (xs-mmin)/(mmax-mmin)

def get_args_terminal():
    argv = sys.argv[1:]
    
    source = None
    output = None
    N = 5
    try:
        opts,args = getopt.getopt(argv,"s:o:n:")
    except getopt.GetoptError:
        print('usage: example.py -s <source> -o <output> -N <n>')

    for opt,arg in opts:

        if opt == '-s':
            source = arg
        elif opt == '-o':
            output = arg
        elif opt == '-N':
            N = arg
    return source,output,N

if __name__ == "__main__":

    source,output,N = get_args_terminal()
    print(source,output,N)

    nx = 10
    ny = 10

    minx,maxx = 0,90
    deltax = (maxx-minx)/nx
    intervalsx = np.arange(minx,maxx+deltax,deltax)

    miny,maxy = 0,1
    deltay = (maxy-miny)/ny
    intervalsy = np.arange(miny,maxy+deltay,deltay)

    args = [intervalsx,intervalsy]

    # xs,ys = read_file_original(filename='data/plos_one_2019.txt')
    # xs = np.asarray([norm(x) for x in xs])
    # ys = np.asarray([norm(y) for y in ys])

    # filename = 'data/plos_one_2019_breakpoints_k4_original1_data_filtered.txt'
    filename = 'r_code/segmented_curves_filtered.txt'i
    
    '''
    slopes,intervals = select_original_breakpoints(N)
    generate_freq_plots(slopes,intervals,n,'teste/density_')
    # generate_hist_plots(slopes,intervals,n,'imgs/original1/',args)
    '''
    
    generate_artificial_data(N,intervalsx,intervalsy,maxx,source,output)
