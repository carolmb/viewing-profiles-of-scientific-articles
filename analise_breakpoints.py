# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict

from data_freq_scatter import generate_freq_plots
from artificial_data import generate_artificial_data,get_i
from data_hist_heat import generate_hist_plots
from read_file import select_original_breakpoints

def norm(xs):
	mmax = max(xs)
	mmin = min(xs)

	return (xs-mmin)/(mmax-mmin)

if __name__ == "__main__":

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

    filename = 'data/plos_one_2019_breakpoints_k4_original1_data_filtered.txt'
    
    for n in [2,3,4,5]:
        slopes,intervals = select_original_breakpoints(n)
        generate_freq_plots(slopes,intervals,n,'imgs/original1/scatter_')
        generate_hist_plots(slopes,intervals,n,'imgs/original1/',args)

    Ns = [2,3,4,5]
    generate_artificial_data(Ns,intervalsx,intervalsy,maxx,'original1')