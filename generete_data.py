# -*- coding: utf-8 -*-
import util
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

if __name__ == "__main__":

    source,output,N,preprocessed = util.get_args_terminal()
    print('Terminal args',source,output,N)

    nx = 10
    ny = 10

    minx,maxx = 0,90
    deltax = (maxx-minx)/nx
    intervalsx = np.arange(minx,maxx+deltax,deltax)

    miny,maxy = 0,1
    deltay = (maxy-miny)/ny
    intervalsy = np.arange(miny,maxy+deltay,deltay)

    if source == None: # em caso de nada ter sido informado
        slopes,intervals = select_original_breakpoints(N)
    elif not preprocessed: # em caso do arquivo informado ser no formato padrao
        slopes,intervals = select_original_breakpoints(N,source)
    else: # em caso do arquivo informado ser preprocessado
        slopes,intervals = util.read_preprocessed_file(N,source)

    generate_artificial_data(slopes,intervals,N,intervalsx,intervalsy,maxx,source,output)


'''
para gerar para todas as curvas com k = 3
    python generete_data.py -N 3 -o data_k3/

para gerar por grupo
    python generete_data.py -s data_by_label_k3/curves_label_0.txt -N 3 -p -o data_by_label_k3/
'''