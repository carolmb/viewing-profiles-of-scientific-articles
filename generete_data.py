# -*- coding: utf-8 -*-
import util
import numpy as np

from artificial_data import generate_artificial_data
# from data_hist_heat import generate_hist_plots
from read_file import select_original_breakpoints, save


def norm(xs):
    mmax = max(xs)
    mmin = min(xs)

    return (xs - mmin) / (mmax - mmin)


if __name__ == "__main__":

    source, output, N, preprocessed = util.get_args_terminal()
    print('Terminal args', source, output, N)

    nx = 10
    ny = 10

    minx, maxx = 0, 90
    deltax = (maxx - minx) / nx
    intervalsx = np.arange(minx, maxx + deltax, deltax)

    miny, maxy = 0, 1
    deltay = (maxy - miny) / ny
    intervalsy = np.arange(miny, maxy + deltay, deltay)

    # if source == None: # em caso de nada ter sido informado
    #     slopes, intervals = select_original_breakpoints(N)
    #     models = generate_artificial_data([slopes], [intervals], N, intervalsx, intervalsy, maxx)
    # elif not preprocessed: # em caso do arquivo informado ser no formato padrao
    #     slopes, intervals = select_original_breakpoints(N, source)
    #     models = generate_artificial_data([slopes], [intervals], N, intervalsx, intervalsy, maxx)
    # else: # em caso do arquivo informado ser com labels
    if True:
        label_sources = [('clusters\\clusters\\clusters_ind_single_0.50_2.txt'),
                         ('clusters\\clusters\\clusters_ind_single_0.35_3.txt'),
                         ('clusters\\clusters\\clusters_ind_single_0.47_4.txt'),
                         ('clusters\\clusters\\clusters_ind_single_0.56_5.txt')]

        source = 'segm\\segmented_curves_filtered.txt'

        for N, labels in zip([2, 3, 4, 5], label_sources):
            label_slopes, label_intervals = util.read_preprocessed_file(N, source, labels)
            models = generate_artificial_data(label_slopes, label_intervals, N, intervalsx, intervalsy, maxx)

            for model_name, values in models.items():
                X, Y = values[0]
                for x, y in values[1:]:
                    X = np.concatenate((X, x), axis=0)
                    Y = np.concatenate((Y, y), axis=0)
                save(X, Y, 'clusters\\' + model_name + '_k' + str(N) + '_label.txt')

    for model_name, values in models.items():
        print(model_name)
        if len(values) == 1:
            X, Y = values
            save(X, Y, output + model_name + '_' + str(N) + '_all.txt')
        # else:
        #     X,Y = values[0]
        #     for x,y in values[1:]:
        #         X = np.concatenate((X,x),axis=0)
        #         Y = np.concatenate((Y,y),axis=0)
        #     save(X,Y,output+model_name+'_k'+str(N)+'_label.txt')

'''
para gerar para todas as curvas com k = 3
    python generete_data.py -N 3 -o data_k3/

para gerar por grupo
    python generete_data.py -s k3/k3_curves_label.txt -N 3 -p -o k3/
'''
