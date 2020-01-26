import json
import xnet
import glob
import numpy as np
import matplotlib.pyplot as plt

from igraph import *
from scipy import signal
from read_file import load_data
from itertools import combinations
from collections import defaultdict

#from collections import defaultdict
from sklearn.decomposition import PCA

from scatter import Score
#from read_file import select_original_breakpoints,read_artificial_breakpoints

def plot_pca(y1,colors,all_colors,xlabel,ylabel,title,filename):

    y1_by_colors = defaultdict(lambda:[])
    for c,y in zip(colors,y1):
        y1_by_colors[c].append(y)

    N = len(y1_by_colors)
    print(y1_by_colors.keys())

    print(N//2, N//(N//2),N)

    fig,axs = plt.subplots(N//2, N//(N//2), figsize=(2*N//(N//2),2*N//2), sharey=True, sharex=True)
    
    fig.suptitle(title)
    
    for c,y1s in y1_by_colors.items():
        print(c)
        y1s = np.asarray(y1s)
        axs[c//2,c%2].scatter(y1s[:,0],y1s[:,1],alpha=0.2)
        axs[c//2,c%2].set_title(all_colors[c])

    
    fig.text(0.5, 0.04, xlabel, ha='center', fontsize=14)
    fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical', fontsize=14)
    
    # fig.set_xlabel(xlabel, fontsize=14)
    # fig.set_ylabel(ylabel, fontsize=14)
    # plt.legend(bbox_to_anchor=(1.2,1.0))
    plt.savefig(filename,bbox_inches='tight')
    plt.clf()

def get_pca_infos(original):
    pca = PCA(n_components=2)
    pca.fit(original)

    y1 = pca.transform(original)
    
    y1_explained, y2_explained = pca.explained_variance_ratio_[:2]
    y1_explained = y1_explained*100
    y2_explained = y2_explained*100

    y1_label = 'PCA1 (%.2f%%)' % y1_explained
    y2_label = 'PCA2 (%.2f%%)' % y2_explained

    return y1,y1_label,y2_label

def get_breaks_freq(data_breaks,x0,x1):
    incr = []
    decr = []
    for sample in data_breaks:
        doi = sample[0]
        #if not sample[0] in dois:
    #    continue
        slopes = sample[1]
        breakpoints = sample[2]
        n = len(breakpoints)
        delta_time = sample[3][-1] - sample[3][0]
        begin = sample[3][0]
        for i in range(n):
            moment = begin + delta_time*breakpoints[i]
            if moment >= x0 and moment < x1:
                if slopes[i+1] > slopes[i]:
                    incr.append(doi)
                else:
                    decr.append(doi)

    return incr,decr

# seleciona os slopes e breaks de curvas com N segmentos
def get_data_by_N(data_breaks,N):
    all_slopes = []
    all_breaks = []
    all_dois = []
    for sample in data_breaks:
        slopes = sample[1]
        if len(slopes) != N:
            continue
        all_dois.append(sample[0])
        breakpoints = sample[2]
        all_slopes.append(np.asarray(slopes))
        all_breaks.append(np.asarray(breakpoints))

    all_slopes = np.asarray(all_slopes)
    all_breaks = np.asarray(all_breaks)

    print(all_slopes.shape,all_breaks.shape)

    all_slopes_breaks = np.concatenate((all_slopes,all_breaks),axis=1)
    print('conferir ------->>',all_slopes_breaks.shape)

    m = np.mean(all_slopes_breaks,axis=0)
    std = np.std(all_slopes_breaks,axis=0)
    all_slopes_breaks = (all_slopes_breaks - m)/std

    return all_slopes_breaks,all_dois

# cada pico representa um pico onde o artigo pode estar ou não
# devolve as cores de cada artigo e devolve as cores possíveis (lista com 2^3)
def get_colors_by_intervals(decr_all,all_dois):
    N = len(all_dois)
    colors = np.zeros((N,3))
    for i,doi in enumerate(all_dois):
        for j,decr in enumerate(decr_all):
            if doi in decr:
                colors[i][j] = 1

    all_colors = set()
    for i in range(len(colors)):
        all_colors.add(tuple(colors[i].tolist()))

    all_colors = list(all_colors)
    print(all_colors)

    cs = []
    for c in colors:
        cs.append(all_colors.index(tuple(c.tolist())))

    return cs,all_colors

if __name__ == '__main__':
    data_breaks = load_data('data/plos_one_2019_breakpoints_k4_original1_data_filtered.txt')

    intervals_decr = [(2014.5,2015.5),(2015.80,2016.5),(2017.5,2018.5)]
    decr_all = []
    for x0,x1 in intervals_decr:
        incr,decr = get_breaks_freq(data_breaks,x0,x1)
        decr_all.append(set(decr))

    intervals_incr = [(2015.5,2016),(2016.5,2017.1),(2017.5,2018.2)]
    incr_all = []
    for x0,x1 in intervals_incr:
        incr,decr = get_breaks_freq(data_breaks,x0,x1)
        incr_all.append(set(incr))

    all_slopes_breaks,all_dois = get_data_by_N(data_breaks,5)

    y1,xlabel,ylabel = get_pca_infos(all_slopes_breaks)

    cs,all_colors = get_colors_by_intervals(decr_all,all_dois)
    plot_pca(y1,cs,all_colors,xlabel,ylabel,"pca colorido por picos (decr)","pca_colorido_por_picos(decr).png")

    cs,all_colors = get_colors_by_intervals(incr_all,all_dois)
    plot_pca(y1,cs,all_colors,xlabel,ylabel,"pca colorido por picos (incr)","pca_colorido_por_picos(incr).png")

