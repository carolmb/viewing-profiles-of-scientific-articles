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

def plot_pca(y1,colors,xlabel,ylabel,title,filename):

    plt.figure()    
    

    plt.scatter(y1[:,0],y1[:,1],c=colors,alpha=0.2)

    plt.title(title)
	
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(bbox_to_anchor=(1.2,1.0))
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

data_breaks = load_data('data/plos_one_2019_breakpoints_k4_original1_data_filtered.txt')

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

intervals = [(2014.5,2015.5),(2016,2016.5),(2017.5,2018.5)]
decr_all = []
for x0,x1 in intervals:
    incr,decr = get_breaks_freq(data_breaks,x0,x1)
    decr_all.append(set(decr))

all_slopes = []
all_breaks = []
all_dois = []
for sample in data_breaks:
    slopes = sample[1]
    if len(slopes) != 5:
        continue
    all_dois.append(sample[0])
    breakpoints = sample[2]
    all_slopes.append(np.asarray(slopes))
    all_breaks.append(np.asarray(breakpoints))

all_slopes = np.asarray(all_slopes)
all_breaks = np.asarray(all_breaks)

print(all_slopes.shape,all_breaks.shape)

all_slopes_breaks = np.concatenate((all_slopes,all_breaks),axis=1)
print('comferir ------->>',all_slopes_breaks.shape)

y1,xlabel,ylabel = get_pca_infos(all_slopes_breaks)

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


plot_pca(y1,cs,xlabel,ylabel,"pca colorido por picos","pca_colorido_por_picos.png")





