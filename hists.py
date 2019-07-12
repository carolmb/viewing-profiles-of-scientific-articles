import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from igraph import *
from scipy import stats
from collections import defaultdict
from analise_breakpoints import read_file_original

def read_file(samples_breakpoints='data/breakpoints_k4it.max100stop.if.errorFALSE.txt'):
    samples_breakpoints = open(samples_breakpoints,'r').read().split('\n')[:-1]
    total_series = len(samples_breakpoints)
    slopes = []
    breakpoints = []
    idxs = []
    for i in range(0,total_series,3):
        idx = int(samples_breakpoints[i]) - 1
        
        slopes_i = [float(n) for n in samples_breakpoints[i+1].split(' ')]
        breakpoints_i = [float(n) for n in samples_breakpoints[i+2].split(' ')]
        
        idxs.append(idx)
        slopes.append(np.asarray(slopes_i))
        breakpoints.append(np.asarray(breakpoints_i))
    
    return np.asarray(idxs),np.asarray(slopes),np.asarray(breakpoints)

xs,ys = read_file_original(filename='data/plos_one_data_total.txt')
idxs,breakpoints,slopes = read_file('data/plos_one_total_breakpoints_k4it.max100stop.if.errorFALSE_filtered.txt')

# for x,y in zip(xs[:100],ys[:100]):
#     print(x,y)

def plot_hist(visualizations,filename):
    n, bins, patches = plt.hist(visualizations, 20, density=True, facecolor='g', alpha=0.75)
    plt.yscale('log')
    plt.savefig(filename)
    plt.cla()

# return indexes
def group_by_num_vis(ys,k1,k2):
    groups = defaultdict(lambda:[])
    deleted = []
    for i,y in enumerate(ys):
        if y > k1:
            if y < k2:
                groups[1].append(i)
            else:
                groups[2].append(i)
        else:
            deleted.append(i)
    return groups,deleted

visualizations = np.asarray([y[-1] for y in ys])
print(stats.describe(visualizations))  
plot_hist(visualizations,'hist_all_data.png')

groups,deleted = group_by_num_vis(visualizations,1000,5000)

deleted_delta_time = []
for d in deleted:
    x = xs[d]
    delta = x[-1]-x[0]
    deleted_delta_time.append(delta)
plot_hist(deleted_delta_time,'hist_deleted_articles.png')

idxs = idxs.tolist()

for k,v in groups.items():
    ys_v = visualizations[v]
    print(k,len(ys_v),min(ys_v),max(ys_v),np.std(ys_v))
    plot_hist(ys_v,'hist_group_'+str(k)+'.png')

    breakpoints_freq = []
    # print(v[:10])
    # print(idxs[:10])
    for i in v:
        try:
            i = idxs.index(i)
            breakpoints_freq.append(len(breakpoints[i]))
        except:
            pass
    plot_hist(breakpoints_freq,'hist_breakpoints_freq_'+str(k)+'.png')