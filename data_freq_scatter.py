import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.utils.random import sample_without_replacement

def plot_scatter(ax,X,Y,title,name):
    ax.scatter(X,Y,alpha=0.3)
    # ax.plot([0,1],[1,0],'k')
    ax.title.set_text(title)
    # ax.set_xlim((0,1))
    # ax.set_ylim((0,1))
    ax.set_xlabel('x '+name.split('_')[0])
    ax.set_ylabel('x+1 '+name.split('_')[-1])

def freq_xi_xi1(X):
    pointx = []
    pointy = []

    for x in X:
        for i in range(len(x)-1):
            pointx.append(x[i])
            pointy.append(x[i+1])
    pearson,_ = pearsonr(pointx,pointy)

    pointx = np.asarray(pointx)
    pointy = np.asarray(pointy)

    if len(X) > 10000:
        idxs = sample_without_replacement(len(X),10000)
        pointx = pointx[idxs]
        pointy = pointy[idxs]

    return pointx,pointy,pearson

def calculate_freqs(n,name,*values):
    freqs = []
    for i in range(n-1):
        values_i = values[0][:,i:i+2]
        if len(values) > 1:
            values_i = np.concatenate((values[0][:,i:i+1],values[1][:,i+1:i+2]),axis=1)
        X,Y,pearson = freq_xi_xi1(values_i)
        freqs.append((X,Y,pearson))
    return name,freqs

def calculate_all_freqs(slopes,intervals,n):
    slopes_freq = calculate_freqs(n,'slopes',slopes)
    intervals_freq = calculate_freqs(n,'intervals',intervals)
    slopes_intervals_freq = calculate_freqs(n,'slopes_intervals',slopes,intervals)
    intervals_slopes_freq = calculate_freqs(n,'intervals_slopes',intervals,slopes)
    return slopes_freq,intervals_freq,slopes_intervals_freq,intervals_slopes_freq

def plot_freqs(fig,name_freqs,header):
    name,freqs = name_freqs
    n = len(freqs)
    for i,(X,Y,pearson) in enumerate(freqs):
        ax = fig.add_subplot(n,1,i+1)
        title = str(i+1) + ' (pearson=' + str(pearson)[:9]+')'
        plot_scatter(ax,X,Y,title,name)
    fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.2)
    # fig.suptitle(name)
    fig.savefig(header+name+"_"+str(n+1)+".pdf",format='pdf',bbox_inches='tight')

def plot_all_freqs(header,*args):
    n = len(args[0][1])
    fig = plt.figure(figsize=(6,4*n))
    for freqs in args:
        plot_freqs(fig,freqs,header)
        fig.clf()

def generate_freq_plots(slopes,intervals,n,header):
    freqs = calculate_all_freqs(slopes,intervals,n)
    plot_all_freqs(header,*freqs)