import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.utils.random import sample_without_replacement

def plot_scatter(X,Y,title,xlabel,ylabel,filename):

    plt.figure(figsize=(4,3))
    
    if np.absolute(np.corrcoef(X,Y)[0][1]) == 1:
        # print(title,np.corrcoef(X,Y))
        # print("aqui",np.corrcoef(X,Y)[0][0])
        plt.scatter(X,Y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
    else: 
        df = pd.DataFrame({'x':X,'y':Y})
        ax1 = sns.jointplot(x="x", y="y", data=df, kind="kde")
        ax1.set_axis_labels(xlabel,ylabel)
        plt.suptitle(title)
    # fig.tight_layout() #(pad=0.5, w_pad=0.5, h_pad=1.2)
    # fig.suptitle(name)
    plt.savefig(filename,format='pdf',bbox_inches='tight')

    plt.close()

    # ax.scatter(X,Y,alpha=0.3)
    # ax.title.set_text(title)
    
    # ax.plot([0,1],[1,0],'k')
    # ax.set_xlim((0,1))
    # ax.set_ylim((0,1))
    # ax.set_xlabel('x '+name.split('_')[0])
    # ax.set_ylabel('x+1 '+name.split('_')[-1])

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
    slopes_freq = (calculate_freqs(n,'alpha',slopes),r"$\alpha$",r"$\alpha$")
    intervals_freq = (calculate_freqs(n,'l',intervals),r"$l$",r"$l$")
    slopes_intervals_freq = (calculate_freqs(n,'alpha_l',slopes,intervals),r"$\alpha$",r"$l$")
    intervals_slopes_freq = (calculate_freqs(n,'l_alpha',intervals,slopes),r"$l$",r"$\alpha$")
    return slopes_freq,intervals_freq,slopes_intervals_freq,intervals_slopes_freq

def plot_freqs(name_freqs,header):
    ((name,freqs),xlabel,ylabel) = name_freqs
    n = len(freqs)
    for i,(X,Y,pearson) in enumerate(freqs):
        title = r"%d ($\rho$ = %.2f)" % (i+1,pearson)
        filename = "%s%s_%d_part_%d.pdf" % (header,name,n+1,i)
        xlabel1 = r"%s$_%d$"%(xlabel,i+1)
        ylabel1 = r"%s$_%d$"%(ylabel,i+2)
        plot_scatter(X,Y,title,xlabel1,ylabel1,filename)
    
def plot_all_freqs(header,*args):
    for freqs in args:
        plot_freqs(freqs,header)
    
def generate_freq_plots(slopes,intervals,n,header):
    freqs = calculate_all_freqs(slopes,intervals,n)
    plot_all_freqs(header,*freqs)