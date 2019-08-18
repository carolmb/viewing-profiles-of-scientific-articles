#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
# from analise_breakpoints import read_file,read_file_original
# from analise_breakpoints import breakpoints2intervals
# from sklearn.cluster import KMeans
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.datasets import make_blobs
# from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import describe
from pyksc import ksc

def read_file_original(filename='data/samples.txt'):
    samples_breakpoints = open(filename,'r').read().split('\n')[:-1]
    total_series = len(samples_breakpoints)
    X = []
    Y = []
    for i in range(0,total_series,2):
        xs = [float(n) for n in samples_breakpoints[i].split(',')]
        ys = [float(n) for n in samples_breakpoints[i+1].split(',')]
        # breakpoints_i.append(1.0)
        X.append(np.asarray(xs))
        Y.append(np.asarray(ys))
    
    return np.asarray(X),np.asarray(Y)

def read_file(samples_breakpoints='data/breakpoints_k4it.max100stop.if.errorFALSE.txt',N=4):
    samples_breakpoints = open(samples_breakpoints,'r').read().split('\n')[:-1]
    total_series = len(samples_breakpoints)
    slopes = []
    breakpoints = []
    idxs = []
    preds = []
    for i in range(0,total_series,4):
        idx = int(samples_breakpoints[i]) - 1
        
        slopes_i = [float(n) for n in samples_breakpoints[i+1].split(' ')]
        
        breakpoints_i = [float(n) for n in samples_breakpoints[i+2].split(' ')]
        y_pred_i = [float(n) for n in samples_breakpoints[i+3].split(' ')]
        # breakpoints_i.append(1.0)


        if len(slopes_i) == N:
            idxs.append(idx)
            slopes.append(np.array(slopes_i))
            breakpoints.append(np.array(breakpoints_i))
            preds.append(np.array(y_pred_i))
    return np.asarray(idxs),np.asarray(slopes),np.asarray(breakpoints),np.asarray(preds)


def plot_pca(ktotal,alg,X):
    kmeans = alg(n_clusters=ktotal)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    print(labels[:10])
    
    label_colors = [colors[l] for l in labels]
    
    pca = PCA(n_components=2)
    pca.fit(X)
    X1 = pca.transform(X)
    plt.scatter(X1[:,0],X1[:,1],color=label_colors)
    plt.show()
    
    return labels

def plot_lines_blox_plot(intervals_by_label,ktotal,dim=8):
    fig1 = plt.figure(figsize=(4,2*(ktotal+1)))
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    x = [i for i in range(dim)]

    colors = {0:'blue',1:'red',2:'green',3:'orange',4:'gray',5:'cyan',6:'magenta'}

    for k,xs in intervals_by_label.items():
        xs = np.asarray([np.asarray(z) for z in xs])
        d = describe(xs,axis=0)
        means = d.mean
        stds = np.sqrt(d.variance)

        ax.errorbar(x,means,yerr=stds,fmt='-o',label=str(k)+'(mean w/ std)',c=colors[k],alpha=0.8)
        
        nx = xs.shape[1]
        data_to_boxplot = []
        for i in range(nx):
            data_to_boxplot.append(xs[:,i])
        ax1 = fig1.add_subplot(ktotal,1,k+1)
        ax1.boxplot(data_to_boxplot,boxprops=dict(color=colors[k],alpha=0.7))
        # ax.plot(x,mins,label=str(k)+' (mins)',c=colors[k],alpha=0.5)
        # ax.plot(x,maxs,label=str(k)+' (maxs)',c=colors[k],alpha=0.5)

    ax.legend(loc='upper center', bbox_to_anchor=(1, 1), ncol=1)
    ax.set_xticks(x,[str(k) for k in x])
    ax.set_xlabel('intervals')
    fig.tight_layout()
    fig1.tight_layout()
    plt.show()


# In[10]:


# range_n_clusters = [2, 3, 4, 5, 6]
# test_n_clusters(range_n_clusters,X,KMeans,'kmeans')

idxs,slopes_artificial,intervals_artificial,preds = read_file(samples_breakpoints='data/plos_one_total_breakpoints_k4_original1_data_filtered.txt')

intervals_artificial = np.asarray(intervals_artificial)
xs,ys = read_file_original('data/plos_one_data_total.txt')
# slopes_original = [s for s in slopes_original if len(s) == 4]
# breakpoints_original = [s for s in breakpoints_original if len(s) == 4]
# intervals_original = np.asarray([breakpoints2intervals(b) for b in breakpoints_original])

# DADOS NORMALIZADOS POR TODOS OS DADOS
# original_data = np.concatenate((slopes_original,intervals_original),axis=1)
artificial_data_original = np.concatenate((slopes_artificial,intervals_artificial),axis=1)
# all_data = np.concatenate((original_data,artificial_data),axis=0)
all_data = artificial_data_original

m = np.mean(all_data,axis=0)
std = np.std(all_data,axis=0)
# original_data = (original_data - m)/std
artificial_data = (artificial_data_original -m)/std
print(artificial_data.shape)

def plot_clusters(labels,artificial_data_original):
    group_by = {0:[],1:[],2:[]}
    for l,i in zip(labels,artificial_data_original):
        group_by[l].append(i)

    for k,values in group_by.items():
        values = np.asarray(values)

        means = np.mean(values,axis=0)

        stds = np.std(values,axis=0)
        degrees = means[:4]
        breaks = means[4:]

        x0 = [0]
        y0 = [0]
        b0 = 0
        s0 = [0]

        for d,b,s in zip(degrees,breaks,stds):
            x0.append(b)
            y0.append(y0[-1]+(b-b0)*d)
            b0 = b
            s0.append(s)

        x0.append(1)
        y0.append(1)
        s0.append(0)
        plt.errorbar(x0,y0,yerr=s0,c=colors[k])

    #     plt.errorbar(list(range(len(means))),means,yerr=stds)

    plt.savefig('ksc_curves.png')


ktotal = 3
colors = plt.get_cmap('magma')
colors = colors(np.linspace(0,1,ktotal))
print(colors)

cents, assign, shift, distc = ksc.ksc(artificial_data, ktotal)
print(cents)
print(assign)

X = artificial_data
pca = PCA(n_components=2)
pca.fit(X)
X1 = pca.transform(X)
plt.scatter(X1[:,0],X1[:,1],c=assign)
plt.savefig('ksc_clusters.png')
# labels = plot_pca(ktotal,alg,artificial_data)

plot_clusters(assign,artificial_data_original)
