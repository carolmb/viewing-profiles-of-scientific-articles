#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import matplotlib.cm as cm
# import matplotlib 
# matplotlib.use('agg') 
import matplotlib.pyplot as plt

# from pyksc import ksc
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

from read_file import select_original_breakpoints

def plot_pca(X,labels,colors,filename):
    pca = PCA(n_components=2)
    pca.fit(X)
    X1 = pca.transform(X)
    label_colors = [colors[l] for l in labels]
    plt.scatter(X1[:,0],X1[:,1],color=label_colors)
    plt.savefig(filename)
    plt.clf()

def norm(data):
    m = np.mean(data)
    std = np.std(data)
    return (data-m)/std

def average_curve(data,labels):
    n = data.shape[1]//2

    print(n,data.shape)

    group_by = defaultdict(lambda:[])
    for sample,label in zip(data,labels):
        group_by[label].append(sample)

    average = []
    for k,values in group_by.items():
        values = np.asarray(values)
        
        means = np.mean(values,axis=0)

        stds = np.std(values,axis=0)[n:]
        degrees = means[:n]
        breaks = means[n:]

        x0 = [0]
        y0 = [0]
        b0 = 0
        s0 = [0]

        for d,b,s in zip(degrees,breaks,stds):
            tan = np.tan(d*0.0174533)
            b0 += b
            x0.append(b0)
            y0.append(y0[-1]+b*tan)
            s0.append(s)

        average.append((x0,y0,s0,k))
    return average

def plot_clusters(data,labels,colors,filename):
    average = average_curve(data,labels)

    
    plt.figure(figsize=(3,3))
    for x0,y0,s0,k in average:
        plt.errorbar(x0,y0,yerr=s0,marker='o',color=colors[k],linestyle='-')
        # plt.scatter(x0,y0,color=colors[k])

    plt.savefig(filename)
    plt.clf()

def get_labels_kmeans(X):
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    return labels

def get_labels_ksc(X):
    cents, labels, shift, distc = ksc.ksc(X, 3)
    return labels

def plot_groups(get_labels,alg_name):
    for ktotal in [4,5]:
        slopes,intervals = select_original_breakpoints(ktotal)
        data = np.concatenate((slopes,intervals),axis=1)
        X = norm(data)
        labels = get_labels(data)
        plot_pca(norm(data),labels,colors,'imgs/pca_%s_%d.pdf'%(alg_name,ktotal))
        plot_clusters(data,labels,colors,'imgs/average_curve_%s_%d.pdf'%(alg_name,ktotal))

colors = ['red','green','blue','magenta','orange']

plot_groups(get_labels_kmeans,'kmeans')
# plot_groups(get_labels_ksc,'ksc')

# pca usando todos os dados OK
# clusters usando todo os dados
# sinais aleatórios para comparar os originais OK
# angulos diferentes e intervalos probabilisticos OK
# visualizar intervalos 1 e 3 OK

# TODO
# encontrar o centro
# a curva média de cada conjunto (usar os labels)
# testar outras medidas

