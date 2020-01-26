#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.cm as cm
# import matplotlib 
# matplotlib.use('agg') 
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

# from pyksc import ksc
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram,linkage,cut_tree

from read_file import select_original_breakpoints

def plot_pca(data,X,labels,colors,filename):
    pca = PCA(n_components=2)
    pca.fit(X)
    X1 = pca.transform(X)
    label_colors = [colors[l] for l in labels]
    print(np.unique(label_colors,return_counts=True))
    plt.figure(figsize=(10,10))
    plt.scatter(X1[:,0],X1[:,1],color=label_colors,alpha=0.4)

    # average = average_curve(data,labels)
    # for x0,y0,s0,k in average:
    #     cov = np.cov(x0,y0, rowvar=False)
    #     vals, vecs = eigsorted(cov)
    #     theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

        # ellipse = Ellipse((np.mean(x0),np.mean(y0)),
        #     width=np.std(x0)*2,
        #     height=np.std(y0)*2,angle=theta,fill=False,linewidth=2.0)
        
        # ellipse.set_alpha(0.4)
        # ellipse.set_edgecolor(colors[k])
        # ellipse.set_facecolor(None)
        # ax = plt.gca()
        # ax.add_artist(ellipse)

    plt.savefig(filename)
    plt.clf()

def norm(data):
    m = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    return (data-m)/std

def get_average_curve(values,n,k):
    means = np.mean(values,axis=0)
    degrees = means[:n]
    stds_degrees = np.std(values,axis=0)[:n]
    
    breaks = means[n:]
    stds_breaks = np.std(values,axis=0)[n:]
    
    ys_stds = [0]
    y0_values = values[:,0]*(values[:,n+0]*0.0174533)
    ys_stds.append(np.std(y0_values))
    for i in range(1,n):
        y1_values = y0_values + values[:,i]*(values[:,n+i]*0.0174533)
        ys_stds.append(np.std(y1_values))
        y0_values = y1_values

    x0 = [0]
    y0 = [0]
    b0 = 0
    s0_breaks = [0]

    for d,b,s1 in zip(degrees,breaks,stds_breaks):
        tan = np.tan(d*0.0174533)
        b0 += b
        x0.append(b0)
        y0.append(y0[-1]+b*tan)
        s0_breaks.append(s1)
    print('x0',x0,'\ny0',y0,'\nstds_breaks',s0_breaks,'\nys_stds',ys_stds)
    average = (x0,y0,s0_breaks,ys_stds,k)
    return average

def average_curve(data,labels):
    n = data.shape[1]//2

    print(n,data.shape)

    group_by = defaultdict(lambda:[])
    for sample,label in zip(data,labels):
        group_by[label].append(sample)

    average = []
    
    for k,values in group_by.items():
        values = np.asarray(values)
        
        average.append(get_average_curve(values,n,k))
    average.append(get_average_curve(data,n,-1))
    return average

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

def plot_clusters(data,labels,colors,filename):
    average = average_curve(data,labels)

    legend = ['1','2','3','4','all']
    plt.figure(figsize=(3,3))
    for x0,y0,s0,s1,k in average:
        print(k,colors[k])
        plt.errorbar(x0,y0,xerr=s0,yerr=s1,marker='o',color=colors[k],linestyle='-',alpha=0.9,label=legend[k])

    plt.legend()

    plt.savefig(filename)
    plt.clf()

def get_labels_kmeans(X):
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    return labels

def get_labels_clustering_hier(X,ktotal):
    # alg = AgglomerativeClustering(n_clusters=3,linkage='single')
    # alg = alg.fit(X)
    # plot_dendrogram(alg, truncate_mode='level', p=3)
    
    plt.figure(figsize=(30,20))
    Z = linkage(X, 'ward')
    dn = dendrogram(Z,leaf_rotation=90.,leaf_font_size=8.,truncate_mode='lastp',p=20,show_contracted=True)
    plt.ylabel('distance',fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('clustering_ward_p20.pdf')
    plt.clf()
    labels = cut_tree(Z,n_clusters=ktotal)
    return [l[0] for l in labels]

def get_labels_ksc(X):
    cents, labels, shift, distc = ksc.ksc(X, 3)
    return labels

def plot_groups(get_labels,alg_name):
    for ktotal in [5]:
        slopes,intervals = select_original_breakpoints(ktotal)
        data = np.concatenate((slopes,intervals),axis=1)
        X = norm(data)
        labels = get_labels(data,ktotal)
        plot_pca(data,norm(data),labels,colors,'imgs/pca_%s_%d.pdf'%(alg_name,ktotal))
        plot_clusters(data,labels,colors,'imgs/average_curve_%s_%d.pdf'%(alg_name,ktotal))

colors = ['#307438','#b50912','#028090','magenta','orange']

if __name__ =='__main__': 
    plot_groups(get_labels_clustering_hier,'hierar')

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

