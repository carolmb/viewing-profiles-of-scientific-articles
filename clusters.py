# -*- coding: utf-8 -*-
#!/usr/bin/env python
import util
import read_file
import sys,getopt
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.cm as cm

import matplotlib.pyplot as plt

# from pyksc import ksc
from collections import defaultdict
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from read_file import select_original_breakpoints
from scipy.cluster.hierarchy import dendrogram,linkage,cut_tree

def multivariateGrid(col_x, col_y, col_k, df, xlabel,ylabel,k_is_color=False, scatter_alpha=.5):
    def colored_scatter(x, y, c=None):
        def scatter(*args, **kwargs):
            args = (x, y)
            if c is not None:
                kwargs['c'] = c
            kwargs['alpha'] = scatter_alpha
            plt.scatter(*args, **kwargs)

        return scatter

    g = sns.JointGrid(
        x=col_x,
        y=col_y,
        data=df
    )

    color = None
    legends=[]
    for name, df_group in df.groupby(col_k):
        legends.append(name)
        if k_is_color:
            color=name
        g.plot_joint(
            colored_scatter(df_group[col_x],df_group[col_y],color),
        )
        ax1 = sns.distplot(
            df_group[col_x].values,
            ax=g.ax_marg_x,
            hist=False,
            color=color,
            kde_kws={"shade":True}
        )
        ax2 = sns.distplot(
            df_group[col_y].values,
            ax=g.ax_marg_y,
            hist=False,
            color=color,
            vertical=True,
            kde_kws={"shade":True}
        )

    g.ax_joint.set_xlabel(xlabel,fontsize=18)
    g.ax_joint.set_ylabel(ylabel,fontsize=18)


def plot_pca(X,labels,colors,filename):
    pca = PCA(n_components=3)
    pca.fit(X)
    X1 = pca.transform(X)
    x_exp, y_exp, z_exp = pca.explained_variance_ratio_[:3]*100
    label_colors = [colors[l] for l in labels]
    print(np.unique(label_colors,return_counts=True))
    fig = plt.figure(figsize=(8,8))
    df = pd.DataFrame({'x':X1[:,0],'y':X1[:,2],'type':label_colors})
    multivariateGrid('x','y','type',df,'PCA1 (%.2f%%)' % x_exp, 'PCA3 (%.2f%%)' % z_exp, k_is_color=True,scatter_alpha=0.5)
    plt.savefig(filename+"_pca1_pca2.png")
    plt.clf()

    pca = PCA(n_components=2)
    pca.fit(X)
    X1 = pca.transform(X)
    x_exp, y_exp = pca.explained_variance_ratio_[:2]*100
    fig = plt.figure(figsize=(10,10))
    df = pd.DataFrame({'x':X1[:,0],'y':X1[:,1],'type':label_colors})
    multivariateGrid('x','y', 'type', df, 'PCA1 (%.2f%%)' % x_exp, 'PCA2 (%.2f%%)' % y_exp, k_is_color=True, scatter_alpha=.5)
    plt.savefig(filename+'_2.png')

def norm(data):
    m = np.mean(data,axis=0)
    std = np.std(data,axis=0)
    return (data-m)/std

def average_curve_point(data):

    n = data.shape[1]//2

    mtx_inter = np.zeros((len(data),100),dtype=np.float)
    for idx_j,sample in enumerate(data):
        degrees = sample[:n]
        intervals = sample[n:]
        cumsum = np.cumsum(intervals)
        for idx_i,i in enumerate(np.linspace(0,1,100)):
            y = 0
            last_delta_x = cumsum[0]
            for degree,cum_x,delta_x in zip(degrees,cumsum,intervals):
                tan_x = np.tan(degree*np.pi/180)
                if i < cum_x:
                    x = i - last_delta_x
                    y += x*tan_x
                    break
                y += delta_x*tan_x
                last_delta_x = cum_x
            mtx_inter[idx_j][idx_i] = y

    mean = np.mean(mtx_inter,axis=0)
    std = np.std(mtx_inter,axis=0)

    output = (np.linspace(0,1,100),mean,np.zeros(100),np.zeros(100))

    return output

def plot_clusters(plots,output):

    extra_file = open(output[:-4]+'_xs_ys.txt','w')

    plt.figure(figsize=(6,3))
    for color,(x0,y0,s0,s1) in plots.items():
        print(color)
        print(x0.shape,y0.shape,s0.shape,s1.shape)
        plt.errorbar(x0,y0,xerr=s0,yerr=s1,marker='o',markersize=0.7,color=color,linestyle='-',alpha=0.7)
        extra_file.write('-1\n')
        extra_file.write(','.join([str(m) for m in x0])+'\n')        
        extra_file.write(','.join([str(m) for m in y0])+'\n')

    extra_file.close()

    plt.xlabel('time',fontsize=16)
    plt.ylabel('views',fontsize=16)
    plt.tight_layout()
    print(output)
    plt.savefig(output)
    plt.clf()

def get_labels_clustering_hier(X,kclusters,ktotal):

    Z = linkage(X, 'ward')
    dn = dendrogram(Z,leaf_rotation=90.,leaf_font_size=8.,truncate_mode='lastp',p=20,show_contracted=True)
    
    plt.figure(figsize=(25,15))
    plt.ylabel('distance',fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('clustering_ward_p20_k%d.pdf'%ktotal)
    plt.clf()
    
    labels = cut_tree(Z,n_clusters=kclusters)
    
    return [l[0] for l in labels]

def save_groups(dois,labels,data,output):
    labels = np.asarray(labels).reshape(-1,1)
    data = np.concatenate((data,labels),axis=1)
    np.savetxt(output+'curves_label.txt',data,delimiter=' ',fmt='%.18e')

    file = open(output+'_dois.txt','w')
    for doi in dois:
        file.write(doi+'\n')
    file.close()

def get_dois(n,filename):
    data = read_file.load_data(filename)
    data = read_file.filter_outliers(data)

    dois = []
    for i,s,b,xs,ys,p in data:
        if len(s) == n:
            dois.append(i)
    return dois

def generate_groups(get_labels):
    for ktotal in [2,3,4,5]:
        slopes,intervals = select_original_breakpoints(ktotal,'segm/segmented_curves_filtered.txt')
        dois = get_dois(ktotal,'segm/segmented_curves_filtered.txt')
        data = np.concatenate((slopes,intervals),axis=1)
        
        labels = get_labels(data,3,ktotal)
        save_groups(dois,labels,data,'k'+str(ktotal)+'/k'+str(ktotal))
        
        
colors = ['tab:red','tab:blue','tab:orange','tab:green','tab:grey']

def read_args():
    argv = sys.argv[1:]
    
    op1,op2,op3 = False,False,False
    N = -1
    try:
        opts,args = getopt.getopt(argv,"N:",['op1','op2','op3'])
    except getopt.GetoptError:
        print('usage: python example.py --op1 --op2 --op3')
        return None

    for opt,arg in opts:
        if opt == '-N':
            N = arg
        if opt == '--op1':
            op1 = True
        if opt == '--op2':
            op2 = True
        if opt == '--op3':
            op3 = True

    return op1,op2,op3,int(N)

def read_preprocessed_file(N,source):
    dois = open('k'+str(N)+'/k'+str(N)+'_dois.txt','r').read().split()

    original = np.loadtxt(source)
    slopes = original[:,:N]
    intervals = original[:,N:-1]
    labels = original[:,-1]
    labels = labels.astype(int)
    total_labels = len(set(labels))

    data = read_file.load_data()
    
    labels_slopes,labels_intervals = [[] for _ in range(total_labels)],[[] for _ in range(total_labels)]
    for doi,label,s,l in zip(dois,labels,slopes,intervals):
        delta_x = -1
        for i,s,b,xs,ys,p in data:
            if i == doi:
                # print(len(s))
                delta_x = xs[-1] - xs[0]
                break
        # print(delta_x)
        if delta_x >= 5 and delta_x <= 7:
            labels_slopes[label].append(s)
            labels_intervals[label].append(l)

    labels_slopes = [np.asarray(values) for values in labels_slopes]
    labels_intervals = [np.asarray(values) for values in labels_intervals]
    return labels_slopes,labels_intervals


if __name__ =='__main__': 
    op1,op2,op3,N = read_args()
    print(op1,op2,op3)

    if op1:
        generate_groups(get_labels_clustering_hier)
    if op2:
        X = norm(data)
        plot_pca(X,labels,colors,'imgs/pca_%s_%d'%(alg_name,ktotal))
    if op3:
        source = 'k%d/k%d_curves_label.txt' % (N,N)
        
        labels_slopes,labels_intervals = read_preprocessed_file(N,source)
        output = 'k%d/average_curves_label_k%d.pdf' % (N,N)    
        plots = dict()
        for label,(slopes,intervals) in enumerate(zip(labels_slopes,labels_intervals)):
            print(slopes.shape,intervals.shape)
            data = np.concatenate((slopes,intervals),axis=1)
            average = average_curve_point(data)
            plots[colors[label]] = average
        
        plot_clusters(plots,output)
