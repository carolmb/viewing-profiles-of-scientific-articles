# -*- coding: utf-8 -*-
#!/usr/bin/env python

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
from sklearn.cluster import AgglomerativeClustering
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
    # Do also global Hist:
    # sns.distplot(
    #     df[col_x].values,
    #     ax=g.ax_marg_x,
    #     color='grey'
    # )
    # sns.distplot(
    #     df[col_y].values.ravel(),
    #     ax=g.ax_marg_y,
    #     color='grey',
    #     vertical=True
    # )

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

def average_curve_point(data,labels):

    n = data.shape[1]//2

    print(n,data.shape)

    group_by = defaultdict(lambda:[])
    for sample,label in zip(data,labels):
        group_by[label].append(sample)

    average = []
    
    for k,values in group_by.items():
        values = np.asarray(values)
        
        mtx_inter = np.zeros((len(values),100),dtype=np.float)
        for idx_j,sample in enumerate(values):
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

        print(mean.shape,std.shape)
        average.append((np.linspace(0,1,100),mean,np.zeros(100),std,k))

    return average

def plot_clusters(data,labels,colors,filename):
    average = average_curve_point(data,labels) # ao invés de average_curve(data,labels)

    plt.figure(figsize=(6,3))
    for x0,y0,s0,s1,k in average:
        print(k,colors[k])
        print(x0.shape,y0.shape,s0.shape,s1.shape)
        plt.errorbar(x0,y0,xerr=s0,yerr=s1,marker='o',markersize=0.7,color=colors[k],linestyle='-',alpha=0.7)
    
    plt.xlabel('time',fontsize=16)
    plt.ylabel('views',fontsize=16)
    plt.tight_layout()
    print(filename)
    plt.savefig(filename)
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

def save_groups(labels,data,output):
    labels = np.asarray(labels).reshape(-1,1)
    data = np.concatenate((data,labels),axis=1)
    
    np.savetxt(output+'curves_label.txt',data,delimiter=' ',fmt='%.18e')

def plot_groups(get_labels,alg_name,output):
    for ktotal in [2,3,4,5]:
        slopes,intervals = select_original_breakpoints(ktotal,'r_code/segmented_curves_filtered.txt')
        data = np.concatenate((slopes,intervals),axis=1)
        X = norm(data)
        print(ktotal,len(X))
        
        labels = get_labels(data,3,ktotal)
        save_groups(labels,data,'k'+str(ktotal)+'_')
        
        #plot_pca(X,labels,colors,'imgs/pca_%s_%d'%(alg_name,ktotal))
        #plot_clusters(data,labels,colors,output+str(ktotal)+'.pdf')

colors = ['tab:red','tab:blue','tab:orange','tab:green','tab:grey']

if __name__ =='__main__': 
    output = ''
    plot_groups(get_labels_clustering_hier,'hierar',output)

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

