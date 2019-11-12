import numpy as np
from changes import norm
from scipy.stats import skew,kurtosis
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def get_window(X,Y,b,window_value):
    n = len(Y)

    pos = None
    for i in range(n):
        if X[i] <= b and X[i+1] > b:
            pos = i
            break 

    values = Y[max(0,pos-window_value):min(pos+window_value,n-1)]
    
    mean = np.mean(values)

    if pos-window_value < 0:
        values = np.concatenate([np.full(abs(pos-window_value),mean),values])
    if pos+window_value > n-1:
        values = np.concatenate([values,np.full(abs(pos+window_value-n+1),mean)])

    return values

def delta_std(data,window_value,is_norm):
    delta_points = {1:[],-1:[]}
    delta_values = {1:[],-1:[]}
    for sample in data:
        idx = sample[0]

        X = sample[3]
        Y = sample[4]
        # print(X)
        if is_norm:
            X = norm(X)
            Y = norm(Y)

        Y_pred = sample[-1]
        slopes = sample[1]
        b_points = sample[2]
        
        for i,b in enumerate(b_points):
            key = 1 if slopes[i+1] - slopes[i] > 0 else -1

            # diffs = get_window(X,np.absolute(Y-Y_pred),b,window_value)
            y_values = get_window(X,Y,b,window_value)

            # diffs = b_points[i+1]-b_points[i]

            mean = np.mean(y_values)
            std = np.std(y_values)
            med = np.median(y_values)
            vmax = np.max(y_values)
            vmin = np.min(y_values)
            vskew = skew(y_values)
            vkur = kurtosis(y_values)
            
            delta_points[key].append(y_values)
            delta_values[key].append((mean,std,med,vskew,vkur))
    
    delta_values[1] = np.asarray(delta_values[1])
    delta_values[-1] = np.asarray(delta_values[-1]) 
    return delta_points,delta_values

def plot_hist(delta_stds,title,filename):

    fig = plt.figure(figsize=(6,3))
    curve_pos_std = np.nanmean(delta_stds)

    # print(curve_pos_std)
    # print(min(delta_stds),max(delta_stds))

    plt.hist(delta_stds, 15, density=True, facecolor='g', alpha=0.75, range=(0,0.01))
    plt.ylim((0,450))
    plt.title(title)
    plt.savefig(filename+'.pdf')
    fig.clf()

def plot_errorbar(delta_points,title,filename,window):
    ys1 = np.nanmean(delta_points,axis=0)
    std1 = np.nanstd(delta_points,axis=0)
    
    fig = plt.figure(figsize=(6,3))
    xs = list(range(2*window))
    plt.errorbar(xs,ys1,yerr=std1)
    plt.scatter(xs,ys1)
    plt.title(title)
    plt.savefig(filename+'.pdf',bbox_inches='tight')
    plt.clf()


def pca_metrics(delta_values,c):
    values_incres = np.asarray(delta_values[1])
    values_decres = np.asarray(delta_values[-1])
    print(values_incres.shape)
    print(values_decres.shape)

    n_incres = len(values_incres)

    values = np.concatenate((values_incres,values_decres), axis=0)
    #print(values.shape)

    #print(values)

    m = np.mean(values,axis=0)
    std = np.std(values,axis=0)
    values = (values-m)/std

    pca = PCA(n_components=2)
    pca.fit(values)

    Y1 = pca.transform(values[:n_incres,:])
    Y2 = pca.transform(values[n_incres:,:])

    y1_explained, y2_explained = pca.explained_variance_ratio_[:2]
    y1_explained = y1_explained*100
    y2_explained = y2_explained*100

    return Y1,Y2,y1_explained,y2_explained,values_incres[:,c],values_decres[:,c]

def plt_pca_metrics(Y1,Y2,y1_explained,y2_explained,N,c1,c2,title):

    y1_label = 'PCA1 (%.2f%%)' % y1_explained
    y2_label = 'PCA2 (%.2f%%)' % y2_explained


    fig = plt.figure(figsize=(12,4))
    fig.suptitle(title)
    plt.subplot(1,2,1)
    im = plt.scatter(Y1[:,0],Y1[:,1],label='incresing',alpha=0.3,s=2,c=c1)
    plt.xlabel(y1_label)
    plt.ylabel(y2_label)
    fig.colorbar(im)
    plt.legend()
    
    plt.subplot(1,2,2)
    im = plt.scatter(Y2[:,0],Y2[:,1],label='decreasing',alpha=0.3,s=2,c=c2)

    plt.xlabel(y1_label)
    plt.ylabel(y2_label)
    plt.legend()
    fig.colorbar(im)
    plt.savefig(title+'_stats_metrics_'+str(N)+'_orig.png',bbox_inches='tight')
    plt.clf()
    # nomalizar
    # plotar

'''
calcula o std da diferença entre as curvas e o histograma das frequencias dos erros
'''
def plot_hists(data,window,is_norm):
    print(window)
    print('is_norm',is_norm)

    delta_points,delta_values = delta_std(data,window,is_norm)
    

    # idxs = np.logical_and(Y2[:,0] > 3,Y2[:,1] > 4)
    # print(idxs)
    # print(delta_values[-1][idxs])
    Y1,Y2,y1_explained,y2_explained,c1,c2 = pca_metrics(delta_values,0)
    plt_pca_metrics(Y1,Y2,y1_explained,y2_explained,window,c1,c2,'mean')

    Y1,Y2,y1_explained,y2_explained,c1,c2 = pca_metrics(delta_values,1)
    plt_pca_metrics(Y1,Y2,y1_explained,y2_explained,window,c1,c2,'std')

    Y1,Y2,y1_explained,y2_explained,c1,c2 = pca_metrics(delta_values,2)
    plt_pca_metrics(Y1,Y2,y1_explained,y2_explained,window,c1,c2,'median')

    Y1,Y2,y1_explained,y2_explained,c1,c2 = pca_metrics(delta_values,-1)
    plt_pca_metrics(Y1,Y2,y1_explained,y2_explained,window,c1,c2,'kurtosis')
    
    Y1,Y2,y1_explained,y2_explained,c1,c2 = pca_metrics(delta_values,-2)
    plt_pca_metrics(Y1,Y2,y1_explained,y2_explained,window,c1,c2,'skew')

    # delta_stds = {}
    # delta_stds[1] = np.asarray(delta_values[1])[:,1]
    # delta_stds[-1] = np.asarray(delta_values[-1])[:,1]

    # plot_hist(delta_stds[1],'Increasing degree','imgs/hist%d_incr_std' % window)
    # plot_hist(delta_stds[-1],'Decreasing degree','imgs/hist%d_decr_std' % window)

    # plot_errorbar(delta_points[1],'Increasing degree','imgs/curve%d_incr_original' % window,window)
    # plot_errorbar(delta_points[-1],'Decreasing degree','imgs/curve%d_decr_original' % window,window)
