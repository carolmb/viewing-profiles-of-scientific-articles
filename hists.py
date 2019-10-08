import numpy as np
import matplotlib.pyplot as plt

from changes import norm

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
    delta_stds = {1:[],-1:[]}
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

            diffs = get_window(X,np.absolute(Y-Y_pred),b,window_value)
            # print(diffs)
            std = np.std(diffs)
            
            delta_points[key].append(diffs)
            delta_stds[key].append(std)
            
    return delta_points,delta_stds

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
    plt.savefig(filename+'.pdf')    

def plot_hists(data,window,is_norm):
    print(window)

    delta_points,delta_stds = delta_std(data,window,is_norm)
    plot_hist(delta_stds[1],'Increasing degree','imgs/hist%d_incr_std' % window)
    plot_hist(delta_stds[-1],'Decreasing degree','imgs/hist%d_decr_std' % window)

    plot_errorbar(delta_points[1],'Increasing degree','imgs/curve%d_incr' % window,window)
    plot_errorbar(delta_points[-1],'Decreasing degree','imgs/curve%d_decr' % window,window)