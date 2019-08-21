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

        X = sample[3][1:]
        Y = sample[4][1:]
        if is_norm:
            X = norm(X)
            Y = norm(Y)

        Y_pred = sample[-1]
        slopes = sample[1]
        b_points = sample[2]
        
        for i,b in enumerate(b_points):
            key = 1 if slopes[i+1] - slopes[i] > 0 else -1

            diffs = get_window(X,np.absolute(Y-Y_pred),b,window_value)
            std = np.std(diffs)
            
            delta_points[key].append(diffs)
            delta_stds[key].append(std)
            
    return delta_points,delta_stds

def plot_hists(data,window,is_norm):
    print(window)

    delta_points,delta_stds = delta_std(data,window,is_norm)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3),sharey=True)
    curve_pos_std = np.nanmean(delta_stds[1])

    print(curve_pos_std)
    print(min(delta_stds[1]),max(delta_stds[1]))

    ax1.hist(delta_stds[1], 10, density=True, facecolor='g', alpha=0.75, range=(0,0.02))
    ax1.set_title('Curve (positive) std mean = %.4f '% curve_pos_std)
    
    curve_neg_std = np.nanmean(delta_stds[-1])
    
    print(curve_neg_std)
    print(min(delta_stds[-1]),max(delta_stds[-1]))

    ax2.hist(delta_stds[-1], 10, density=True, facecolor='g', alpha=0.75, range=(0,0.02))
    ax2.set_title('Curve (negative) std mean = %.4f '% curve_neg_std)
    
    fig.suptitle(window)
    fig.savefig('imgs/hist%d.png' % window)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3),sharey=True)
    
    ys1 = np.nanmean(delta_points[1],axis=0)
    std1 = np.nanstd(delta_points[1],axis=0)
    
    xs = list(range(2*window))
    ax1.errorbar(xs,ys1,yerr=std1)
    ax1.scatter(xs,ys1)
    ax1.set_title('Curve (positive)')
    
    ys2 = np.nanmean(delta_points[-1],axis=0)
    std2 = np.nanstd(delta_points[-1],axis=0)
    
    ax2.errorbar(xs,ys2,yerr=std2)
    ax2.scatter(xs,ys2)
    ax2.set_title('Curve (negative)')
    fig.savefig('imgs/hist_curve%d.png' % window)

