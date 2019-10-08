import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def get_abs_diff(slopes):
    last = slopes[0]
    max_diff = 0
    for slope in slopes[1:]:
        diff = slope - last
        last = slope
        max_diff = max(max_diff,abs(diff))
    return max_diff

def jumps(data,N,reverse):
    changes = {2:[],3:[],4:[],5:[]}
    for sample in data:
        slopes = sample[1]
        n = len(slopes)
        changes[n].append((get_abs_diff(slopes),sample))

    rankings = {2:[],3:[],4:[],5:[]}
    for k,diffs in changes.items():
        ranking = sorted(diffs,reverse=reverse,key=lambda s:s[0])[:N]
        for diff,sample in ranking:
            rankings[k].append(sample)
    return rankings

def norm(x):
    return (x-min(x))/(max(x)-min(x))

def plot(xs,ys,color):
    plt.scatter(xs,ys,color=color)

def plot_breakpoints(X,Y,b_points):
    Y_min = min(Y)
    Y_max = max(Y)

    plt.vlines(b_points,ymin=Y_min,ymax=Y_max,color='blue')
    n = len(X)

    print(b_points)

    for b in b_points:
        for i in range(n-1):
            if X[i] <= b and X[i+1] > b:
                pos = i
                break
            if X[i] >= b_points[-1]:
                pos = i
                break

        windows = [X[max(0,pos-3)],X[min(pos+3,n-1)]]
        plt.vlines(windows,ymin=Y_min,ymax=Y_max,color='gray',linestyle='--')

def plot_original_vs_pred(X,Y,Y_pred,b_points,n_intervals,idx,header):
    print(len(X),len(Y_pred),len(Y))
    plot(X,Y_pred,'red')
    plot(X,Y,'green')
    
    plot_breakpoints(X,Y,b_points)
    
    plt.savefig(header+str(n_intervals)+'_'+str(idx)+'.png')
    # plt.show()
    plt.clf()

def plot_diff_original_vs_pred(X,Y,Y_pred,b_points,n_intervals,idx,header):
    plt.plot(X,Y-Y_pred,color='red')
    
    plot_breakpoints(X,Y-Y_pred,b_points)
    
    plt.savefig(header+str(n_intervals)+'_'+str(idx)+'_diff.png')
    # plt.show()
    plt.clf()

def plot_jumps(data,is_norm,reverse,header):
    rankings = jumps(data,10,reverse)
    
    for n_intervals,ranking in rankings.items():
        for sample in ranking:
            idx = sample[0]
            idx = idx.replace('/','_')

            X = sample[3][1:]
            Y = sample[4][1:]
            if is_norm:
                X = norm(X)
                Y = norm(Y)

            Y_pred = sample[-1]
            b_points = [min(X)] + sample[2].tolist()

            plot_original_vs_pred(X,Y,Y_pred,b_points,n_intervals,idx,header)

            plot_diff_original_vs_pred(X,Y,Y_pred,b_points,n_intervals,idx,header)
