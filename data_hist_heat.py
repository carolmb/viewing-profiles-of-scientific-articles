import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D

def calculate_cond_xi_xi1(X,intervalsx):
    prob_xi = defaultdict(lambda:0)
    prob_xi1 = defaultdict(lambda:defaultdict(lambda:0))
    
    for x in X:
        for j in range(len(intervalsx)-1):
            if x[0] >= intervalsx[j] and x[0] < intervalsx[j+1]:
                prob_xi[j] += 1
                for k in range(len(intervalsx)-1):
                    if x[1] > intervalsx[k] and x[1] <= intervalsx[k+1]:
                        prob_xi1[j][k] += 1
    for i,prob in prob_xi1.items():
        total = prob_xi[i]
        for j in prob.keys():
            prob[j] = prob[j]/total
    return prob_xi1

def plot_heat(ax,hist,xedges,yedges,labelx,labely):
    nx = len(xedges)
    ny = len(yedges)

    ax.imshow(hist, cmap='hot', interpolation='nearest',extent=(0,nx,ny,0))
    ax.set_xticks(range(nx))
    ax.tick_params(axis='both',labelsize=6)
    ax.set_xticklabels([str(f)[:4] for f in xedges],rotation=60)
    ax.set_yticks(range(ny))
    ax.set_yticklabels([str(f)[:4] for f in yedges])
    ax.set_xlabel(labelx)
    ax.set_ylabel(labely)
 
def plot_hist(ax,hist,xedges,yedges,labelx,labely):
    xpos, ypos = np.meshgrid(xedges, yedges, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    dx = (max(xedges)-min(xedges))/len(xedges)
    dy = (max(yedges)-min(yedges))/len(yedges)
    dz = hist.ravel()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average',edgecolors='white')

    ax.set_xlabel(labelx)
    ax.set_ylabel(labely)

def calculate_hists(n,name,intervalsx,*values):
    hists = []
    for i in range(n-1):
        values_i = values[0][:,i:i+2]
        prob_xi_xi1 = calculate_cond_xi_xi1(values_i,intervalsx)

        hist = np.zeros((len(intervalsx),len(intervalsx)))
        for j,ps in prob_xi_xi1.items():
            for k,p in ps.items():
                hist[j][k] = p

        hists.append((hist,intervalsx,intervalsx))
    return name,hists

# TODO
def get_i(V,intervals):
    idxs = []
    for v in V:
        for k in range(len(intervals)-1):
            if v > intervals[k] and v <= intervals[k+1]:
                idxs.append(k)
                break
    idxs = np.asarray(idxs)
    return idxs

# TODO
def plot_intervals_slopes(slopes,intervals,n):
    # PARA HISTOGRAMA DO SLOPE POR INTERVAL
    fig_hist = plt.figure(figsize=(5,20))
    fig_heat = plt.figure(figsize=(5,20))
    for i in range(n):
        slopes_i = slopes[:,i]
        intervals_i = intervals[:,i]
        slopes_i = get_i(slopes_i,intervalsx)
        intervals_i = get_i(intervals_i,intervalsy)
        
        hist = np.zeros((len(intervalsx),len(intervalsy)))
        for x,y in zip(slopes_i,intervals_i):
            hist[x][y] += 1
        print(hist)
        sub_plot_hist(hist,intervalsy,intervalsx,'intervals'+str(i),'slopes'+str(i),nx,ny,fig_hist,fig_heat,i+1,deltay,deltax)
    fig_hist.tight_layout()
    fig_hist.savefig('slope_interval_hist.pdf',format='pdf')
    fig_heat.tight_layout()
    fig_heat.savefig('slope_interval_heat.pdf',format='pdf')

def calculate_all_hists(slopes,intervals,n,intervalsx,intervalsy):
    hists_x = calculate_hists(n,'slopes',intervalsx,slopes)
    hists_y = calculate_hists(n,'intervals',intervalsy,intervals)
    return hists_x,hists_y

def plot_hists(fig,name,hists,header,plot_func):
    n = len(hists)
    for i,(hist,xedges,yedges) in enumerate(hists):
        ax = fig.add_subplot(n,1,i+1, projection='3d')
        plot_hist(ax,hist,xedges,yedges,'x+1','x')
    fig.suptitle(name)
    fig.savefig(header+name+"_"+str(n+1)+".pdf",format='pdf',bbox_inches='tight')

def plot_heats(fig,name,hists,header,plot_func):
    n = len(hists)
    for i,(hist,xedges,yedges) in enumerate(hists):
        ax = fig.add_subplot(n,1,i+1)
        plot_heat(ax,hist,xedges,yedges,'x+1','x')
    fig.suptitle(name)
    fig.savefig(header+name+"_"+str(n+1)+".pdf",format='pdf',bbox_inches='tight')

def plot_all_hists(header,*args):
    n = len(args[0][1])
    fig = plt.figure(figsize=(6,4*n))
    for (name,hists) in args:
        plot_hists(fig,name+'_hist',hists,header,plot_hist)
        fig.clf()
        plot_heats(fig,name+'_heat',hists,header,plot_heat)
        fig.clf()

def generate_hist_plots(slopes,intervals,n,header,args):
    hists = calculate_all_hists(slopes,intervals,n,*args)
    plot_all_hists(header,*hists)

'''

    fig_slopes.suptitle('slopes')
    fig_slopes.savefig(header+"slopes_"+str(n)+".pdf",format='pdf',bbox_inches='tight')

    fig_slopes.suptitle('intervals')
    fig_intervals.savefig(header+"intervals_"+str(n)+".pdf",format='pdf',bbox_inches='tight')

    fig_slopes_intervals.suptitle('x(i) is slope, y(i+1) is interval')
    fig_slopes_intervals.savefig(header+"slopes_intervals_"+str(n)+".pdf",format='pdf',bbox_inches='tight')

    fig_intervals_slopes.suptitle('x(i) is interval, y(i+1) is slope')
    fig_intervals_slopes.savefig(header+"intervals_slopes_"+str(n)+".pdf",format='pdf',bbox_inches='tight')

    fig_hist_slopes.tight_layout()
    fig_hist_slopes.suptitle('P(x+1|x), x is slope')
    fig_hist_slopes.savefig(header+"hist_slopes_"+str(n)+".pdf",format='pdf',bbox_inches='tight')

    fig_heat_slopes.tight_layout()
    fig_heat_slopes.suptitle('P(x+1|x), x is slope', y=1.08)
    fig_heat_slopes.savefig(header+"heat_slopes_"+str(n)+".pdf",format='pdf',bbox_inches='tight')

    fig_hist_intervals.tight_layout()
    fig_hist_intervals.suptitle('P(x+1|x), x is intervals')
    fig_hist_intervals.savefig(header+"hist_intervals_"+str(n)+".pdf",format='pdf',bbox_inches='tight')

    fig_heat_intervals.tight_layout()
    fig_heat_intervals.suptitle('P(x+1|x), x is intervals', y=1.08)
    fig_heat_intervals.savefig(header+"heat_intervals_"+str(n)+".pdf",format='pdf',bbox_inches='tight')

'''