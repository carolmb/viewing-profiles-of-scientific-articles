import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
from artificial_data import calculate_cond_xi_xi1
from read_file import read_original_breakpoints

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

def calculate_hists(n,name,intervalsx,intervalsy,*values):
    hists = []
    for i in range(n-1):
        values_i = values[0][:,i:i+2]
        if len(values) > 1:
            values_i = np.concatenate((values[0][:,i:i+1],values[1][:,i+1:i+2]),axis=1)
        hist = calculate_cond_xi_xi1(values_i,intervalsx,intervalsy)

        # hist = np.zeros((len(intervalsy),len(intervalsx)))
        # for j,ps in prob_xi_xi1.items():
        #     for k,p in ps.items():
        #         hist[j][k] = p

        hists.append((hist,intervalsy,intervalsx))
    return name,hists

def calculate_all_hists(slopes,intervals,n,intervalsx,intervalsy):
    hists_x = calculate_hists(n,'slopes',intervalsx,intervalsx,slopes)
    hists_y = calculate_hists(n,'intervals',intervalsy,intervalsy,intervals)
    hists_xy = calculate_hists(n,'slopes_intervals',intervalsx,intervalsy,slopes,intervals)
    hists_yx = calculate_hists(n,'intervals_slopes',intervalsy,intervalsx,intervals,slopes)
    return hists_x,hists_y,hists_xy,hists_yx

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