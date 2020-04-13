import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from read_file import select_original_breakpoints
from sklearn.utils.random import sample_without_replacement

def plot_density(all_data,xlim,ylim,filename):

    i = 0
    for (xlabel,ylabel,corr),data in all_data.items():
        ax1 = sns.jointplot(x=xlabel, y=ylabel, data=data, kind="kde",xlim=xlim,ylim=ylim)
        ax1.set_axis_labels(xlabel,ylabel,fontsize=18)
        plt.savefig(filename+str(i)+'.pdf')   
        i += 1
    
def corr_density_plots(slopes,intervals,n,header):
    # to_plot = dict()
    # for i in range(n-1):
    #     x = slopes[:,i]
    #     y = slopes[:,i+1]
    #     xlabel = '$\\alpha_%d$' % (i+1) 
    #     ylabel = '$\\alpha_%d$' % (i+2)
    #     data = pd.DataFrame({xlabel:x,ylabel:y})
        
    #     corr = pearsonr(x,y)[0]
    #     to_plot[(xlabel,ylabel,corr)] = data
    #     print(xlabel,ylabel,corr)
    # xlim = (0,90)
    # ylim = (0,90)
    # plot_density(to_plot,xlim,ylim,'imgs/density_alpha_alpha_')

    # to_plot = dict()
    # for i in range(n-1):
    #     x = intervals[:,i]
    #     y = intervals[:,i+1]
    #     xlabel = '$l_%d$' % (i+1) 
    #     ylabel = '$l_%d$' % (i+2)
    #     data = pd.DataFrame({xlabel:x,ylabel:y})
        
    #     corr = pearsonr(x,y)[0]
    #     to_plot[(xlabel,ylabel,corr)] = data
    #     print(xlabel,ylabel,corr)
    # xlim = (0,0.6)
    # ylim = (0,0.6)
    # plot_density(to_plot,xlim,ylim,'imgs/density_l_l_')

    # to_plot = dict()
    # for i in range(n-1):
    #     x = slopes[:,i]
    #     y = intervals[:,i+1]
    #     xlabel = '$\\alpha_%d$' % (i+1) 
    #     ylabel = '$l_%d$' % (i+2)
    #     data = pd.DataFrame({xlabel:x,ylabel:y})

    #     corr = pearsonr(x,y)[0]
    #     to_plot[(xlabel,ylabel,corr)] = data
    #     print(xlabel,ylabel,corr)
    # xlim = (0,90)
    # ylim = (0,0.6)
    # plot_density(to_plot,xlim,ylim,'imgs/density_alpha_l_')

    # to_plot = dict()
    # for i in range(n-1):
    #     x = intervals[:,i]
    #     y = slopes[:,i+1]
    #     xlabel = '$l_%d$' % (i+1) 
    #     ylabel = '$\\alpha_%d$' % (i+2)
    #     data = pd.DataFrame({xlabel:x,ylabel:y})

    #     corr = pearsonr(x,y)[0]
    #     to_plot[(xlabel,ylabel,corr)] = data
    #     print(xlabel,ylabel,corr)
    # xlim = (0,0.6)
    # ylim = (0,90)
    # plot_density(to_plot,xlim,ylim,'imgs/density_l_alpha_')

    to_plot = dict()
    for i in range(n):
        x = intervals[:,i]
        y = slopes[:,i]
        xlabel = '$l_%d$' % (i+1) 
        ylabel = '$\\alpha_%d$' % (i+1)
        data = pd.DataFrame({xlabel:x,ylabel:y})

        corr = pearsonr(x,y)[0]
        to_plot[(xlabel,ylabel,corr)] = data
        print(xlabel,ylabel,corr)
    xlim = (0,0.6)
    ylim = (0,90)
    plot_density(to_plot,xlim,ylim,'imgs/density_alpha_l_i_')


slopes,intervals = select_original_breakpoints(5,'segm/segmented_curves_filtered.txt')
corr_density_plots(slopes,intervals,5,'teste/density_')
# generate_hist_plots(slopes,intervals,n,'imgs/original1/',args)
