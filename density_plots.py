import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from matplotlib.ticker import StrMethodFormatter, ScalarFormatter
from read_file import select_original_breakpoints

import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def plot_density(all_data, xlim, ylim, filename):
    i = 0
    for (xlabel, ylabel, corr), data in all_data.items():
        ax1 = sns.jointplot(x=xlabel, y=ylabel, data=data, kind="kde", xlim=xlim, ylim=ylim)
        ax1.set_axis_labels(xlabel, ylabel, fontsize=20)

        ax1.ax_joint.set_xticklabels(ax1.ax_joint.get_xticks(), size=16)
        ax1.ax_joint.set_yticklabels(ax1.ax_joint.get_yticks(), size=16)

        ax1.ax_joint.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax1.ax_joint.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        # ax1.ax_joint.xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
        # ax1.ax_joint.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
        plt.tight_layout()
        plt.savefig(filename + str(i) + '.pdf')
        i += 1


def corr_density_plots(slopes, intervals, n, header):
    to_plot = dict()
    for i in range(n - 1):
        x = slopes[:, i]
        y = slopes[:, i + 1]
        xlabel = '$\\alpha_%d$' % (i + 1)
        ylabel = '$\\alpha_%d$' % (i + 2)
        data = pd.DataFrame({xlabel: x, ylabel: y})

        corr = pearsonr(x, y)[0]
        to_plot[(xlabel, ylabel, corr)] = data
        print(xlabel, ylabel, corr)
    xlim = (0, 90)
    ylim = (0, 90)
    # plot_density(to_plot, xlim, ylim, 'imgs/density_alpha_alpha_')

    to_plot = dict()
    for i in range(n-1):
        x = intervals[:,i]
        y = intervals[:,i+1]
        xlabel = '$l_%d$' % (i+1)
        ylabel = '$l_%d$' % (i+2)
        data = pd.DataFrame({xlabel:x,ylabel:y})

        corr = pearsonr(x,y)[0]
        to_plot[(xlabel,ylabel,corr)] = data
        print(xlabel,ylabel,corr)
    xlim = (0,0.6)
    ylim = (0,0.6)
    plot_density(to_plot,xlim,ylim,'imgs/density_l_l_')

    to_plot = dict()
    for i in range(n-1):
        x = slopes[:,i]
        y = intervals[:,i+1]
        xlabel = '$\\alpha_%d$' % (i+1)
        ylabel = '$l_%d$' % (i+2)
        data = pd.DataFrame({xlabel:x,ylabel:y})

        corr = pearsonr(x,y)[0]
        to_plot[(xlabel,ylabel,corr)] = data
        print(xlabel,ylabel,corr)
    # xlim = (0,90)
    # ylim = (0,0.6)
    # plot_density(to_plot,xlim,ylim,'imgs/density_alpha_l_')

    to_plot = dict()
    for i in range(n-1):
        x = intervals[:,i]
        y = slopes[:,i+1]
        xlabel = '$l_%d$' % (i+1)
        ylabel = '$\\alpha_%d$' % (i+2)
        data = pd.DataFrame({xlabel:x,ylabel:y})

        corr = pearsonr(x,y)[0]
        to_plot[(xlabel,ylabel,corr)] = data
        print(xlabel,ylabel,corr)
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
    # xlim = (0,0.6)
    # ylim = (0,90)
    # plot_density(to_plot,xlim,ylim,'imgs/density_alpha_l_i_')


'''
# HTML VIEWS ONLY
$\alpha_1$ $\alpha_2$ -0.16566577082009307
$\alpha_2$ $\alpha_3$ -0.3899387155851549
$\alpha_3$ $\alpha_4$ -0.3488593898057022
$\alpha_4$ $\alpha_5$ 0.21679917145403074
$l_1$ $l_2$ -0.6219753514408883
$l_2$ $l_3$ -0.47482484862562835
$l_3$ $l_4$ -0.5180720159705602
$l_4$ $l_5$ -0.472258433628872
$\alpha_1$ $l_2$ 0.326069566085871
$\alpha_2$ $l_3$ 0.26837113640909493
$\alpha_3$ $l_4$ 0.40082462467801405
$\alpha_4$ $l_5$ 0.42791098799736504
$l_1$ $\alpha_2$ 0.17462915246271704
$l_2$ $\alpha_3$ 0.35715284176411155
$l_3$ $\alpha_4$ 0.04514592281296922
$l_4$ $\alpha_5$ -0.08265757995920772
$l_1$ $\alpha_1$ -0.5360087910074902
$l_2$ $\alpha_2$ -0.3575724589850012
$l_3$ $\alpha_3$ -0.5564860193258271
$l_4$ $\alpha_4$ -0.16323017713364651
$l_5$ $\alpha_5$ 0.13710343658357016

'''

if __name__ == '__main__':
    f_input = 'segm/segmented_curves_filtered.txt'
    slopes, intervals = select_original_breakpoints(5, f_input)
    corr_density_plots(slopes, intervals, 5, 'teste/density_')
    # generate_hist_plots(slopes,intervals,n,'imgs/original1/',args)
