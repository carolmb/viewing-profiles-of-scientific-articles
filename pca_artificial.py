import sys
import util
import getopt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats
from collections import defaultdict
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from read_file import select_original_breakpoints, read_artificial_breakpoints
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import StrMethodFormatter
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# source0 = modelos sinteticos
# source1 = labels
def get_data(source0, source1, source2 = 'segm\\segmented_curves_filtered.txt', preprocessed = True):

    label_slopes, label_intervals = util.read_preprocessed_file(N, source2, source1)

    slopes_original = []
    intervals_original = []
    for slopes, intervals in zip(label_slopes, label_intervals):
        if slopes_original == []:
            slopes_original = slopes
            intervals_original = intervals
        else:
            slopes_original = np.concatenate((slopes_original, slopes), axis=0)
            intervals_original = np.concatenate((intervals_original, intervals), axis=0)
    # else:
        # slopes_original, intervals_original = select_original_breakpoints(n, source1)

    _, slopes_artificial, intervals_artificial = read_artificial_breakpoints(source0)

    original_data = np.concatenate((slopes_original, intervals_original), axis=1)
    # print(slopes_artificial[:5],intervals_artificial[:5])
    artificial_data = np.concatenate((slopes_artificial, intervals_artificial), axis=1)

    all_data = np.concatenate((original_data, artificial_data), axis=0)
    m = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0)

    all_data_norm = (all_data - m) / std
    original_norm = (original_data - m) / std
    artificial_norm = (artificial_data - m) / std

    return artificial_data, original_data, all_data_norm, artificial_norm, original_norm


def plot_pca(y1, y2, xlabel, ylabel, filename):
    df1 = pd.DataFrame({'x': y1[:, 0], 'y': y1[:, 1], 'type': 'artificial'})
    df2 = pd.DataFrame({'x': y2[:, 0], 'y': y2[:, 1], 'type': 'original'})

    xmin, xmax = min(df1['x'].min(), df2['x'].min()), max(df1['x'].max(), df2['x'].max())
    ymin, ymax = min(df1['y'].min(), df2['y'].min()), max(df1['y'].max(), df2['y'].max())

    ax1 = sns.jointplot(x="x", y="y", data=df1, kind="kde", color='red')
    ax1.ax_marg_x.set_xlim(xmin, xmax)
    ax1.ax_marg_y.set_ylim(ymin, ymax)
    ax1.set_axis_labels(xlabel, ylabel, fontsize=24)
    ax1.ax_joint.set_xticklabels(ax1.ax_joint.get_xticks(), size=20)
    ax1.ax_joint.set_yticklabels(ax1.ax_joint.get_yticks(), size=20)
    ax1.ax_joint.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.ax_joint.xaxis.set_major_locator(MaxNLocator(integer=True))

    # rect = patches.Rectangle((box1[0],box1[2]),box1[1]-box1[0],box1[3]-box1[2],linewidth=1,edgecolor='blue',facecolor='none')
    # ax1.ax_joint.add_patch(rect)

    plt.savefig(filename + 'artificial_pca.pdf', bbox_inches='tight')

    ax2 = sns.jointplot(x="x", y="y", data=df2, kind="kde", color='green')
    ax2.set_axis_labels(xlabel, ylabel, fontsize=24)
    ax2.ax_marg_x.set_xlim(xmin, xmax)
    ax2.ax_marg_y.set_ylim(ymin, ymax)
    ax2.ax_joint.set_xticklabels(ax2.ax_joint.get_xticks(), size=20)
    ax2.ax_joint.set_yticklabels(ax2.ax_joint.get_yticks(), size=20)
    ax2.ax_joint.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.ax_joint.xaxis.set_major_locator(MaxNLocator(integer=True))

    # rect = patches.Rectangle((box2[0],box2[2]),box2[1]-box2[0],box2[3]-box2[2],linewidth=1,edgecolor='blue',facecolor='none')
    # ax2.ax_joint.add_patch(rect)

    plt.savefig(filename + 'original_pca.pdf', bbox_inches='tight')
    return xmin, xmax, ymin, ymax


def get_linspace(syn_trans, original_trans):
    print('syn', syn_trans.shape, original_trans.shape)
    xmin = min(syn_trans[:, 0])
    xmax = max(syn_trans[:, 0])
    ymin = min(syn_trans[:, 1])
    ymax = max(syn_trans[:, 1])

    xmin = min(xmin, min(original_trans[:, 0]))
    xmax = max(xmax, max(original_trans[:, 0]))
    ymin = min(ymin, min(original_trans[:, 1]))
    ymax = max(ymax, max(original_trans[:, 1]))

    X = np.linspace(xmin, xmax, 100)
    Y = np.linspace(ymin, ymax, 100)
    xx, yy = np.meshgrid(X, Y)
    data = np.concatenate((xx.ravel()[:, None], yy.ravel()[:, None]), axis=1)

    return data, xx, yy


def plot_pca_manual(xx, yy, Z, data, filename, title, xlabel, ylabel, cmap, color):
    fig = plt.figure(figsize=(9, 9))
    plt.subplots_adjust(wspace=0, hspace=0)
    gs = gridspec.GridSpec(5, 5)
    ax = plt.subplot(gs[1:, :-1])
    CS = ax.contourf(xx, yy, Z.reshape(100, -1), levels=8, cmap=cmap)
    ax.set_xlabel(xlabel, fontsize=28)
    ax.set_ylabel(ylabel, fontsize=28)
    ax.tick_params(labelsize=24)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.margins(0, 0)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax1 = plt.subplot(gs[1:, -1], sharey=ax)

    sns.distplot(data[:, 1], ax=ax1, vertical=True, color=color, hist=False, axlabel=False, kde_kws={"shade": True})
    # ax1.margins(0,0)
    ax1.set_xticks([])
    ax1.set_xlabel('')
    ax1.label_outer()
    ax1.set_ylim(ylim)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    # X = sum(stats.multivariate_normal(x,cov=0.05).pdf(xx[0]) for x in data[:,0])

    ax2 = plt.subplot(gs[0, :-1], sharex=ax)
    # Y = sum(stats.multivariate_normal(x,cov=0.05).pdf(yy[0]) for x in data[:,1])
    sns.distplot(data[:, 0], ax=ax2, color=color, hist=False, kde_kws={"shade": True})
    ax2.label_outer()
    ax2.set_xlim(xlim)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.set_yticks([])
    ax2.set_ylabel('')

    plt.savefig(filename + title + '_pca.pdf')
    return xlim, ylim


def surface_test(syn, original, syn_trans, original_trans, filename, xlabel, ylabel):
    N = len(syn_trans)
    data, xx, yy = get_linspace(syn_trans, original_trans)

    Z1 = sum(stats.multivariate_normal(xy, cov=0.05).pdf(data) for xy in syn_trans)
    Z2 = sum(stats.multivariate_normal(xy, cov=0.05).pdf(data) for xy in original_trans)

    kde = stats.gaussian_kde(syn_trans.T)
    Z1 = kde.pdf(data.T)

    kde = stats.gaussian_kde(original_trans.T)
    Z2 = kde.pdf(data.T)

    plot_pca_manual(xx, yy, Z1, syn_trans, filename, 'Synthetic', xlabel, ylabel, 'Reds', 'red')
    plot_pca_manual(xx, yy, Z2, original_trans, filename, 'Original', xlabel, ylabel, 'Greens', 'green')

    fig = plt.figure(figsize=(9, 9))
    plt.subplots_adjust(wspace=0, hspace=0)
    gs = gridspec.GridSpec(5, 5)
    ax = plt.subplot(gs[1:, :-1])
    CS = ax.contourf(xx, yy, np.absolute(Z1 - Z2).reshape(100, -1), levels=8, cmap='Blues')
    ax.set_xlabel(xlabel, fontsize=28)
    ax.set_ylabel(ylabel, fontsize=28)
    ax.tick_params(labelsize=24)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.margins(0, 0)

    # plt.xlim(xlim)
    # plt.ylim(ylim)

    total = sum(np.absolute(Z1 - Z2))
    print('total sum diff', total)
    plt.title("$\\epsilon$ = %.2f" % total, fontsize=28)
    # plt.tight_layout()
    colorbar_ax = plt.subplot(gs[1:, -1])
    colorbar_ax.set_frame_on(False)
    colorbar_ax.set_xticks([])
    colorbar_ax.set_xlabel('')
    colorbar_ax.set_yticks([])
    colorbar_ax.set_ylabel('')

    divider = make_axes_locatable(colorbar_ax)
    cax = divider.new_horizontal(size='50%', pad=0.6, pack_start=True)
    fig.add_axes(cax)
    colorbar = fig.colorbar(CS, cax=cax, orientation='vertical')

    # colorbar = fig.colorbar(CS, ax=colorbar_ax, pad=0, anchor=(0, 1), orientation='vertical')
    colorbar.ax.tick_params(labelsize=24)
    colorbar.ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))

    plt.savefig(filename + 'diff_test.pdf')


def get_pca_infos(all_data_norm, artificial_norm, original_norm):
    pca = PCA(n_components=2)
    pca.fit(all_data_norm)

    y1 = pca.transform(artificial_norm)
    y2 = pca.transform(original_norm)

    y1_explained, y2_explained = pca.explained_variance_ratio_[:2]
    y1_explained = y1_explained * 100
    y2_explained = y2_explained * 100

    y1_label = 'PCA1 (%.2f%%)' % y1_explained
    y2_label = 'PCA2 (%.2f%%)' % y2_explained

    return y1, y2, y1_label, y2_label


def get_args_terminal():
    argv = sys.argv[1:]

    source1, source2 = 'segm/segmented_curves_filtered.txt', None
    output = None
    N = 5
    preprocessed = False
    try:
        opts, args = getopt.getopt(argv, "o:N:p", ['s1=', 's2='])
    except getopt.GetoptError:
        print('usage: python example.py -s1 <source1> -s2 <source2> -o <output> -N <n> -p')
        return None

    for opt, arg in opts:

        if opt == '--s1':
            source1 = arg
        if opt == '--s2':
            source2 = arg
        elif opt == '-o':
            output = arg
        elif opt == '-N':
            N = int(arg)
        elif opt == '-p':
            preprocessed = True

    return source1, source2, output, N, preprocessed


if __name__ == "__main__":
    source1, source2, output, N, preprocessed = get_args_terminal()
    print(preprocessed, source1, source2, output)

    label_sources = ['clusters\\clusters\\clusters_ind_single_0.50_2.txt',
                     'clusters\\clusters\\clusters_ind_single_0.35_3.txt',
                     'clusters\\clusters\\clusters_ind_single_0.47_4.txt',
                     'clusters\\clusters\\clusters_ind_single_0.56_5.txt']
    model_sources = ['clusters\\markov1_multi_k', 'clusters\\markov1_uni_k',
                     'clusters\\no_memory_k', 'clusters\\null_model_k']

    for model in model_sources:
        for N, labels in zip([2, 3, 4, 5], label_sources):
            if N != 5:
                continue
            print(N, model)
            artificial_data, original_data, all_data_norm, artificial_norm, original_norm = \
                get_data("%s%d_label.txt" % (model, N), labels)

            y1, y2, y1_label, y2_label = get_pca_infos(all_data_norm, artificial_norm, original_norm)
            surface_test(artificial_data, original_data, y1, y2, "%s%d" % (model, N), y1_label, y2_label)
# plot_pca(y1,y2,y1_label,y2_label,output)

'''
# para todas as curvas com label: 
python pca_artificial.py --s2 k3/markov1_uni_k3_label.txt -N 3 -o k3/

'''
