# -*- coding: utf-8 -*-
# !/usr/bin/env python

import read_file
import sys, getopt
import numpy as np
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from matplotlib.ticker import StrMethodFormatter
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from read_file import select_original_breakpoints, read_original_breakpoints

import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def multivariateGrid(col_x, col_y, col_k, df, xlabel, ylabel, k_is_color=False, scatter_alpha=.5):
    def colored_scatter(x, y, c=None):
        def scatter(*args, **kwargs):
            if c is not None:
                kwargs['c'] = c

            args = (x, y)

            kwargs['alpha'] = scatter_alpha
            plt.scatter(*args, **kwargs, rasterized=True, s=0.9)

        return scatter

    g = sns.JointGrid(
        x=col_x,
        y=col_y,
        data=df
    )
    grey_rows = df[df[col_k] == 'tab:grey']
    legends = ['tab:grey']
    if k_is_color:
        color = 'tab:grey'
    g.plot_joint(
        colored_scatter(grey_rows[col_x], grey_rows[col_y], color),
    )
    ax1 = sns.distplot(
        grey_rows[col_x].values,
        ax=g.ax_marg_x,
        hist=False,
        color=color,
        kde_kws={"shade": True},
        rug_kws={"rasterized": True}
    )
    ax2 = sns.distplot(
        grey_rows[col_y].values,
        ax=g.ax_marg_y,
        hist=False,
        color=color,
        vertical=True,
        kde_kws={"shade": True},
        rug_kws={"rasterized": True}
    )

    for name, df_group in df.groupby(col_k, sort=True):
        if name == 'tab:grey':
            continue
        legends.append(name)
        if k_is_color:
            color = name
        g.plot_joint(
            colored_scatter(df_group[col_x], df_group[col_y], color),
        )
        ax1 = sns.distplot(
            df_group[col_x].values,
            ax=g.ax_marg_x,
            hist=False,
            color=color,
            kde_kws={"shade": True},
            rug_kws={"rasterized": True}
        )
        ax2 = sns.distplot(
            df_group[col_y].values,
            ax=g.ax_marg_y,
            hist=False,
            color=color,
            vertical=True,
            kde_kws={"shade": True},
            rug_kws={"rasterized": True}
        )

    g.ax_joint.set_xlabel(xlabel, fontsize=20)
    g.ax_joint.set_ylabel(ylabel, fontsize=20)
    g.ax_joint.set_xticklabels(g.ax_joint.get_xticks(), size=16, rotation=45)
    g.ax_joint.set_yticklabels(g.ax_joint.get_yticks(), size=16)
    g.ax_joint.xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    g.ax_joint.yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.tight_layout()


def plot_pca(X, labels, colors, filename):
    pca = PCA(n_components=3)
    pca.fit(X)
    X1 = pca.transform(X)
    x_exp, y_exp, z_exp = pca.explained_variance_ratio_[:3] * 100
    label_colors = [colors[l] for l in labels]
    print(np.unique(label_colors, return_counts=True))
    fig = plt.figure(figsize=(5, 5))
    df = pd.DataFrame({'x': X1[:, 0], 'y': X1[:, 2], 'type': label_colors})
    multivariateGrid('x', 'y', 'type', df, 'PCA1 (%.2f%%)' % x_exp, 'PCA3 (%.2f%%)' % z_exp,
                     k_is_color=True, scatter_alpha=0.5)
    plt.savefig(filename + "_pca1_pca3.pdf")
    plt.clf()

    pca = PCA(n_components=2)
    pca.fit(X)
    X1 = pca.transform(X)
    x_exp, y_exp = pca.explained_variance_ratio_[:2] * 100
    fig = plt.figure(figsize=(5, 5))
    df = pd.DataFrame({'x': X1[:, 0], 'y': X1[:, 1], 'type': label_colors})
    multivariateGrid('x', 'y', 'type', df, 'PCA1 (%.2f%%)' % x_exp, 'PCA2 (%.2f%%)' % y_exp, k_is_color=True,
                     scatter_alpha=.5)
    plt.savefig(filename + '_pca1_pca2.pdf')
    plt.clf()


def norm(data):
    m = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - m) / std


def average_curve_point(data):
    n = data.shape[1] // 2

    mtx_inter = np.zeros((len(data), 100), dtype=np.float)
    for idx_j, sample in enumerate(data):
        degrees = sample[:n]
        intervals = sample[n:]
        cumsum = np.cumsum(intervals)
        for idx_i, i in enumerate(np.linspace(0, 1, 100)):
            y = 0
            last_delta_x = 0
            for degree, cum_x, delta_x in zip(degrees, cumsum, intervals):
                tan_x = np.tan(degree * np.pi / 180)
                if i < cum_x:
                    x = i - last_delta_x
                    y += x * tan_x
                    break
                y += delta_x * tan_x
                last_delta_x = cum_x
            mtx_inter[idx_j][idx_i] = y

    mean = np.mean(mtx_inter, axis=0)
    std = np.std(mtx_inter, axis=0)

    output = (np.linspace(0, 1, 100), mean, np.zeros(100), std)

    return output


def plot_clusters(plots, output):
    extra_file = open(output[:-4] + '_xs_ys.txt', 'w')

    plt.figure(figsize=(6, 3))
    for color, (x0, y0, s0, s1) in plots.items():
        print(color)
        print(x0.shape, y0.shape, s0.shape, s1.shape)
        # plt.errorbar(x0, y0, xerr=s0, yerr=s1, marker='o', markersize=0.7, color=color, linestyle='-', alpha=0.7)
        plt.errorbar(x0, y0, xerr=None, yerr=None, marker='o', markersize=0.7, color=color, linestyle='-', alpha=0.7)
        extra_file.write('-1\n')
        extra_file.write(','.join([str(m) for m in x0]) + '\n')
        extra_file.write(','.join([str(m) for m in y0]) + '\n')

    extra_file.close()

    plt.xlabel('time', fontsize=16)
    plt.ylabel('views', fontsize=16)
    plt.tight_layout()
    print(output)
    plt.savefig(output)
    plt.clf()


def get_labels_clustering_hier(X, method, cut_value, ktotal, output):
    Z = linkage(X, method=method)
    clusters = fcluster(Z, cut_value, criterion='distance')

    # dn = dendrogram(Z, leaf_rotation=90., leaf_font_size=8., truncate_mode='lastp', p=20, show_contracted=True)

    plt.figure(figsize=(25, 15))
    plt.ylabel('distance', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('%sclustering_%s_p20_k_%.2f_%d.pdf' % (output, method, cut_value, ktotal))
    plt.clf()

    return clusters


def save_groups(dois, labels, data, output):
    labels = np.asarray(labels).reshape(-1, 1)
    data = np.concatenate((data, labels), axis=1)
    np.savetxt(output + 'curves_label.txt', data, delimiter=' ', fmt='%.18e')

    file = open(output + '_dois.txt', 'w')
    for doi in dois:
        file.write(doi + '\n')
    file.close()


def get_dois(n, filename):
    data = read_file.load_data(filename)
    data = read_file.filter_outliers(data)

    dois = []
    for i, s, b, xs, ys, p in data:
        if len(s) == n:
            dois.append(i)
    return dois


def generate_groups(get_labels, f_input='segm/segmented_curves_filtered.txt', output=''):
    for cut_value, ktotal in [(0.7, 2), (0.35, 3), (0.4, 4), (0.56, 5)]:
        slopes, intervals = select_original_breakpoints(ktotal, f_input)
        dois = get_dois(ktotal, f_input)
        data = np.concatenate((slopes, intervals), axis=1)
        data = norm(data)

        labels = get_labels(data, 'single', cut_value, ktotal, output)
        save_groups(dois, labels, data, output + 'k' + str(ktotal) + '/k' + str(ktotal))


def read_args():
    argv = sys.argv[1:]

    op1, op2, op3, op4 = False, False, False, False
    N = -1
    try:
        opts, args = getopt.getopt(argv, "N:", ['op1', 'op2', 'op3', 'op4'])
    except getopt.GetoptError:
        print('usage: python example.py --op1 --op2 --op3 --op4')
        return None

    for opt, arg in opts:
        if opt == '-N':
            N = arg
        if opt == '--op1':
            op1 = True
        if opt == '--op2':
            op2 = True
        if opt == '--op3':
            op3 = True
        if opt == '--op4':
            op4 = True

    return op1, op2, op3, op4, int(N)


def read_preprocessed_file(N, source):
    dois = open('k' + str(N) + '/k' + str(N) + '_dois.txt', 'r').read().split()

    original = np.loadtxt(source)
    slopes = original[:, :N]
    intervals = original[:, N:-1]
    labels = original[:, -1]
    labels = labels.astype(int)
    total_labels = len(set(labels))

    data = read_file.load_data()
    valid_dois = set()
    for i, s, b, xs, ys, p in data:
        # print(len(s))
        delta_x = xs[-1] - xs[0]
        if 5 <= delta_x <= 7:
            valid_dois.add(i)

    labels_slopes, labels_intervals = [[] for _ in range(total_labels)], [[] for _ in range(total_labels)]
    for doi, label, s, l in zip(dois, labels, slopes, intervals):
        if doi in valid_dois:
            labels_slopes[label].append(s)
            labels_intervals[label].append(l)

    labels_slopes = [np.asarray(values) for values in labels_slopes]
    labels_intervals = [np.asarray(values) for values in labels_intervals]
    return labels_slopes, labels_intervals


def read_data_with_label(source):
    data = np.loadtxt(source)
    original = data[:, :-1]
    labels = data[:, -1]
    labels = labels.astype(int)
    return original, labels


colors = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'tab:grey']

if __name__ == '__main__':
    op1, op2, op3, op4, N = read_args()
    print(op1, op2, op3, N)

    if op1:
        generate_groups(get_labels_clustering_hier)
    if op2:
        sources = ['clusters\\clusters\\clusters_ind_single_0.50_2.txt',
                   'clusters\\clusters\\clusters_ind_single_0.35_3.txt',
                   'clusters\\clusters\\clusters_ind_single_0.47_4.txt',
                   'clusters\\clusters\\clusters_ind_single_0.56_5.txt']
        for k, source in zip([2, 3, 4, 5], sources):
            # data, labels = read_data_with_label("k%d/k%d_curves_label.txt"%(k,k))
            slopes, intervals = select_original_breakpoints(k, 'segm/segmented_curves_filtered.txt')
            data = np.concatenate((slopes, intervals), axis=1)
            X = norm(data)

            labels = np.loadtxt(source, dtype=np.int)
            unique, counts = np.unique(labels, return_counts=True)
            unique = unique[counts >= 10]
            counts = counts[counts >= 10]
            unique_idxs = np.argsort(counts)[-3:]
            unique = unique[unique_idxs].tolist()
            print(unique)
            labels = [unique.index(l) if l in unique else -1 for l in labels]

            print(np.unique(labels, return_counts=True))
            # plot_pca(X, labels, colors, 'imgs/pca_%s_%d' % ('hierar', k))
    if op3:
        # source = 'k%d/k%d_curves_label.txt' % (N,N)
        # # já são os artigos filtrados 5 a 7 anos
        # labels_slopes, labels_intervals = read_preprocessed_file(N, source)
        sources = ['clusters\\clusters\\clusters_ind_single_0.50_2.txt',
                   'clusters\\clusters\\clusters_ind_single_0.35_3.txt',
                   'clusters\\clusters\\clusters_ind_single_0.47_4.txt',
                   'clusters\\clusters\\clusters_ind_single_0.56_5.txt']
        for N, source in zip([2, 3, 4, 5], sources):
            labels = np.loadtxt(source, dtype=np.int)
            slopes, intervals = select_original_breakpoints(N, 'segm/segmented_curves_filtered.txt')
            unique, counts = np.unique(labels, return_counts=True)
            unique = unique[counts >= 10]
            counts = counts[counts >= 10]
            unique_idxs = np.argsort(counts)[-3:]
            unique = unique[unique_idxs]
            output = 'k%d/average_curves_label_k%d_2.pdf' % (N, N)
            plots = dict()
            for i, label in enumerate(unique):
                idxs = labels == label
                print(N, label, colors[i])
                data = np.concatenate((slopes[idxs], intervals[idxs]), axis=1)
                average = average_curve_point(data)
                plots[colors[i]] = average

            # for label, (slopes, intervals) in enumerate(zip(labels_slopes, labels_intervals)):
            #     print(slopes.shape, intervals.shape)
            #     data = np.concatenate((slopes, intervals), axis=1)
            #     average = average_curve_point(data)
            #     plots[colors[label]] = average

            plot_clusters(plots, output)
    if op4:
        _, slopes, breakpoints, preds = read_original_breakpoints('segm/k%d_test.txt' % N, None)
        original = open('k%d/average_curves_label_k%d_xs_ys.txt' % (N, N), 'r').read().split('\n')
        original = [[float(m) for m in original[i + 2].split(',')] for i in range(0, len(original) - 1, 3)]
        x = np.linspace(0, 1, 100)
        i = 0
        for s, b, p, y in zip(slopes, breakpoints, preds, original):
            print(s)
            plt.plot(x, p, alpha=0.7, c=colors[i])
            plt.scatter(x, y, alpha=0.8, s=2, c=colors[i])
            i += 1
        plt.xlabel('Time', fontsize=18)
        plt.ylabel('Views', fontsize=18)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig("k%d/average_curve_k%d.pdf" % (N, N))
