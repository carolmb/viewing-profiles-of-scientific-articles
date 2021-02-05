import read_file
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from collections import defaultdict

from matplotlib.ticker import ScalarFormatter


def get_all_data():
    data = read_file.load_data()
    data = read_file.filter_outliers(data)
    # freq1 = defaultdict(lambda:[])
    freq = []
    for sample in data:
        N = len(sample[1])
        delta_t = sample[3][-1] - sample[3][0]
        # freq1[N].append((delta_t,sample[-2][-1]))
        views = sample[-2][-1]
        if views == 0:
            continue
        freq.append((delta_t, views))
    return freq


def get_data_by_number_segm():
    data = read_file.load_data()
    data = read_file.filter_outliers(data)
    freq_delta_t = defaultdict(lambda: [])
    freq_views = defaultdict(lambda: [])
    for sample in data:
        N = len(sample[1])
        delta_t = sample[3][-1] - sample[3][0]
        views = sample[-2][-1]
        freq_delta_t[N].append(delta_t)
        freq_views[N].append(views)

    return freq_delta_t, freq_views


def plot_lifetime_views(freq):
    # pearson
    v = np.asarray(freq)
    X = v[:, 0]
    Y = v[:, 1]
    p = pearsonr(X, Y)[0]
    print(p)

    plt.figure(figsize=(5, 4))
    df = pd.DataFrame({'lifetime': X, 'views': Y})
    g = sns.jointplot(x="lifetime", y="views", data=df, kind="kde");

    g.ax_joint.set_xlabel('Lifetime', fontsize=16)
    g.ax_joint.set_ylabel('Views', fontsize=16)
    # g.ax_joint.set_xscale('log')

    g.ax_joint.set_xticklabels(g.ax_joint.get_xticks(), size=12)
    g.ax_joint.set_yticklabels(g.ax_joint.get_yticks(), size=12)
    g.ax_joint.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    g.ax_joint.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    g.ax_joint.set_yscale('log')

    plt.savefig('lifetime_views.pdf')


def plot_lifetime_by_number_segm(freq):
    for k, v in freq.items():
        mean = np.mean(v)
        std = np.std(v)

        plt.figure(figsize=(5, 3))
        n = plt.hist(v)
        plt.text(9.5, 0.78 * max(n[0]), "$\mu$ = %.2f\n$\sigma$ = %.2f" % (mean, std), fontsize=16,
                 bbox=dict(edgecolor='black', facecolor='none', alpha=0.5))
        plt.xlim(0, 13)
        plt.xlabel('Lifetime', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig('dist_delta_t_by_breaks_%d.pdf' % k)


def plot_lifetime_all_curves(freq):
    all_v = []
    for k, v in freq.items():
        all_v += v

    plt.figure()
    bins = np.linspace(np.floor(min(all_v)), np.ceil(max(all_v)), 10)
    plt.hist(all_v, bins=bins)
    print(min(all_v), max(all_v))
    plt.xlabel('Lifetime', fontsize=16)
    plt.ylabel('Number of view profiles', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig('lifetime_v3.pdf')


if __name__ == '__main__':
    freq_delta_t, freq_views = get_data_by_number_segm()
    # plot_lifetime_by_number_segm(freq_delta_t)
    plot_lifetime_all_curves(freq_delta_t)