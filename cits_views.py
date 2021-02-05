import json
from collections import defaultdict

import read_file
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr


def plot_views_cits():
    input_cits = open('data/series_cits.json', 'r')
    series_cits = json.load(input_cits)
    months_cits = series_cits['months']
    months_cits = [int(m.split('-')[0]) + int(m.split('-')[1]) / 12 for m in months_cits]
    data = read_file.load_data()

    X = []
    Y = []
    for i, s, b, xs, ys, p in data:
        try:
            x = ys[-1]
            idx_begin = months_cits.index(xs[0])
            idx_end = months_cits.index(xs[-1])
            y = sum(series_cits['data'][i]['citations'][idx_begin:idx_end + 1])
        except:
            continue
        X.append(x)
        Y.append(y)

    c = pearsonr(X, Y)
    plt.scatter(X, Y, alpha=0.3, s=1)
    plt.xlabel('views')
    plt.ylabel('number of citations')
    plt.title("pearson = %.2f" % c)
    plt.savefig('views_cits.pdf')


def plot_views_cits_corr():
    input_cits = open('data/series_cits.json', 'r')
    series_cits = json.load(input_cits)
    months_cits = series_cits['months']
    months_cits = [int(m.split('-')[0]) + int(m.split('-')[1]) / 12 for m in months_cits]

    data = read_file.load_data()
    data = read_file.filter_outliers(data)

    corrs = []
    for i, _, _, xs, ys, _ in data:
        try:
            idx_begin = months_cits.index(xs[0])
            idx_end = months_cits.index(xs[-1])
            x = series_cits['data'][i]['citations'][idx_begin:idx_end + 1]
            x = np.diff(x)
            y = np.diff(ys)
            if np.count_nonzero(x[0] == x) == len(x) or np.count_nonzero(y[0] == y) == len(y):
                continue

            if np.count_nonzero(x) > len(x) / 2 and np.count_nonzero(y) > len(y) / 2:
                corr = pearsonr(x, y)[0]
                corrs.append(corr)
        except:
            pass

    plt.hist(corrs, bins=100)
    plt.title('correlação entre número de visualizações e número de citações')
    plt.savefig('corr_views_cits.pdf')


def plt_cits(dois, labels, y):
    input_cits = open('data/series_cits.json', 'r')
    series_cits = json.load(input_cits)
    months_cits = series_cits['months']
    idx_end = months_cits.index('2020-5')

    cits = defaultdict(lambda: [])
    for doi, lab in zip(dois, labels):
        try:  # quantos nunca recebem uma citação? quantos pontos foram pro cinza (excluidos)
            publish_date = series_cits['data'][doi]['date'][:-3]
            x = series_cits['data'][doi]['citations'][months_cits.index(publish_date) + 12 * y - 1]
            cits[lab].append(x)
        except:
            pass
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(9, 3))
    for k, v in cits.items():
        print(k, v.count(0))
        v = np.asarray(v)
        if k == -1:
            continue
        bins = np.logspace(0, np.log10(max(v)), 10)
        axes[k].hist(v, label=k, bins=bins, density=True)  # mudar a escala e os bins
        axes[k].legend()
        # axes[k].set_xscale('log')
        # axes[k].set_yscale('log')
    plt.savefig('cits_dist_labels_%dy.pdf' % y)


def plt_views(dois, labels):
    data = read_file.load_data()
    data = read_file.filter_outliers(data)

    views = []
    for i, s, b, xs, ys, p in data:
        if i in dois:
            views.append(ys[-1])
    views = np.asarray(views)

    values = []
    unique = np.unique(labels)
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(9, 3))

    for u in unique:
        if u == -1:
            continue
        axes[u].hist(views[labels == u], label=u, bins=30)
        axes[u].legend()

    plt.savefig('views_dist_labels.pdf')


def cits_breaks_corr(months_cits, data, series_cits):
    for interval in [1, 2, 3, 6]:

        x = []
        y = []
        for sample in data:
            xs = sample[3]
            idx_begin = months_cits.index(xs[0])
            idx_end = months_cits.index(xs[-1])
            cits = None
            try:
                cits = series_cits['data'][sample[0]]['citations'][idx_begin:idx_end + 1]
            except:
                continue
            breakpoints = sample[2]
            xs = xs[1:]
            ys = sample[4][1:]
            time = (xs - min(xs)) / (max(xs) - min(xs))
            ys = (ys - min(ys)) / (max(ys) - min(ys))

            # plt.plot(time, ys)
            # plt.vlines(breakpoints, 0, 1)
            # plt.show()
            # plt.clf()

            index = np.searchsorted(time, breakpoints)

            j = 1
            for i in index:
                c = cits[i - interval - 1:i]
                if len(c) == 0:
                    j += 1
                    continue
                c = c[-1] - c[0]
                if c == 0:
                    j += 1
                    continue
                x.append(sample[1][j])
                y.append(c)
                j += 1
        x = np.asarray(x)
        y = np.asarray(y)
        plt.scatter(x, y, alpha=0.3)
        plt.title("delta intervalo=%d: pearson corr=%.2f" % (interval, pearsonr(x, y)[0]))
        plt.savefig("delta intervalo %d pearson corr %.2f.png" % (interval, pearsonr(x, y)[0]))
        plt.clf()
        # print()


if __name__ == '__main__':
    # plot_views_cits()
    # plot_views_cits_corr()

    sources = ['clusters\\clusters\\clusters_ind_single_0.50_2.txt',
               'clusters\\clusters\\clusters_ind_single_0.35_3.txt',
               'clusters\\clusters\\clusters_ind_single_0.47_4.txt',
               'clusters\\clusters\\clusters_ind_single_0.56_5.txt']
    labels3 = np.loadtxt(sources[1], dtype=np.int)
    unique, count = np.unique(labels3, return_counts=True)
    unique = unique[count >= 10]
    count = count[count >= 10]
    unique_idxs = np.argsort(count)[-3:]
    unique = unique[unique_idxs].tolist()
    labels3 = [unique.index(l) if l in unique else -1 for l in labels3]

    data = read_file.load_data()
    data = read_file.filter_outliers(data)
    # dois = {2: [], 3: [], 4: [], 5: []}
    # slopes = []
    # intervals = []
    # for i, s, b, xs, ys, p in data:
    #     dois[len(s)].append(i)
    #
    # print('1y')
    # plt_cits(dois[3], labels3, 1)
    # print('2y')
    # plt_cits(dois[3], labels3, 2)
    # print('3y')
    # plt_cits(dois[3], labels3, 5)

    # # pegar os labels
    # # ver a distribuição dos grupos kk views cits tweets

    input_cits = open('data/series_cits.json', 'r')
    series_cits = json.load(input_cits)
    months_cits = series_cits['months']
    months_cits = [int(m.split('-')[0]) + int(m.split('-')[1]) / 12 for m in months_cits]

    cits_breaks_corr(months_cits, data, series_cits)
    dois = {2: [], 3: [], 4: [], 5: []}
    for sample in data:
        dois[len(sample[1])].append(sample)

    for segs in np.arange(2, 6):
        print('numero de segmentos', segs)
        labels = np.loadtxt(sources[segs-2], dtype=np.int)
        unique, count = np.unique(labels, return_counts=True)
        unique = unique[count >= 10]
        count = count[count >= 10]
        unique_idxs = np.argsort(count)[-3:]
        unique = unique[unique_idxs].tolist()
        labels = np.asarray([unique.index(l) if l in unique else -1 for l in labels])

        if segs != 2:
            continue
        samples = np.asarray(dois[segs])
        for u in set(labels):
            if u == -1:
                continue
            print(u)
            cits_breaks_corr(months_cits, samples[labels == u], series_cits)
