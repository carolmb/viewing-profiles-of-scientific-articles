import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from read_file import select_original_breakpoints
from clusters import norm


def get_colors(labels):
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']

    labels_min = min(len(labels) * 0.10, 70)
    unique, count = np.unique(labels, return_counts=True)
    invalid = []
    for u, c in zip(unique, count):
        if c <= labels_min:
            invalid.append(u)
    labels = [-1 if l in invalid else l for l in labels]

    unique = np.unique(labels).tolist()

    new_labels = []
    for label in labels:
        if label == -1:
            a = '#dcdcdc'
        else:
            idx = unique.index(label)
            a = colors[idx]
        new_labels.append(a)

    return new_labels


def plot_pca(X, colors, filename):
    pca = PCA(n_components=3)
    pca.fit(X)
    X1 = pca.transform(X)
    x_exp, y_exp, z_exp = pca.explained_variance_ratio_[:3] * 100

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X1[:, 0], X1[:, 1], X1[:, 2], c=colors, s=0.4)

    x_exp, y_exp, z_exp = pca.explained_variance_ratio_[:3] * 100
    ax.set_xlabel('PCA1 %.2f' % x_exp)
    ax.set_ylabel('PCA2 %.2f' % y_exp)
    ax.set_zlabel('PCA3 %.2f' % z_exp)

    plt.savefig(filename)
    plt.clf()


if __name__ == '__main__':

    sys.setrecursionlimit(100000)
    f_input = 'segm/segmented_curves_html.txt'
    f_output = 'data/html_by_cat/'
    for N in [2, 3, 4, 5]:
        print(N)
        slopes, intervals = select_original_breakpoints(N, f_input)
        data = np.concatenate((slopes, intervals), axis=1)
        data = norm(data)
        print(data.shape)

        Z = linkage(data[:50000], method='single')

        print('Z complete')

        # plt.figure(figsize=(25, 15))
        # dn = dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
        # plt.ylabel('distance', fontsize=18)
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # plt.savefig(f_output + 'dendrogram_%d.pdf' % N)
        # plt.close()

        for i in np.linspace(0.2, 1, 10):
            labels = fcluster(Z, i, criterion='distance')
            filename = 'k%s/clusters_ind_single_%.2f_%d.txt' % (N, i, N)
            np.savetxt(f_output+filename, labels, delimiter=',')
            colors = get_colors(labels)
            plot_pca(data[:50000], colors, f_output + 'pca_clusters_single_%.2f_%d.pdf' % (i, N))
