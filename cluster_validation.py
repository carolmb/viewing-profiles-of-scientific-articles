
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from clusters import read_data_with_label, norm
from read_file import select_original_breakpoints
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree, fcluster
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, silhouette_samples

# colors = ['tab:red','tab:blue','tab:orange','tab:green','tab:grey','tab:olive','tab:cyan','tab:pink','tab:purple']
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']


def plot_pca(X, labels, filename, title):
    pca = PCA(n_components=3)
    pca.fit(X)
    X1 = pca.transform(X)
    x_exp, y_exp, z_exp = pca.explained_variance_ratio_[:3] * 100
    # label_colors = [colors[l] for l in labels]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X1[:, 0], X1[:, 1], X1[:, 2], c=labels, s=0.4)

    x_exp, y_exp, z_exp = pca.explained_variance_ratio_[:3] * 100
    ax.set_xlabel('PCA1 %.2f' % x_exp)
    ax.set_ylabel('PCA2 %.2f' % y_exp)
    ax.set_zlabel('PCA3 %.2f' % z_exp)

    plt.title(title)
    plt.savefig(filename)


def get_labels_clustering_hier(Z, ktotal, criterion):
    plt.figure(figsize=(25, 10))
    dendrogram(Z)

    plt.title("%s limiar = %.2f, k segmentos = %d, critério = %s" % (method, cut_value, ktotal, criterion))
    plt.ylabel('distance', fontsize=18)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=16)
    plt.savefig('clustering_%s_p20_k%d.pdf' % (method, ktotal))
    plt.clf()


#     labels = cut_tree(Z,n_clusters=kclusters)    
#     return [l[0] for l in labels]
#     


def filter_labels(labels):
    labels_len = min(len(labels) * 0.10, 70)
    unique, count = np.unique(labels, return_counts=True)
    invalid = []
    for u, c in zip(unique, count):
        if c <= labels_len:
            invalid.append(u)
    labels = [-1 if l in invalid else l for l in labels]

    unique = np.unique(labels).tolist()

    new_labels = []
    for label in labels:
        if label == -1:
            l = '#dcdcdc'
        else:
            idx = unique.index(label)
            l = colors[idx]
        new_labels.append(l)

    return new_labels


def linkage_aux(data, method):
    Z = linkage(data, method=method)
    return Z


if __name__ == '__main__':

    # data = read_file.load_data()
    # data_consts = dict()
    # for sample in data:
    #         data_consts[sample[0]] = {'view':sample[-2],'month':sample[-3]}

    # for k in [2,3]:
    #         dois = open('k%d/k%d_dois.txt'%(k,k),'r').read().split('\n')
    #         _,labels = read_data_with_label("k%d/k%d_curves_label.txt"%(k,k))

    #         idxs = np.random.randint(0,len(labels),50)

    #         i = 0
    #         for doi,label in zip(dois,labels):

    #                 if i in idxs:
    #                         x = data_consts[doi]['month']
    #                         y = data_consts[doi]['view']
    #                         print(x,y)
    #                         plt.plot(x,y,color=colors[label],alpha=0.5)
    #                 i += 1
    #         plt.title(k)
    #         plt.savefig('samples_k%d.pdf'%k)
    #         plt.clf()

    # ---------------------------------------------------------------------------------
    # CLUSTERING
    # method = 'distance'
    # for ktotal in [2, 3]:
    #     print(ktotal)
    #     slopes, intervals = select_original_breakpoints(ktotal, 'segm/segmented_curves_filtered.txt')
    #     dois = get_dois(ktotal, 'segm/segmented_curves_filtered.txt')
    #     data = np.concatenate((slopes, intervals), axis=1)
    #     data = norm(data)
    #
    #     sys.setrecursionlimit(100000)
    #     Z = linkage_aux(data, 'single')
    #
    #     for cut_value in [0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]:
    #         single_labels = fcluster(Z, cut_value, criterion=method)
    #
    #         np.savetxt('clusters\\clusters\\custers_ind_single_%.2f_%d.txt' % (cut_value, ktotal), single_labels)
    #
    #         single_labels = filter_labels(single_labels)
    #
    #         title = "%s (limiar=%.2f, segmentos=%d, critério=%s)" % ('single link', cut_value, ktotal, method)
    #         plot_pca(data, single_labels,
    #                  'clusters\\plots\\pca_clusters_single_%.2f_%d.pdf' % (cut_value, ktotal), title)

    # ---------------------------------------------------------------------------------
    # PLOT 3D
    # for k in [2,3,4,5]:
    #         data,labels = read_data_with_label("k%d/k%d_curves_label.txt"%(k,k))
    #         X = norm(data)
    #         plot_pca(X,labels,colors) #,'imgs/pca_%s_%d'%('hierar',k))

    # ---------------------------------------------------------------------------------
    # PLOT DAS CONSTANTES DE CADA CURVA

    # data = read_file.load_data()
    # print(data[0][-2])
    # print((data[0][-2]-min(data[0][-2]))/(max(data[0][-2])-min(data[0][-2])))
    # data_consts = dict()
    # for sample in data:
    #         month = min(sample[-3]),max(sample[-3])
    #         view = min(sample[-2]),max(sample[-2])
    #         data_consts[sample[0]] = {'view':view,'month':month}

    # for k in [2,3,4,5]:
    #         dois = open('k%d/k%d_dois.txt'%(k,k),'r').read().split('\n')
    #         _,labels = read_data_with_label("k%d/k%d_curves_label.txt"%(k,k))
    #         X = defaultdict(lambda:[])
    #         Y = defaultdict(lambda:[])

    #         for doi,label in zip(dois,labels):
    #                 current_doi = data_consts[doi]
    #                 x = current_doi['month'][1] - current_doi['month'][0]
    #                 X[label].append(x)
    #                 y = current_doi['view'][1]
    #                 Y[label].append(y)

    #         for label1 in X.keys():
    #                 for label2 in X.keys():
    #                         if label1 != label2:
    #                                 plt.scatter(X[label2],Y[label2],alpha=0.4,s=0.7,label=label,c='gray')
    #                 plt.scatter(X[label1],Y[label1],alpha=0.5,s=0.7,label=label,c='red')

    #                 plt.legend(markerscale=5)
    #                 plt.savefig('abs_k%d_label%d.pdf'%(k,label1))
    #                 plt.clf()

    # ---------------------------------------------------------------------------------
    # MEDIDAS DE QUALIDADE DE CLUSTER
    sources = ['clusters\\clusters\\clusters_ind_single_0.50_2.txt',
               'clusters\\clusters\\clusters_ind_single_0.35_3.txt',
               'clusters\\clusters\\clusters_ind_single_0.47_4.txt',
               'clusters\\clusters\\clusters_ind_single_0.56_5.txt']
    for k, source in enumerate(sources):
        if k == 0:
            continue
        slopes, intervals = select_original_breakpoints(k + 2, 'segm/segmented_curves_filtered.txt')
        data = np.concatenate((slopes, intervals), axis=1)
        X = norm(data)

        labels = np.loadtxt(source, dtype=np.int)
        unique, count = np.unique(labels, return_counts=True)
        unique = unique[count >= 10]
        count = count[count >= 10]
        unique_idxs = np.argsort(count)[-3:]
        unique = unique[unique_idxs].tolist()
        labels = [unique.index(l) if l in unique else -1 for l in labels]

        selected_samples = []
        selected_labels = []
        for sample, label in zip(X, labels):
            if label != -1:
                selected_samples.append(sample)
                selected_labels.append(label)
        selected_labels = np.asarray(selected_labels)
        selected_samples = np.asarray(selected_samples)

        print("-> k = %d" % (k + 2))
        print("\tsilhouette score = %.2f" % silhouette_score(selected_samples, selected_labels))
        print("\tcalinski harabasz score = %.2f" % calinski_harabasz_score(selected_samples, selected_labels))
        print("\tdavies bouldin score = %.2f" % davies_bouldin_score(selected_samples, selected_labels))
        # print("\tdunn index = %.2f" % base.dunn_fast(X, labels))

        indexes = np.arange(len(selected_samples))
        np.random.shuffle(indexes)
        X_shuffled = selected_samples[indexes]

        print("-> after shuffle k = %d" % (k + 2))
        print("\tsilhouette score = %.2f" % silhouette_score(X_shuffled, selected_labels))
        print("\tcalinski harabasz score = %.2f" % calinski_harabasz_score(X_shuffled, selected_labels))
        print("\tdavies bouldin score = %.2f" % davies_bouldin_score(X_shuffled, selected_labels))
        # print("\tdunn index = %.2f" % base.dunn_fast(X_shuffled, labels))

        print()

'''

-> k = 3
	silhouette score = 0.48
	calinski harabasz score = 4608.23
	davies bouldin score = 0.68
-> after shuffle k = 3
	silhouette score = -0.01
	calinski harabasz score = 1.06
	davies bouldin score = 43.84


-> k = 4
	silhouette score = -0.02
	calinski harabasz score = 2890.65
	davies bouldin score = 1.47
-> after shuffle k = 4
	silhouette score = -0.01
	calinski harabasz score = 0.39
	davies bouldin score = 73.69


-> k = 5
	silhouette score = 0.22
	calinski harabasz score = 3376.62
	davies bouldin score = 1.40
-> after shuffle k = 5
	silhouette score = -0.00
	calinski harabasz score = 0.83
	davies bouldin score = 91.79
	
2 descartados 20 
3 descartados 1639  (utilizados 143 + 804 + 2677])
4 descartados 14457 (utilizados 90 + 1566 + 14770)
5 descartados 69120 (utilizados 661 + 3045,  7970)
'''
