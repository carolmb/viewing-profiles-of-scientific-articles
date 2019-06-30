import numpy as np
from analise_breakpoints import read_file
from analise_breakpoints import breakpoints2intervals
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import describe

def test_n_clusters(range_n_clusters,X,alg,name):
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
        
        clusterer = alg(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
            "The average silhouette_score is :", silhouette_avg)

        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        pca = PCA(n_components=2)
        pca.fit(X)
        X1 = pca.transform(X)
        ax2.scatter(X1[:, 0], X1[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")
        
        plt.suptitle(("Silhouette analysis for AgglomerativeClustering on sample data "
                    "with n_clusters = %d" % n_clusters),
                    fontsize=14, fontweight='bold')
        plt.savefig(name+'_'+str(n_clusters)+'.pdf',format='pdf')

    plt.show()

def get_values_by_label(ktotal,alg,X):
    kmeans = alg(n_clusters=ktotal)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    intervals_by_label = dict()
    for i in range(ktotal):
        intervals_by_label[i] = []
    for l,x in zip(labels,X):
        intervals_by_label[l].append(x)

    return intervals_by_label

def plot_lines_blox_plot(intervals_by_label,ktotal,dim=8):
    fig1 = plt.figure(figsize=(4,2*(ktotal+1)))
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    x = [i for i in range(dim)]

    colors = {0:'blue',1:'red',2:'green',3:'orange',4:'gray',5:'cyan',6:'magenta'}

    for k,xs in intervals_by_label.items():
        xs = np.asarray([np.asarray(z) for z in xs])
        d = describe(xs,axis=0)
        means = d.mean
        stds = np.sqrt(d.variance)

        ax.errorbar(x,means,yerr=stds,fmt='-o',label=str(k)+'(mean w/ std)',c=colors[k],alpha=0.8)
        
        nx = xs.shape[1]
        data_to_boxplot = []
        for i in range(nx):
            data_to_boxplot.append(xs[:,i])
        ax1 = fig1.add_subplot(ktotal,1,k+1)
        ax1.boxplot(data_to_boxplot,boxprops=dict(color=colors[k],alpha=0.7))
        # ax.plot(x,mins,label=str(k)+' (mins)',c=colors[k],alpha=0.5)
        # ax.plot(x,maxs,label=str(k)+' (maxs)',c=colors[k],alpha=0.5)

    ax.legend(loc='upper center', bbox_to_anchor=(1, 1), ncol=1)
    ax.set_xticks(x,[str(k) for k in x])
    ax.set_xlabel('intervals')
    fig.tight_layout()
    fig1.tight_layout()
    plt.show()

# range_n_clusters = [2, 3, 4, 5, 6]
# test_n_clusters(range_n_clusters,X,KMeans,'kmeans')

slopes_artificial,intervals_artificial = read_file(samples_breakpoints='data/artificial_intervals.txt')
intervals_artificial = np.asarray(intervals_artificial)
slopes_original,breakpoints_original = read_file()
intervals_original = np.asarray([breakpoints2intervals(b) for b in breakpoints_original])

# DADOS NORMALIZADOS POR TODOS OS DADOS
original_data = np.concatenate((slopes_original,intervals_original),axis=1)
artificial_data = np.concatenate((slopes_artificial,intervals_artificial),axis=1)
all_data = np.concatenate((original_data,artificial_data),axis=0)

m = np.mean(all_data,axis=0)
std = np.std(all_data,axis=0)
original_data = (original_data - m)/std
artificial_data = (artificial_data -m)/std

ktotal = 3
alg = KMeans
intervals_by_label = get_values_by_label(ktotal,alg,original_data)
plot_lines_blox_plot(intervals_by_label,ktotal)

plt.scatter(original_data[:, 4], original_data[:, 6], marker='.', s=30, lw=0, alpha=0.7)
plt.xlabel('interval 1')
plt.ylabel('interval 2')
plt.show()

# pca usando todos os dados OK
# clusters usando todo os dados
# sinais aleatórios para comparar os originais OK
# angulos diferentes e intervalos probabilisticos OK
# visualizar intervalos 1 e 3 OK