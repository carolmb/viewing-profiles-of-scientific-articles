import numpy as np
import pandas as pd
from clusters import select_original_breakpoints, norm, plot_pca
import matplotlib.pyplot as plt
import json


def get_types_views():
    f_input = open('others/papers_plos_data_time_series2_filtered_xml_pdf_html.json', 'r')
    data = json.load(f_input)
    html_dist = []
    pdf_dist = []
    xml_dist = []
    for doi, paper in data.items():
        html = 0
        xml = 0
        pdf = 0
        if len(paper['time_series']['html_views']) > 0:
            html = paper['time_series']['html_views'][-1]
        if len(paper['time_series']['pdf_views']) > 0:
            pdf = paper['time_series']['pdf_views'][-1]
        if len(paper['time_series']['xml_views']) > 0:
            xml = paper['time_series']['xml_views'][-1]
        total = html + pdf + xml
        if total > 0:
            html_dist.append(html/total)
            pdf_dist.append(pdf/total)
            xml_dist.append(xml/total)

    total_html = sum(html_dist)
    total_pdf = sum(pdf_dist)
    total_xml = sum(xml_dist)

    total = total_html + total_pdf + total_xml
    print(total_xml/total)
    print(total_pdf/total)
    print(total_html/total)

    plt.hist(html_dist)
    plt.title('html')
    plt.show()

    plt.hist(xml_dist)
    plt.title('xml')
    plt.show()

    plt.hist(pdf_dist)
    plt.title('pdf')
    plt.show()

    # 0,006
    # 0,1646
    # 0,1781
    # 0,6508


if __name__ == '__main__':

    categories = pd.read_csv('data/DOI2Category.txt', delimiter='\t', names=['doi', 'wos', 'cat'])
    print(categories[:10])

    colors = ['tab:red', 'tab:blue', 'tab:orange', 'tab:green', 'tab:grey']

    f_input = 'segm/segmented_curves_html.txt'
    f_output = 'data/html_by_cat/'
    # generate_groups(get_labels_clustering_hier, f_input, f_output)

    sources = [f_output+'k2/clusters_ind_single_0.47_2.txt',
               f_output+'k3/clusters_ind_single_0.38_3.txt',
               f_output+'k4/clusters_ind_single_0.47_4.txt',
               f_output+'k5/clusters_ind_single_0.47_5.txt']
    total = 0
    for k, source in zip([2, 3, 4, 5], sources):
        # dois = pd.read_csv(f_output+'k%d/k%d_dois.txt' % (k, k), header=None)
        # cats = categories.loc[categories['doi'].isin(dois.values.flatten())]

        # total += len(dois)
        #
        # cats['cat'].hist()
        # plt.show()
        #
        # idxs = []
        # b2_group = cats[cats['cat'] == 'B6']['doi']
        # for doi in dois.values.flatten():
        #     idxs.append(doi in b2_group.values.flatten())
        # idxs = np.asarray(idxs)[:50000]

        # data, labels = read_data_with_label("k%d/k%d_curves_label.txt"%(k,k))
        slopes, intervals = select_original_breakpoints(k, f_input)
        data = np.concatenate((slopes, intervals), axis=1)

        X = norm(data)[:50000]

        labels = np.loadtxt(source, dtype=np.int)
        unique, counts = np.unique(labels, return_counts=True)

        unique = unique[counts >= 10]
        counts = counts[counts >= 10]
        unique_idxs = np.argsort(counts)[-3:]
        unique = unique[unique_idxs].tolist()
        labels = [unique.index(l) if l in unique else -1 for l in labels]
        labels = np.asarray(labels)

        # X = X[idxs]
        # labels = labels[idxs]

        print(np.unique(labels, return_counts=True))
        print(len(X), len(labels))
        plot_pca(X, labels, colors, f_output + 'imgs/pca_%s_%d' % ('hierar', k))

    print(total)