import json
import numpy as np
from read_file import load_data, filter_outliers
sources = ['clusters\\clusters\\clusters_ind_single_0.50_2.txt',
                   'clusters\\clusters\\clusters_ind_single_0.35_3.txt',
                   'clusters\\clusters\\clusters_ind_single_0.47_4.txt',
                   'clusters\\clusters\\clusters_ind_single_0.56_5.txt']

data = load_data()
data = filter_outliers(data)
for N, source in zip([2, 3, 4, 5], sources):
    labels = np.loadtxt(source, dtype=np.int).tolist()
    unique, counts = np.unique(labels, return_counts=True)
    unique = unique[counts >= 10]
    counts = counts[counts >= 10]
    unique_idxs = np.argsort(counts)[-3:]
    unique = unique[unique_idxs].tolist()
    labels = [unique.index(l) if l in unique else -1 for l in labels]

    dois = []
    for i, s, b, xs, ys, p in data:
        if len(s) == N:
            dois.append(i)
    print(len(dois), len(labels))
    doi2cluster = dict(zip(dois, labels))
    str_json = json.dumps(doi2cluster)
    out = open('doi2cluster_%d_3.json' % N, 'w')
    out.write(str_json)
    out.close()