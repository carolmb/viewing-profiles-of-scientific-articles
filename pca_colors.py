import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from analise_breakpoints import read_file
from analise_breakpoints import breakpoints2intervals
from analise_breakpoints import read_file_original
from scipy.stats import pearsonr
import WOSGetter

def plot_pca_colors(slopes,breakpoints,colors,k):
    slopes = np.asarray([(np.arctan(s)*57.2958) for s in slopes])
    intervals = [breakpoints2intervals(b) for b in breakpoints]
    all_data = np.concatenate((slopes,intervals),axis=1)

    m = np.mean(all_data,axis=0)
    std = np.std(all_data,axis=0)
    all_data = (all_data - m)/std

    pca = PCA(n_components=2)
    pca.fit(all_data)

    y = pca.transform(all_data)

    plt.figure()
    plt.scatter(y[:,0],y[:,1],c=colors,alpha=0.4)
    plt.title(k)
    # plt.title(title+' original std='+ str(original_std[0])[:5]+' artificial std='+str(artificial_std[0])[:5])
    # plt.xlabel('original std ='+str(original_std[1])[:5]+' artificial std='+str(artificial_std[1])[:5])
    plt.colorbar()
    plt.show()
    # plt.savefig(filename[:-4]+'_pca.png')

plt.ion()

completeTimeSeries = WOSGetter.GZJSONLoad("plosone2016_hits.json.gz")


completeTimeSeries = WOSGetter.GZJSONLoad("plosone2016_hits.json.gz")

keys = ['total','pdf','html','xml']
for k in keys:
    xs,ys,cs=[],[],[]
    for serie in completeTimeSeries:
        data = np.asarray(serie[k])
        years = data[:,0]
        counts = data[:,1]
        xs.append(years)
        ys.append(counts)
        cs.append(counts[-1])


xs,ys = read_file_original()
for n in [2,3,4,5]:
    print(n)
    idxs,slopes,breakpoints = read_file(n=n)
    xs_n = [xs[idx] for idx in idxs]
    # deltas = [1 if xs[-1]-xs[0] > 5 else 0 for xs in xs_n ]

    # cor por número de visualização
    deltas = [ys[idx][-1] for idx in idxs]
    q75 = np.quantile(deltas,0.75)
    q25 = np.quantile(deltas,0.25)
    iqr = q75 - q25
    deltas = [min(q75+1.5*iqr,d) for d in deltas]
    deltas = [max(q25-1.5*iqr,d) for d in deltas]
    print(deltas[:10])

    # cor por delta anos
    # deltas =read_file [xs[-1]-xs[0] for xs in xs_n]

    plot_pca_colors(slopes,breakpoints,deltas,n)
    # print(np.unique(delta_xs_n,return_counts=True))
