import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from analise_breakpoints import read_file
from analise_breakpoints import breakpoints2intervals
from analise_breakpoints import read_file_original
from scipy.stats import pearsonr
import WOSGetter
from matplotlib import cm
from sklearn.utils.random import sample_without_replacement

def plot_pca_colors(slopes,breakpoints,colors,k,filename):
    colors = np.asarray(colors)
    slopes = np.asarray([(np.arctan(s)*57.2958) for s in slopes])
    intervals = [breakpoints2intervals(b) for b in breakpoints]
    all_data = np.concatenate((slopes,intervals),axis=1)

    m = np.mean(all_data,axis=0)
    std = np.std(all_data,axis=0)
    all_data = (all_data - m)/std

    pca = PCA(n_components=2)
    pca.fit(all_data)

    y = pca.transform(all_data)

    print(y.shape)
    if len(y) > 20000:
        idxs = sample_without_replacement(len(y),20000)
        y = y[idxs]
        colors = colors[idxs]

    plt.figure()
    cmap = cm.get_cmap('inferno', 10)
    plt.scatter(y[:,0],y[:,1],c=colors,cmap=cmap,alpha=0.3)
    plt.title(k)
    # plt.title(title+' original std='+ str(original_std[0])[:5]+' artificial std='+str(artificial_std[0])[:5])
    # plt.xlabel('original std ='+str(original_std[1])[:5]+' artificial std='+str(artificial_std[1])[:5])
    plt.colorbar()
    plt.title(filename+' pca')
    plt.show()
    plt.savefig(filename+'_pca.png')

plt.ion()

xs,ys = read_file_original(filename='data/plos_one_data_total.txt')

for n in [2,3,4,5]:
    print(n)
    idxs,slopes,breakpoints = read_file(samples_breakpoints='data/plos_one_total_breakpoints_k4it.max100stop.if.errorFALSE_filtered.txt',n=n)
    xs_n = [xs[idx] for idx in idxs]
    # deltas = [1 if xs[-1]-xs[0] > 5 else 0 for xs in xs_n ]

    # cor por número de visualização
    deltas = [ys[idx][-1] for idx in idxs]
    print(deltas[:10])
    q75 = np.quantile(deltas,0.75)
    q25 = np.quantile(deltas,0.25)
    iqr = q75 - q25
    deltas = [min(q75+1.5*iqr,d) for d in deltas]
    deltas = [max(q25-1.5*iqr,d) for d in deltas]
    print(deltas[:10])
    plot_pca_colors(slopes,breakpoints,deltas,n,'imgs_python/colors_delta_visual_'+str(n))

    # cor por delta anos
    deltas = [xs[-1]-xs[0] for xs in xs_n]

    plot_pca_colors(slopes,breakpoints,deltas,n,'imgs_python/colors_delta_years_'+str(n))
    # print(np.unique(delta_xs_n,return_counts=True))
