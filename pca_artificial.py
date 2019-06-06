import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from analise_breakpoints import read_file
from analise_breakpoints import breakpoints2intervals
from scipy.stats import pearsonr

def pca_fit(original,artificial):
    pca = PCA(n_components=2)
    pca.fit(original)

    y1 = pca.transform(original)
    y2 = pca.transform(artificial)

    plt.scatter(y1[:,0],y1[:,1],c='green',alpha=0.3)
    plt.scatter(y2[:,0],y2[:,1],c='red',alpha=0.3)
    plt.title('green is original data, red is artificial data')
    plt.show()


slopes_artificial,intervals_artificial = read_file(samples_breakpoints='data/artificial_intervals.txt')
intervals_artificial = np.asarray(intervals_artificial)
slopes_original,breakpoints_original = read_file()
# intervals_artificial = [breakpoints2intervals(b) for b in breakpoints_artificial]
intervals_original = np.asarray([breakpoints2intervals(b) for b in breakpoints_original])
pca_fit(intervals_original[:,:-1],intervals_artificial[:,:-1])
pca_fit(intervals_original,intervals_artificial)

slopes_artificial,intervals_artificial = read_file(samples_breakpoints='data/artificial_intervals_norm.txt')
slopes_original,breakpoints_original = read_file()
# intervals_artificial = [breakpoints2intervals(b) for b in breakpoints_artificial]
intervals_original = [breakpoints2intervals(b) for b in breakpoints_original]
pca_fit(intervals_original,intervals_artificial)


slopes_artificial,intervals_artificial = read_file(samples_breakpoints='data/artificial_slopes.txt')
slopes_original,breakpoints_original = read_file()
intervals_original = [breakpoints2intervals(b) for b in breakpoints_original]
# pca_fit(slopes_original,slopes_artificial)
