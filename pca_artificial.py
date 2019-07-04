import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from analise_breakpoints import read_file
from analise_breakpoints import breakpoints2intervals
from scipy.stats import pearsonr
from analise_breakpoints import read_file_original
import glob

def get_data(filename):
    n = int(filename.split('_')[-1].split('.txt')[0])
    _,slopes_original,breakpoints_original = read_file(samples_breakpoints='data/plos_one_total_breakpoints_k4it.max100stop.if.errorFALSE.txt',n=n)
    _,slopes_artificial,intervals_artificial = read_file(samples_breakpoints=filename,n=n)

    slopes_original = np.asarray([(np.arctan(s)*57.2958) for s in slopes_original])
    #print(slopes_original[:1])
    intervals_original = [breakpoints2intervals(b) for b in breakpoints_original]

    print(len(slopes_original),len(intervals_original))

    # DADOS NORMALIZADOS POR TODOS OS DADOS
    original_data = np.concatenate((slopes_original,intervals_original),axis=1)
    artificial_data = np.concatenate((slopes_artificial,intervals_artificial),axis=1)
    all_data = np.concatenate((original_data,artificial_data),axis=0)

    m = np.mean(all_data,axis=0)
    std = np.std(all_data,axis=0)
    original_data = (original_data - m)/std
    artificial_data = (artificial_data -m)/std
    
    all_data = (all_data - m)/std

    return artificial_data,original_data,all_data

def pca_fit(original,artificial,all_data,title):
    pca = PCA(n_components=2)
    pca.fit(all_data)

    y1 = pca.transform(original)
    y2 = pca.transform(artificial)

    original_std = np.std(y1,axis=0)
    artificial_std = np.std(y2,axis=0)

    plt.figure()
    plt.scatter(y1[:,0],y1[:,1],c='green',alpha=0.3)
    plt.scatter(y2[:,0],y2[:,1],c='red',alpha=0.2)
    plt.title(title+' original std='+ str(original_std[0])[:5]+' artificial std='+str(artificial_std[0])[:5])

    plt.xlabel('original std ='+str(original_std[1])[:5]+' artificial std='+str(artificial_std[1])[:5])

    plt.show()
    plt.savefig('imgs_python/'+filename[5:-4]+'_pca.png')


plt.ion()
# DADOS GERADOS PARA INTERVALOS ARTIFICIAIS (ANGULO FIXO, INTERVALO VARIANDO)
filenames = glob.glob('data/plos_one_artificial_*.txt')

for filename in filenames:
    artificial_data,original_data,all_data = get_data(filename)
    pca_fit(original_data,artificial_data,all_data,filename)
