import glob
import numpy as np
from scipy import stats
from read_file import select_original_breakpoints,read_artificial_breakpoints

def Score(X,num_obj_cat):
    """
    IN:
    X: [N,M] array, where M is the number of features and N the number of objects.
    num_obj_cat: array containing number of objects for each class

    OUT:
    Tc: Scatter distance
    """

    [numSamples,Dim] = X.shape

    numCat = num_obj_cat.size

    u = np.mean(X,axis=0)
    s = np.std(X,axis=0,ddof=1)

    B = X - u
    Z = B/s


    ind = np.cumsum(num_obj_cat)
    ind = np.concatenate(([0],ind))

    uCat = np.zeros([numCat,Dim])
    for k in range(numCat):
        data_class = Z[ind[k]:ind[k+1]]
        uCat[k] = np.mean(data_class,axis=0)

    X_aux = Z.copy()
    for k in range(numCat):
        X_aux[ind[k]:ind[k+1]] -=  uCat[k]

    Sw = np.zeros([Dim,Dim]) # Within-cluster scatter matrix
    Sb = np.zeros([Dim,Dim]) # Between-cluster scatter matrix
    for k in range(numCat):
        data_class = X_aux[ind[k]:ind[k+1]]

        Sw += np.dot(data_class.T,data_class) 

        aux = (uCat[k]-0.).reshape([1,Dim])
        Sb += num_obj_cat[k]*np.dot(aux.T,aux)


    C = np.dot(np.linalg.inv(Sw),Sb)
    Tc = np.trace(C)

    return Tc

def norm_data(slopes_original,intervals_original,slopes_artificial,intervals_artificial):
    original_data = np.concatenate((slopes_original,intervals_original),axis=1)
    artificial_data = np.concatenate((slopes_artificial,intervals_artificial),axis=1)
    all_data = np.concatenate((original_data,artificial_data),axis=0)

    m = np.mean(all_data,axis=0)
    std = np.std(all_data,axis=0)
    original_data = (original_data - m)/std
    artificial_data = (artificial_data -m)/std
    
    all_data = (all_data - m)/std
    return original_data,artificial_data,all_data

def score_all_files(filenames,original_data_filename):
    for i,f in enumerate(filenames):
        n = 2 + i%4
        print(f)
        slopes_original,intervals_original = select_original_breakpoints(n)
        _,slopes_artificial,intervals_artificial = read_artificial_breakpoints(f)
        
        original,artificial,all_data = norm_data(slopes_original,intervals_original,slopes_artificial,intervals_artificial)
        try:
            print("%.6f" % Score(all_data,np.array([len(original),len(artificial)])))
        except:
            print('Error',f)

# base = [2,3,4,5]
# print('original1')
# filenames = sorted(glob.glob('data/original1/*_test.txt'))
# original_data_filename = 'data/plos_one_total_breakpoints_k4_original1_data_filtered.txt'
# score_all_files(filenames,original_data_filename)

# print()

# print('original1')
# filenames = sorted(glob.glob('data/original1/*.txt'))
# original_data_filename = 'data/plos_one_total_breakpoints_k4_original1_data_filtered.txt'
# score_all_files(filenames,original_data_filename)
