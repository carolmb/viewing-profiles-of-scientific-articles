import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.decomposition import PCA

from scatter import Score
from read_file import select_original_breakpoints,read_artificial_breakpoints

def get_data(filename,n):
	slopes_original,intervals_original = select_original_breakpoints(n)
	_,slopes_artificial,intervals_artificial = read_artificial_breakpoints(filename)
	original_data = np.concatenate((slopes_original,intervals_original),axis=1)
	artificial_data = np.concatenate((slopes_artificial,intervals_artificial),axis=1)
	print(original_data.shape,artificial_data.shape)

	all_data = np.concatenate((original_data,artificial_data),axis=0)
	m = np.mean(all_data,axis=0)
	std = np.std(all_data,axis=0)
	all_data = (all_data-m)/std

	original_data = (original_data-m)/std
	artificial_data = (artificial_data-m)/std

	return artificial_data,original_data,all_data

def plot_pca(y1,y2,xlabel,ylabel,title,filename):

	# plt.title(title+' original std='+ str(original_std[0])[:5]+' artificial std='+str(artificial_std[0])[:5])
	# plt.xlabel('original std ='+str(original_std[1])[:5]+' artificial std='+str(artificial_std[1])[:5])

	plt.figure()
	plt.scatter(y1[:,0],y1[:,1],c='#307438',alpha=0.1,label='original')
	plt.scatter(y2[:,0],y2[:,1],c='#b50912',alpha=0.1,label='artificial')

	plt.title(title)
	
	plt.xlabel(xlabel, fontsize=14)
	plt.ylabel(ylabel, fontsize=14)
	plt.legend(bbox_to_anchor=(1.2,1.0))
	plt.savefig('imgs/'+filename+'_pca.pdf',bbox_inches='tight')
	plt.clf()

def get_pca_infos(original,artificial,all_data):
	pca = PCA(n_components=2)
	pca.fit(all_data)

	scatter_dist = -1000
	try:
		scatter_dist = Score(all_data,np.array([len(original),len(artificial)]))
	except:
		pass

	y1 = pca.transform(original)
	y2 = pca.transform(artificial)

	# original_std = np.std(y1,axis=0)
	# artificial_std = np.std(y2,axis=0)

	y1_explained, y2_explained = pca.explained_variance_ratio_[:2]
	y1_explained = y1_explained*100
	y2_explained = y2_explained*100

	title = "\n(scatter dist = %.6f)" % scatter_dist

	y1_label = 'PCA1 (%.2f%%)' % y1_explained
	y2_label = 'PCA2 (%.2f%%)' % y2_explained

	return y1,y2,y1_label,y2_label,title

if __name__ == "__main__":
	filenames = sorted(glob.glob('data/original1/plos_one_2019*.txt'))

	for i,filename in enumerate(filenames):
	    artificial_data,original_data,all_data = get_data(filename,2+i%4)
	    
	    y1,y2,y1_label,y2_label,title = get_pca_infos(artificial_data,original_data,all_data)
	    
	    filename = filename.split('/')[-1][:-4]
	    title = filename + title
	    plot_pca(y1,y2,y1_label,y2_label,title,filename)