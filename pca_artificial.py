import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
from scatter import Score
from clusters import get_average_curve
from clusters import average_curve
from read_file import select_original_breakpoints,read_artificial_breakpoints

def get_data(filename,n):
	slopes_original,intervals_original = select_original_breakpoints(n)
	_,slopes_artificial,intervals_artificial = read_artificial_breakpoints(filename)
	original_data = np.concatenate((slopes_original,intervals_original),axis=1)
	artificial_data = np.concatenate((slopes_artificial,intervals_artificial),axis=1)

	all_data = np.concatenate((original_data,artificial_data),axis=0)
	m = np.mean(all_data,axis=0)
	std = np.std(all_data,axis=0)
	
	all_data_norm = (all_data-m)/std
	original_norm = (original_data-m)/std
	artificial_norm = (artificial_data-m)/std

	return artificial_data,original_data,all_data_norm,artificial_norm,original_norm

def plot_pca(y1,y2,xlabel,ylabel,box1,box2,filename):

	df1 = pd.DataFrame({'x':y1[:,0],'y':y1[:,1],'type':'artificial'})
	df2 = pd.DataFrame({'x':y2[:,0],'y':y2[:,1],'type':'original'})

	xmin,xmax = min(df1['x'].min(),df2['x'].min()),max(df1['x'].max(),df2['x'].max())
	ymin,ymax = min(df1['y'].min(),df2['y'].min()),max(df1['y'].max(),df2['y'].max())

	ax1 = sns.jointplot(x="x", y="y", data=df1, kind="kde", color='red')
	ax1.ax_marg_x.set_xlim(xmin,xmax)
	ax1.ax_marg_y.set_ylim(ymin,ymax)
	ax1.set_axis_labels(xlabel,ylabel,fontsize=16)

	rect = patches.Rectangle((box1[0],box1[2]),box1[1]-box1[0],box1[3]-box1[2],linewidth=1,edgecolor='blue',facecolor='none')
	ax1.ax_joint.add_patch(rect)

	plt.savefig('imgs/'+filename+'_artificial_pca_%s_%s_%s_%s.pdf' % box1,bbox_inches='tight')

	ax2 = sns.jointplot(x="x", y="y", data=df2, kind="kde", color='green')	
	ax2.set_axis_labels(xlabel,ylabel,fontsize=16)
	ax2.ax_marg_x.set_xlim(xmin,xmax)
	ax2.ax_marg_y.set_ylim(ymin,ymax)

	rect = patches.Rectangle((box2[0],box2[2]),box2[1]-box2[0],box2[3]-box2[2],linewidth=1,edgecolor='blue',facecolor='none')
	ax2.ax_joint.add_patch(rect)

	plt.savefig('imgs/'+filename+'_original_pca_%s_%s_%s_%s.pdf' % box2,bbox_inches='tight')

def plot_zoom(y1,y2,artificial,original,box1,box2,filename):
	original_comp = []
	artificial_comp = []
	for (x,y),z in zip(y1,artificial):
		if x >= box1[0] and x <= box1[1] and y >= box1[2] and y <= box1[3]:
			artificial_comp.append(z)
	for (x,y),z in zip(y2,original):
		if x >= box2[0] and x <= box2[1] and y >= box2[2] and y <= box2[3]:
			original_comp.append(z)

	artificial_comp = np.asarray(artificial_comp)
	original_comp = np.asarray(original_comp)

	artificial_average = get_average_curve(artificial_comp,5,0)
	original_average = get_average_curve(original_comp,5,1)
	average = [artificial_average,original_average]

	colors = ['red','green']
	legend = ['artificial','original']
	plt.figure(figsize=(3,3))
	for x0,y0,s0,s1,k in average:
		plt.errorbar(x0,y0,xerr=s0,yerr=s1,marker='o',color=colors[k],linestyle='-',alpha=0.9,label=legend[k])

	plt.legend()
	plt.savefig(filename+'_original_artificial_comp_%s_%s_%s_%s_%s_%s_%s_%s.pdf' % (box1[0],box1[1],box1[2],box1[3],box2[0],box2[1],box2[2],box2[3]))

def get_pca_infos(all_data_norm,artificial_norm,original_norm):
	pca = PCA(n_components=2)
	pca.fit(all_data_norm)

	scatter_dist = -1000
	try:
		scatter_dist = Score(all_data_norm,np.array([len(original_norm),len(artificial_norm)]))
	except:
		pass

	y1 = pca.transform(artificial_norm)
	y2 = pca.transform(original_norm)

	y1_explained, y2_explained = pca.explained_variance_ratio_[:2]
	y1_explained = y1_explained*100
	y2_explained = y2_explained*100

	title = "\n(scatter dist = %.6f)" % scatter_dist

	y1_label = 'PCA1 (%.2f%%)' % y1_explained
	y2_label = 'PCA2 (%.2f%%)' % y2_explained

	return y1,y2,y1_label,y2_label,title

if __name__ == "__main__":
	filenames = sorted(glob.glob('data/original1_segmented_curves/plos_one_2019*.txt'))
	filenames = [f for f in filenames if '5' in f and 'comb' in f]
	
	for i,filename in enumerate(filenames):
		artificial_data,original_data,all_data_norm,artificial_norm,original_norm = get_data(filename,5)

		filename = filename.split('/')[-1][:-4]
		
		y1,y2,y1_label,y2_label,title = get_pca_infos(all_data_norm,artificial_norm,original_norm)
		print(filename,title)

		box1 = (-1.5,1.5,-1.5,1) # x0,x1,y0,y1
		box2 = (-1.5,1.5,-1.5,0.6)
		plot_zoom(y1,y2,artificial_data,original_data,box1,box2,filename)
		plot_pca(y1,y2,y1_label,y2_label,box1,box2,filename)
