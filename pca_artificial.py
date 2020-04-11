import sys
import util
import getopt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats
from collections import defaultdict
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
from scatter import Score
from clusters import get_average_curve
from clusters import average_curve
from read_file import select_original_breakpoints,read_artificial_breakpoints

def get_data(source1,source2,n,preprocessed):
	if preprocessed:
		slopes_original,intervals_original = util.read_preprocessed_file(n,source1)
	else:
		slopes_original,intervals_original = select_original_breakpoints(n,source1)
	
	_,slopes_artificial,intervals_artificial = read_artificial_breakpoints(source2)
	
	original_data = np.concatenate((slopes_original,intervals_original),axis=1)
	# print(slopes_artificial[:5],intervals_artificial[:5])
	artificial_data = np.concatenate((slopes_artificial,intervals_artificial),axis=1)

	all_data = np.concatenate((original_data,artificial_data),axis=0)
	m = np.mean(all_data,axis=0)
	std = np.std(all_data,axis=0)
	
	all_data_norm = (all_data-m)/std
	original_norm = (original_data-m)/std
	artificial_norm = (artificial_data-m)/std

	return artificial_data,original_data,all_data_norm,artificial_norm,original_norm

def plot_pca(y1,y2,xlabel,ylabel,filename):

	df1 = pd.DataFrame({'x':y1[:,0],'y':y1[:,1],'type':'artificial'})
	df2 = pd.DataFrame({'x':y2[:,0],'y':y2[:,1],'type':'original'})

	xmin,xmax = min(df1['x'].min(),df2['x'].min()),max(df1['x'].max(),df2['x'].max())
	ymin,ymax = min(df1['y'].min(),df2['y'].min()),max(df1['y'].max(),df2['y'].max())

	ax1 = sns.jointplot(x="x", y="y", data=df1, kind="kde", color='red')
	ax1.ax_marg_x.set_xlim(xmin,xmax)
	ax1.ax_marg_y.set_ylim(ymin,ymax)
	ax1.set_axis_labels(xlabel,ylabel,fontsize=16)

	# rect = patches.Rectangle((box1[0],box1[2]),box1[1]-box1[0],box1[3]-box1[2],linewidth=1,edgecolor='blue',facecolor='none')
	# ax1.ax_joint.add_patch(rect)

	plt.savefig(filename+'artificial_pca.pdf',bbox_inches='tight')

	ax2 = sns.jointplot(x="x", y="y", data=df2, kind="kde", color='green')	
	ax2.set_axis_labels(xlabel,ylabel,fontsize=16)
	ax2.ax_marg_x.set_xlim(xmin,xmax)
	ax2.ax_marg_y.set_ylim(ymin,ymax)

	# rect = patches.Rectangle((box2[0],box2[2]),box2[1]-box2[0],box2[3]-box2[2],linewidth=1,edgecolor='blue',facecolor='none')
	# ax2.ax_joint.add_patch(rect)

	plt.savefig(filename+'original_pca.pdf',bbox_inches='tight')

def plot_zoom(artificial,original,ax):
	artificial_comp = artificial
	original_comp = original
	# for (x,y),z in zip(y1,artificial):
	# 	if x >= box1[0] and x <= box1[1] and y >= box1[2] and y <= box1[3]:
	# 		artificial_comp.append(z)
	# for (x,y),z in zip(y2,original):
	# 	if x >= box2[0] and x <= box2[1] and y >= box2[2] and y <= box2[3]:
	# 		original_comp.append(z)

	artificial_comp = np.asarray(artificial_comp)
	original_comp = np.asarray(original_comp)

	artificial_average = get_average_curve(artificial_comp,5,0)
	original_average = get_average_curve(original_comp,5,1)
	average = [artificial_average,original_average]

	colors = ['red','green']
	legend = ['artificial','original']
	#plt.figure(figsize=(3,3))
	for x0,y0,s0,s1,k in average:
		ax.errorbar(x0,y0,xerr=s0,yerr=s1,marker='o',color=colors[k],linestyle='-',alpha=0.9,label=legend[k])

	plt.legend()
	#plt.savefig(filename+'_original_artificial_comp2.pdf')

def get_linspace(syn_trans,original_trans):
	print('syn',syn_trans.shape,original_trans.shape)
	xmin = min(syn_trans[:,0])
	xmax = max(syn_trans[:,0])
	ymin = min(syn_trans[:,1])
	ymax = max(syn_trans[:,1])

	xmin = min(xmin,min(original_trans[:,0]))
	xmax = max(xmax,max(original_trans[:,0]))
	ymin = min(ymin,min(original_trans[:,1]))
	ymax = max(ymax,max(original_trans[:,1]))

	X = np.linspace(xmin,xmax,100)
	Y = np.linspace(ymin,ymax,100)
	xx,yy = np.meshgrid(X,Y)
	data = np.concatenate((xx.ravel()[:,None],yy.ravel()[:,None]),axis=1)

	return data,xx,yy

def plot_pca_manual(xx,yy,Z,data,filename,title,xlabel,ylabel,cmap,color):
	fig = plt.figure(figsize=(9,9))
	plt.subplots_adjust(wspace=0, hspace=0)
	gs = gridspec.GridSpec(5, 5) 
	ax = plt.subplot(gs[1:,:-1])
	CS = ax.contourf(xx,yy,Z.reshape(100,-1),levels=8,cmap=cmap)
	ax.set_xlabel(xlabel,fontsize=16)
	ax.set_ylabel(ylabel,fontsize=16)
	


	ax1 = plt.subplot(gs[1:,-1],sharey=ax)
	sns.distplot(data[:,1], vertical=True, ax=ax1, color=color, hist=False, kde_kws={"shade": True})
	ax1.set_xticks([])
	ax1.label_outer()
	ax1.spines['top'].set_visible(False)
	ax1.spines['right'].set_visible(False)
	ax1.spines['bottom'].set_visible(False)
	# X = sum(stats.multivariate_normal(x,cov=0.05).pdf(xx[0]) for x in data[:,0])
	
	ax2 = plt.subplot(gs[0,:-1],sharex=ax)
	# Y = sum(stats.multivariate_normal(x,cov=0.05).pdf(yy[0]) for x in data[:,1])
	sns.distplot(data[:,0], ax=ax2, color=color, hist=False, kde_kws={"shade": True})
	ax2.label_outer()
	ax2.spines['top'].set_visible(False)
	ax2.spines['right'].set_visible(False)
	ax2.spines['left'].set_visible(False)
	ax2.set_yticks([])
	
	# ax3 = plt.subplot(gs[-1,:])
	# fig.colorbar(CS, cax=ax3, orientation='horizontal')
	cax = fig.add_axes([0.8, 0.55, 0.02, 0.35])
	fig.colorbar(CS, cax=cax, orientation='vertical')
	plt.savefig('imgs/'+filename+title+'_original_pca2.pdf')

def surface_test(syn,original,syn_trans,original_trans,filename,title,xlabel,ylabel):
	N = len(syn_trans)
	data,xx,yy = get_linspace(syn_trans,original_trans)

	# Z1 = sum(stats.multivariate_normal(xy,cov=0.05).pdf(data) for xy in syn_trans)
	# Z2 = sum(stats.multivariate_normal(xy,cov=0.05).pdf(data) for xy in original_trans)
	
	kde = stats.gaussian_kde(syn_trans.T)
	Z1 = kde.pdf(data.T)

	kde = stats.gaussian_kde(original_trans.T)
	Z2 = kde.pdf(data.T)

	# plot_pca_manual(xx,yy,Z1,syn_trans,filename,'Synthetic',xlabel,ylabel,'Reds','red')
	# plot_pca_manual(xx,yy,Z2,original_trans,filename,'Original',xlabel,ylabel,'Greens','green')
	
	fig = plt.figure(figsize=(4.5,4))
	plt.subplots_adjust(wspace=0, hspace=0)
	CS = plt.contourf(xx,yy,np.absolute(Z1-Z2).reshape(100,-1), levels=8,cmap='Blues')
	fig.colorbar(CS, orientation='vertical')
	plt.xlabel(xlabel,fontsize=12)
	plt.ylabel(ylabel,fontsize=12)
	
	total = sum(np.absolute(Z1-Z2))
	print('total sum diff',total)
	plt.title(total)
	plt.tight_layout()

	plt.savefig(filename+'diff_test.pdf')

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

	title = "\nscatter dist = %.6f" % scatter_dist

	y1_label = 'PCA1 (%.2f%%)' % y1_explained
	y2_label = 'PCA2 (%.2f%%)' % y2_explained

	return y1,y2,y1_label,y2_label,title

def get_args_terminal():
    argv = sys.argv[1:]
    
    source1,source2 = 'segm/segmented_curves_filtered.txt',None
    output = None
    N = 5
    preprocessed = False
    try:
        opts,args = getopt.getopt(argv,"o:N:p",['s1=','s2='])
    except getopt.GetoptError:
        print('usage: python example.py -s1 <source1> -s2 <source2> -o <output> -N <n> -p')
        return None

    for opt,arg in opts:

        if opt == '--s1':
            source1 = arg
        if opt == '--s2':
            source2 = arg
        elif opt == '-o':
            output = arg
        elif opt == '-N':
            N = int(arg)
        elif opt == '-p':
            preprocessed = True

    return source1,source2,output,N,preprocessed

if __name__ == "__main__":
   
	source1,source2,output,N,preprocessed = get_args_terminal()
	print(preprocessed,source1,source2)

	artificial_data,original_data,all_data_norm,artificial_norm,original_norm = get_data(source1,source2,N,preprocessed)

	y1,y2,y1_label,y2_label,title = get_pca_infos(all_data_norm,artificial_norm,original_norm)
	surface_test(artificial_data,original_data,y1,y2,output,title,y1_label,y2_label)

	print(title)
	plot_pca(y1,y2,y1_label,y2_label,output)

'''
# para todas as curvas com label: 
python pca_artificial.py --s2 k3/markov1_uni_k3_label.txt -N 3 -o k3/

'''