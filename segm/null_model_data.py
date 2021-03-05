import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import gaussian,convolve

def get_slope(f_input,n_input):
	print(f_input)
	all_slopes = []
	content = open(f_input,'r').read().split('\n')[:-1]
	
	N = len(content)
	for i in range(0,N,n_input):
		slopes = content[i+1].split(' ')
		slopes = [np.arctan(float(m))*57.2958 for m in slopes]
		all_slopes += slopes
	return all_slopes

def breakpoints2intervals(x):
    intervals = [x[0]]
    for i in range(len(x)-1):
        intervals.append(x[i+1]-x[i])
    intervals.append(1-x[-1])
    return intervals

def get_intervals(f_input,n_input):
	print(f_input)
	all_intervals = []
	content = open(f_input,'r').read().split('\n')[:-1]
	
	N = len(content)
	for i in range(0,N,n_input):
		intervals = content[i+2].split(' ')
		intervals = breakpoints2intervals([float(m) for m in intervals])
		all_intervals += intervals
	return all_intervals

def syn_data():
	f_input = "../data/plos_one_2019.txt"
	f_output = 'syn_data.txt'
	out = open(f_output,'w')

	content = open(f_input,'r').read().split('\n')[:-1]
	N = len(content)
	for i in range(0,N,3):
		views = content[i+2].split(',')
		try:
			views = [float(m) for m in views]
		except:
			print(content[i])
			print(views)
			continue
		M = len(views)
		max_diff = 0
		for j in range(M-1):
			max_diff = max(views[j+1]-views[j],max_diff)

		syn_data = [np.random.randint(1,max_diff) for j in range(M)]
		syn_data = np.cumsum(syn_data)
		syn_data = [str(s) for s in syn_data]
		syn_str = ','.join(syn_data)
		out.write(content[i]+'\n')
		out.write(content[i+1]+'\n')
		out.write(syn_str+'\n')

	out.close()

def dist(original,synthetic,x_label,f_output):
	x = np.linspace(0,90,300)
	y = gaussian(300,1)
	print()
	ori_gaussian = gaussian_kde(original)
	syn_gaussian = gaussian_kde(synthetic)

	ax = sns.distplot(original,label='real-world profiles',color='tab:green',hist=False,kde_kws={"shade":True})
	sns.distplot(synthetic,label='control synthetic profiles',ax=ax,color='tab:red',hist=False,kde_kws={"shade":True})
	# plt.plot(x,ori_gaussian.pdf(x), color='tab:green', label='original')
	# plt.plot(x,syn_gaussian.pdf(x), color='tab:red', label='synthetic')
	plt.legend()
	plt.xlabel(x_label,fontsize=14)
	# plt.ylabel('')
	plt.savefig(f_output)
	plt.clf()

f_ori = 'segmented_curves_filtered.txt'
f_syn = 'segmented_curves_syn_data_filtered.txt'

ori_slopes = get_slope(f_ori,4)
syn_slopes = get_slope(f_syn,4)

ori_intervals = get_intervals(f_ori,4)
syn_intervals = get_intervals(f_syn,4)

dist(ori_slopes,syn_slopes,'angle','dist_angle.pdf')
dist(ori_intervals,syn_intervals,'interval','dist_intervals.pdf')