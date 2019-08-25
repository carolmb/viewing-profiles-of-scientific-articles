import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_hist_int(X,Y,filename,xlabel):
    plt.bar(X, height=Y)
    plt.locator_params(axis='x',integer=MaxNLocator(integer=True))
    plt.xlabel(xlabel)
    plt.ylabel('occurrences')
    plt.savefig(filename+'.pdf')
    plt.clf()

def get_life_time(data):
	life_time = []
	for _,_,_,X,_,_ in data:
		life_time.append(X[-1]-X[0])
	life_time = np.asarray(life_time)
	return life_time

def get_no_of_intervals(data):
	intervals = []
	for _,slopes,_,_,_,_ in data:
		intervals.append(len(slopes))
	intervals = np.asarray(intervals)
	return intervals

def plot_hist_real(X,bins,filename,xlabel,ylabel):
	plt.hist(X,bins=bins)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.savefig(filename+'.pdf')
	plt.clf()

def plot_life_time_hist(data,filename):
	life_time = get_life_time(data)
	xlabel = 'lifetime (years)'
	ylabel = 'number of papers'
	plot_hist_real(life_time,10,filename,xlabel,ylabel)

def arg_remove_outliers(values):
	Q1 = np.quantile(values,0.25)
	Q3 = np.quantile(values,0.75)
	IQR = Q3 - Q1
	args1 = values > (Q1 - 1.5*IQR)
	args2 = values < (Q3 + 1.5*IQR)
	args = np.logical_and(args1,args2)
	return args

def get_no_of_visual(data):
	visual = []
	for _,_,_,_,Y,_ in data:
		visual.append(Y[-1])
	visual = np.asarray(visual)
	return visual

def plot_no_of_visual(data,filename):
	visual = get_no_of_visual(data)
	args = arg_remove_outliers(visual)
	visual = visual[args]
	xlabel = 'number of visualizations'
	ylabel = 'number of papers'
	plot_hist_real(visual,10,filename,xlabel,ylabel)

def filter_outliers(data):
	life_time = get_life_time(data)
	args1 = arg_remove_outliers(life_time)
	args3 = life_time > 3
	args1 = np.logical_and(args1,args3)
	visual = get_no_of_visual(data)
	args2 = arg_remove_outliers(visual)

	args = np.logical_and(args1,args2)

	data0 = data[args]
	return data0

def plot_no_of_intervals(data,filename):
    intervals = get_no_of_intervals(data)
    X,Y = np.unique(intervals,return_counts=True)
    plot_hist_int(X,Y,filename,'number of intervals')

def group_by_num_visual(data):
	visual = get_no_of_visual(data)
	Q2 = np.quantile(visual,0.5)
	less_Q2 = []
	greater_Q2 = []
	for sample in data:
		if sample[4][-1] < Q2:
			less_Q2.append(sample)
		else:
			greater_Q2.append(sample)
	return less_Q2,greater_Q2
