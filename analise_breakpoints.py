# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict

from data_freq_scatter import generate_freq_plots
from data_hist_heat import generate_hist_plots,get_i
from read_file import read_file_original,read_artificial_breakpoints,read_original_breakpoints,save

def breakpoints2intervals(x):
    intervals = [x[0]]
    for i in range(len(x)-1):
        intervals.append(x[i+1]-x[i])
    intervals.append(1-x[-1])
    return intervals

def get_prob(X,intervalsx):
    x_by_i = defaultdict(lambda:[])
    idxs = get_i(X,intervalsx)
    for i,x in zip(idxs,X):
        x_by_i[i].append(x)
    unique,count = np.unique(idxs,return_counts=True)
    prob = defaultdict(lambda:0)
    total = len(idxs)
    for u,c in zip(unique,count):
        prob[u] = c/total
    return x_by_i,prob

def get_slope_val(nx,prob_slope1,x_by_i):
    prob = np.zeros(nx)
    for k,p in prob_slope1.items():
        prob[k] = p
    slopei = np.random.choice(np.arange(nx), p=prob)
    to_select = x_by_i[slopei]
    slopeiv = np.random.randint(0,len(to_select))
    slopev = to_select[slopeiv]
    return slopev,slopei

def generate_artificial_series(samples,prob_slope1,prob_cond,x_by_i,n):
	series = []
	for _ in range(samples):
		slope_artificial = np.zeros(n)
		slope_artificial[0],slope0_i = get_slope_val(nx,prob_slope1,x_by_i[0])

		for i in range(n-1):
			prob = prob_cond[i][slope0_i]
			slope_artificial[i+1],slope0_i = get_slope_val(nx,prob,x_by_i[i+1])

        # prob = prob_cond[1][slope1_i]
        # slope_artificial[2],slope2_i = get_slope_val(nx,prob,x_by_i[2])
        
        # prob = prob_cond[2][slope2_i]
        # slope_artificial[3],_ = get_slope_val(nx,prob,x_by_i[3])
        
		series.append(slope_artificial)
	return series

def artificial_series(slopes,intervalsx,n,samples):
    x_by_i = [None for _ in range(n)]
    print(type(slopes[:,0]),type(intervalsx))
    x_by_i[0],prob_slope1 = get_prob(slopes[:,0],intervalsx)
    for i in range(1,n):
	    x_by_i[i],_ = get_prob(slopes[:,i],intervalsx)
    # x_by_i[2],_ = get_prob(slopes[:,2],intervalsx)
    # x_by_i[3],_ = get_prob(slopes[:,3],intervalsx)
    
    prob_cond = []
    for i in range(n-1):
	    prob_cond.append(calculate_cond_xi_xi1(slopes[:,i:i+2],intervalsx))
    # prob_cond.append(calculate_cond_xi_xi1(slopes[:,1:3],intervalsx))
    # prob_cond.append(calculate_cond_xi_xi1(slopes[:,2:4],intervalsx))

    series_slopes = generate_artificial_series(samples,prob_slope1,prob_cond,x_by_i,n)
    
    return series_slopes

def norm(xs):
	mmax = max(xs)
	mmin = min(xs)

	return (xs-mmin)/(mmax-mmin)

def preprocess_original_breakpoints(filename,n):
    idxs,slopes,breakpoints,_ = read_original_breakpoints(filename,n)
    intervals = np.asarray([np.asarray(breakpoints2intervals(b)) for b in breakpoints])
    slopes = np.asarray([(np.arctan(s)*57.2958) for s in slopes])

    return idxs,slopes,intervals

if __name__ == "__main__":

    nx = 10
    ny = 10

    minx,maxx = 0,90
    deltax = (maxx-minx)/nx
    intervalsx = np.arange(minx,maxx+deltax,deltax)

    miny,maxy = 0,1
    deltay = (maxy-miny)/ny
    intervalsy = np.arange(miny,maxy+deltay,deltay)

    args = [intervalsx,intervalsy]

    # xs,ys = read_file_original(filename='data/plos_one_data_total.txt')
    # xs = np.asarray([norm(x) for x in xs])
    # ys = np.asarray([norm(y) for y in ys])

    for n in [2,3,4,5]:
        filename = 'data/plos_one_total_breakpoints_k4it.max100stop.if.errorFALSE_original0_data_filtered.txt'
        idxs,slopes,intervals = preprocess_original_breakpoints(filename,n)
        generate_freq_plots(slopes,intervals,n,'imgs/original0/scatter_')
        # generate_hist_plots(slopes,intervals,n,'imgs/original0/',args)
    '''
    ls = []
    for x in slopes:
        ls.append(len(x))
    unique,count = np.unique(ls,return_counts=True)
    for u,c in zip(unique,count):
        print(u,c)
    '''

    '''
    samples = 10000
    for n in [2,3,4,5]:
        _,slopes,breakpoints = read_original_breakpoints(samples_breakpoints='data/plos_one_total_breakpoints_k4it.max100stop.if.errorFALSE_original0_data_filtered.txt',n=n)
        intervals = np.asarray([np.asarray(breakpoints2intervals(b)) for b in breakpoints])
        slopes = np.asarray([(np.arctan(s)*57.2958) for s in slopes])

        # INTERVALO SEGUINDO PROB COND/ANGULO MEDIO DE CADA INTERVALO
        mean_slopes = np.mean(slopes,axis=0)
        articifial_xs = artificial_series(intervals,intervalsy,n,samples)
        mean_slopes = [mean_slopes.tolist()]*samples
        articifial_xs = np.asarray(articifial_xs)
        save(mean_slopes,articifial_xs,'data/original0/plos_one_artificial_intervals_slope_axis0_'+str(n)+'.txt')
        
        # INTERVALO SEGUINDO PROB COND/ANGULO ALEATÓRIO DE CADA INTERVALO
        artificial_slopes = np.random.choice(slopes.flatten(),size=samples*n).reshape(samples,n)
        save(artificial_slopes,articifial_xs,'data/original0/plos_one_artificial_intervals_slope_random_'+str(n)+'.txt')

        # # TUDO ALEATORIO (qualquer eixo)
        artificial_slopes = (np.random.rand(samples,n)*maxx)
        artificial_intervals = np.random.rand(samples,n)
        save(artificial_slopes,artificial_intervals,'data/original0/plos_one_artificial_all_random_'+str(n)+'.txt')

        # SLOPES SEGUINDO PROB COND/ANGULO MEDIO DE CADA INTERVALO
        mean_intervals = np.mean(intervals,axis=0)
        articifial_slopes = artificial_series(slopes,intervalsx,n,samples)
        mean_intervals = [mean_intervals.tolist()]*samples
        mean_intervals = np.asarray(mean_intervals)
        save(articifial_slopes,mean_intervals,'data/original0/plos_one_artificial_slopes_interval_axis0_'+str(n)+'.txt')
        
        # # INTERVALO SEGUINDO PROB COND/ANGULO ALEATÓRIO DE CADA INTERVALO
        artificial_intervals = np.random.choice(intervals.flatten(),size=samples*n).reshape(samples,n)
        save(articifial_slopes,artificial_intervals,'data/original0/plos_one_artificial_slopes_interval_random_'+str(n)+'.txt')

        # INTERVALOS E SLOPES SEGUINDO O MODELO
        articifial_intevals = artificial_series(intervals,intervalsy,n,samples)        
        # articifial_intevals = np.asarray(articifial_intevals)
        articifial_slopes = artificial_series(slopes,intervalsx,n,samples)
        save(articifial_slopes,articifial_intevals,'data/original0/plos_one_artificial_intervals_slopes_'+str(n)+'.txt')

    '''