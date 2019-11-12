import json
import xnet
import glob
import numpy as np
import matplotlib.pyplot as plt

from igraph import *
from scipy import signal
from read_file import load_data
from itertools import combinations
from collections import defaultdict

complete_data = open('data/plos_one_2019_subj_areas.json','r').read()
# complete_data = open('data/wosPlosOne2016_citations.json','r').read()
complete_data = json.loads(complete_data)

# new_data = open('data/papers_plos_data_time_series2_filtered.json','r').read()
# new_data = json.loads(new_data)

# a = set()
# b = set()
# c = 0
# for article in complete_data:
# 	# sample = complete_data[doi]
# 	a.add(article['WC'])
# 	b.add(article['SC'])
# 	c += 1
# print(a,b,c)

'''
# CRIA AS REDES POR ANO
file = 'data/plos_one_2019_subj_areas.json'
content = open(file,'r').read()
json_content = json.loads(content)

for year in range(2008,2018):
	vertices = set()
	edges = []
	for doi,paper in json_content.items():
		if len(paper['time_series']['months']) == 0:
			continue
		c_year = int(float(paper['time_series']['months'][0]))
		if year <= c_year and year + 4 >= c_year:
		
			subs =  paper['infos']['subj_areas']

			vertices |= set(subs)
			combs = combinations(subs,2)
			for pair in combs:
				edges.append(pair)

	g = Graph()
	g.add_vertices(len(vertices))
	g.vs['name'] = list(vertices)
	g.add_edges(edges)
	g.es['weight'] = 1
	g.simplify(combine_edges=sum)

	i = 0
	for c in g.community_multilevel(weights='weight'):
		for idx in c:
			g.vs[idx]['comm'] = i

		i += 1

	xnet.igraph2xnet(g,'data/subj_areas/nets/all_with_comm_%d_4.xnet'%year)
'''


#data_breaks = load_data('data/plos_one_2019_breakpoints_k4_original1_data_filtered.txt')

'''
def incr_decr(dois,x0,x1):
	incr = []
	decr = []
	for sample in data_breaks:
	    if not sample[0] in dois:
	    	continue
	    slopes = sample[1]
	    breakpoints = sample[2]
	    n = len(breakpoints)
	    delta_time = sample[3][-1] - sample[3][0]
	    begin = sample[3][0]
	    for i in range(n):
	        moment = begin + delta_time*breakpoints[i]
	        if moment >= x0 and moment < x1:
	        	if slopes[i+1] > slopes[i]:
		            incr.append(moment)
		        else:
		            decr.append(moment)

	# incr = np.asarray(incr)
	# decr = np.asarray(decr)
	return incr,decr

def jaccard(s1,s2):
	inter = len(set(s1) & set(s2))
	uni = len(set(s1) | set(s2))
	return float(inter)/float(uni)

def gaussian(x, a, mu, sig):
    return a*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

files = glob.glob('data/subj_areas/nets/all_with_comm_*.xnet')

gs = []
for file in files:
	gs.append(xnet.xnet2igraph(file))

year = 2008
comms = set(gs[0].vs['comm'])
map_to_all = [[(year,c)] for c in comms]

for i in range(len(gs)-3):
	g1 = gs[i]
	g2 = gs[i+1]

	comms1 = defaultdict(lambda:[])
	comms2 = defaultdict(lambda:[])
	for v in g1.vs:
		comms1[v['comm']].append(v['name'])
	for v in g2.vs:
		comms2[v['comm']].append(v['name'])
	
	if len(comms1) <= 0 or len(comms2) <= 0:
		continue
	for c1,vtxs1 in comms1.items():
		sims = []
		for c2,vtxs2 in comms2.items():
			sim = jaccard(vtxs1,vtxs2)
			sims.append((sim,c2))
		s_max = max(sims)

		if s_max[0] < 0.2:
			continue
		print(sims)

		stop = False
		N = len(map_to_all)
		for j in range(N):
			m = map_to_all[j]
			if m[-1][1] == c1 and year == m[-1][0]:
				m.append((year+1,s_max[1]))
				stop = True
				break

		if not stop:
			map_to_all.append([(year,c1),(year+1,s_max[1])])
	year += 1

x_gaussian = np.arange(-10,10,0.2)
gauss = gaussian(x_gaussian,1,0,0.2)

bins = 100

fig,axs = plt.subplots(len(map_to_all),1,sharex=True,sharey=True,figsize=(12,3*len(map_to_all)))
for i,group in enumerate(map_to_all):
	if len(group) == 1:
		continue
	print(i,group)
	incr,decr = [],[]
	for year,c in group:
		dois = []
		f = glob.glob('data/subj_areas/nets/all_with_comm_%s.xnet'%year)[0]
		# print(f)
		g = xnet.xnet2igraph(f)
		words = set(g.vs.select(comm_eq=c)['name'])
		for doi,paper in complete_data.items():
			for w in paper['infos']['subj_areas']:
				if w in words:
					dois.append(doi)
					break
		dois = set(dois)
		
		incr_temp,decr_temp = incr_decr(dois,year,year+1)
		incr += incr_temp
		decr += decr_temp

	range0 = (2008,2020)
	hist0,bins_edges0 = np.histogram(incr,bins=bins,range=range0)
	hist1,bins_edges1 = np.histogram(decr,bins=bins,range=range0)

	# plt.figure(figsize=(12,3))
	y0 = np.convolve(hist0,gauss,mode='same')
	y1 = np.convolve(hist1,gauss,mode='same')
	x = np.arange(bins_edges0[0],bins_edges1[-1],(bins_edges1[-1]-bins_edges1[0])/len(y0))
	axs[i].set_title("%d - %d"%(group[0][0],group[-1][0]))
	axs[i].bar(x-0.05,y0,width=0.05,label='incr')
	axs[i].bar(x,y1,width=0.05,label='decr')
	axs[i].legend()

plt.tight_layout()
plt.savefig('hist_areas.pdf')
'''

f = 'data/subj_areas/nets/all.xnet'
g = xnet.xnet2igraph(f)
print(g.vs.attributes())

i = 0
for c in g.community_multilevel(weights='weight'):
    for idx in c:
        #g.vs[idx]['comm'] = i
        print(g.vs[idx]['name'],end=' ')
        i += 1
    print()
# N = len(set(g.vs['comm']))
# fig,axs = plt.subplots(N,1,sharex=True,sharey=True,figsize=(12,3*N))

# for i in range(N):

# 	incr,decr = [],[]
# 	dois = []
# 	words = set(g.vs.select(comm_eq=i)['name'])
# 	print(list(words)[:10])
# 	for doi,paper in complete_data.items():
# 		for w in paper['infos']['subj_areas']:
# 			if w in words:
# 				dois.append(doi)
# 				break
# 	dois = set(dois)
		
# 	incr,decr = incr_decr(dois,2008,2020)

# 	range0 = (2008,2020)

# 	hist0,bins_edges0 = np.histogram(incr,bins=bins,range=range0)
# 	hist1,bins_edges1 = np.histogram(decr,bins=bins,range=range0)

# 	y0 = np.convolve(hist0,gauss,mode='same')
# 	y1 = np.convolve(hist1,gauss,mode='same')
# 	x = np.arange(bins_edges0[0],bins_edges1[-1],(bins_edges1[-1]-bins_edges1[0])/len(y0))
# 	axs[i].set_title(i)
# 	axs[i].bar(x-0.05,y0,width=0.05,label='incr')
# 	axs[i].bar(x,y1,width=0.05,label='decr')
# 	axs[i].legend()

# plt.tight_layout()
# plt.savefig('hists_areas_all.pdf')
