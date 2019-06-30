import WOSGetter;
import igraph;
import xnet;
import json;
import sys;
import math;

import numpy as np;
import matplotlib
# matplotlib.use('Qt5Agg');
import matplotlib.pyplot as plt;


def date2float(date):
	dateSplitted = date.split("/")
	return int(dateSplitted[1])+(0.5+int(dateSplitted[0])-1)/12;



#wosArticles = WOSGetter.GZJSONLoad("wosPlosOne2016_citations.json.gz");
#wosNetwork = xnet.xnet2igraph("wosPlosOne2016_cocitation.xnet");
completeTimeSeries = WOSGetter.GZJSONLoad("plosone2016_hits.json.gz")

sampledSet = []+completeTimeSeries;
filteredSampledSet = [];
for sample in sampledSet:
	if (len(sample)>0 and len(sample["total"])>0) and date2float(sample["total"][0][0]) < 2010:
		filteredSampledSet.append(sample);
np.random.shuffle(filteredSampledSet);
sampledSet = filteredSampledSet[0:40];

featureName = "total";
with open("samples.txt","w") as fd:
	for sample in sampledSet:
		x = [date2float(data[0]) for data in sample[featureName]];
		y = np.cumsum([data[1] for data in sample[featureName]]);
		fd.write("\t".join([str(v) for v in x]));
		fd.write("\n");
		fd.write("\t".join([str(v) for v in y]));
		fd.write("\n");
		plt.plot(x,y);
plt.savefig("samples.pdf");
plt.close();


