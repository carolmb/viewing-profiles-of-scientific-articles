import WOSGetter;
import igraph;
import xnet;
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Value, Lock, Process
from ctypes import c_int
import json;
import sys;
import math;
import traceback;

import urllib;
import os.path
import urllib.request
from urllib.parse import urlencode
from http.cookiejar import CookieJar,MozillaCookieJar

#wosArticles = WOSGetter.GZJSONLoad("wosPlosOne2016_citations.json.gz");
#wosNetwork = xnet.xnet2igraph("wosPlosOne2016_cocitation.xnet");



cj = MozillaCookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
urllib.request.install_opener(opener)

cookie_file=os.path.abspath('./cookies.txt')

requestTrials = 3;
connectionTimeout = 60;
processTrials=3;
errorVerbose = True;

def dorequest(url,cj=None,data=None,timeout=10,encoding='UTF-8'):
	data = urlencode(data).encode(encoding) if data else None

	request = urllib.request.Request(url)
	request.add_header('User-Agent','Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)')
	f = urllib.request.urlopen(request,data,timeout=timeout)
	return f.read()


def getURL(url):
	k = 0;
	theContents = None;
	while((theContents is None) and k<requestTrials):
		k+=1;
		try:
			theContents = dorequest(url,timeout=connectionTimeout).decode("utf-8-sig")
		except urllib.error.URLError as e:
			theContents=None;
			if(errorVerbose):
				sys.stdout.write("\n##"+("="*40)+"\n");
				sys.stdout.write("##%r (%s) url = %s\n"%(e,str(sys.exc_info()[0]),url));
				traceback.print_exc(file=sys.stdout);
				sys.stdout.write("\n##"+("="*40)+"\n");
				sys.stdout.flush();
		except:
			theContents=None;
			if(errorVerbose):
				sys.stdout.write("\n##"+("="*40)+"\n");
				sys.stdout.write("##A general exception occurred (%s). Could not process URL: %s\n"%(str(sys.exc_info()[0]),url));
				traceback.print_exc(file=sys.stdout);
				sys.stdout.write("\n##"+("="*40)+"\n");
				sys.stdout.flush();
	if(theContents is None):
		sys.stdout.write("\n!!WARNING: m trials reached. Could not process URL: %s\n"%url);
		sys.stdout.flush();
	return theContents;

# # what are your inputs, and what operation do you want to 
# # perform on each input. For example...
# DOIs = [article["DI"] for article in wosArticles];
# chunkSize = 1000;
# totalChunks = math.ceil(len(DOIs)/chunkSize);
# for chunkIndex in range(14,totalChunks):
# 	inputs = DOIs[chunkIndex*chunkSize:min((chunkIndex+1)*chunkSize,len(DOIs))];
# 	totalProcessed = 0;
# 	counter = Value(c_int);
# 	counter_lock = Lock();
# 	counterMaximum = len(inputs)-1;
# 	def processInput(DOIURL):
# 		if(counter.value%10==0):
# 			print("Tokenizing: %d/%d             "%(counter.value,counterMaximum),end="\r");
# 		with counter_lock:
# 			counter.value += 1;
# 		doi = DOIURL;#'10.1371/journal.pone.0073791';
# 		if(len(doi)>0):
# 			jsonData = getURL('http://alm.plos.org/api/v5/articles?api_key=3pezRBRXdyzYW6ztfwft&ids='+urllib.parse.quote(doi, safe='')+'&info=detail');
# 			return jsonData;
# 		else:
# 			return None;
# 	num_cores = 30;#multiprocessing.cpu_count()
		    
# 	plosOneJSONData = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs);
# 	WOSGetter.GZJSONDump(plosOneJSONData,"/Volumes/Remote/PLOSONE/plosoneJSONChunk-%d.json.gz"%chunkIndex);
# 	print("Done                   %d/%d"%(chunkIndex,totalChunks));




def date2float(date):
	dateSplitted = date.split("/")
	return int(dateSplitted[1])+(0.5+int(dateSplitted[0])-1)/12;



clusterIndex=0;
completeTimeSeries = [];
while(1):
	print("Calculating: %d             "%(clusterIndex),end="\r");
	plosData = WOSGetter.GZJSONLoad("data/plosoneJSONChunk-%d.json.gz"%clusterIndex);
	if(not plosData):
		break;
	chunkTimeseries = [];
	for plosDocumentData in plosData:
		if(not plosDocumentData):
			completeTimeSeries.append([]);
			continue;

		try:
			plosJSONData = json.loads(plosDocumentData);
		except:
			if(errorVerbose):
				sys.stdout.write("\n##"+("="*40)+"\n");
				sys.stdout.write("##A general exception occurred (%s). Could not process URL: %s\n"%(str(sys.exc_info()[0]),plosDocumentData));
				traceback.print_exc(file=sys.stdout);
				sys.stdout.write("\n##"+("="*40)+"\n");
				sys.stdout.flush();
			plosJSONData = None;

		if(not plosJSONData):
			completeTimeSeries.append([]);
			continue;

		if(len(plosJSONData["data"])>1):
			print("Len larger than 1 (%d)"%len(plosJSONData["data"]));

		if(len(plosJSONData["data"])<1):
			completeTimeSeries.append([]);
			continue;

		selectedData = None;
		for source in plosJSONData["data"][0]["sources"]:
			if(source["display_name"]=="Counter"):
				selectedData = source;

		timeSeries = {};
		timeSeries["pdf"] = {};
		timeSeries["html"] = {};
		timeSeries["total"] = {};
		timeSeries["xml"] = {};

		for item in selectedData["by_month"]:
			date = "%02d/%04d"%(int(item["month"]),int(item["year"]));
			timeSeries["html"][date] = int(item["html"]);
			timeSeries["pdf"][date] = int(item["pdf"]);
			timeSeries["total"][date] = int(item["total"]);


		for item in selectedData["events"]:
			date = "%02d/%04d"%(int(item["month"]),int(item["year"]));
			timeSeries["html"][date] = int(item["html_views"]);
			timeSeries["pdf"][date] = int(item["pdf_views"]);
			timeSeries["xml"][date] = int(item["xml_views"]);
			timeSeries["total"][date] = int(timeSeries["html"][date]+timeSeries["pdf"][date]+timeSeries["xml"][date]);

		newTimeSeries = {};
		for featureName in timeSeries:
			newTimeSeries[featureName] = sorted(timeSeries[featureName].items(),key=lambda d:date2float(d[0]));
		chunkTimeseries.append(newTimeSeries);
	completeTimeSeries+=chunkTimeseries;
	clusterIndex+=1;

WOSGetter.GZJSONDump(completeTimeSeries,"plosone2016_hits.json.json.gz");

# import numpy as np;
# import matplotlib
# matplotlib.use('Qt5Agg');
# import matplotlib.pyplot as plt;
# sampledSet = []+completeTimeSeries;
# np.random.shuffle(sampledSet);
# sampledSet = sampledSet[0:40];

# featureName = "total";
# for sample in sampledSet:
# 	x = [date2float(data[0]) for data in sample[featureName]];
# 	y = np.cumsum([data[1] for data in sample[featureName]]);
# 	plt.plot(x,y);
# plt.show();





