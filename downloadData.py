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

wosArticles = WOSGetter.GZJSONLoad("wosPlosOne2016_citations.json.gz");
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

# what are your inputs, and what operation do you want to 
# perform on each input. For example...
DOIs = [article["DI"] for article in wosArticles];
chunkSize = 1000;
totalChunks = math.ceil(len(DOIs)/chunkSize);
for chunkIndex in range(14,totalChunks):
	inputs = DOIs[chunkIndex*chunkSize:min((chunkIndex+1)*chunkSize,len(DOIs))];
	totalProcessed = 0;
	counter = Value(c_int);
	counter_lock = Lock();
	counterMaximum = len(inputs)-1;
	def processInput(DOIURL):
		if(counter.value%10==0):
			print("Tokenizing: %d/%d             "%(counter.value,counterMaximum),end="\r");
		with counter_lock:
			counter.value += 1;
		doi = DOIURL;#'10.1371/journal.pone.0073791';
		if(len(doi)>0):
			jsonData = getURL('http://alm.plos.org/api/v5/articles?api_key=3pezRBRXdyzYW6ztfwft&ids='+urllib.parse.quote(doi, safe='')+'&info=detail');
			return jsonData;
		else:
			return None;
	num_cores = 30;#multiprocessing.cpu_count()
		    
	plosOneJSONData = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs);
	WOSGetter.GZJSONDump(plosOneJSONData,"/Volumes/Remote/PLOSONE/plosoneJSONChunk-%d.json.gz"%chunkIndex);
	print("Done                   %d/%d"%(chunkIndex,totalChunks));






