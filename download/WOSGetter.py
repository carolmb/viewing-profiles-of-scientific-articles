import urllib;
import json;
import gzip;
import sys;
import itertools;
from pyquery import PyQuery as pq;
import os;
import math;
import traceback;


import os.path
import urllib.request
from urllib.parse import urlencode
from http.cookiejar import CookieJar,MozillaCookieJar

cj = MozillaCookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
urllib.request.install_opener(opener)

cookie_file=os.path.abspath('./cookies.txt')

def load_cookies(cj,cookie_file):
	cj.load(cookie_file)
def save_cookies(cj,cookie_file):
	cj.save(cookie_file,ignore_discard=True,ignore_expires=True)

def dorequest(url,cj=None,data=None,timeout=10,encoding='UTF-8'):
	data = urlencode(data).encode(encoding) if data else None

	request = urllib.request.Request(url)
	request.add_header('User-Agent','Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)')
	f = urllib.request.urlopen(request,data,timeout=timeout)
	return f.read()

def dopost(url,cj=None,data=None,timeout=10,encoding='UTF-8'):
	body = dorequest(url,cj,data,timeout,encoding)
	return body.decode(encoding)




def GZJSONDump(obj, filename):
	with gzip.open(filename, 'wb') as fp:
		iterable = json._default_encoder.iterencode(obj)
		for chunk in iterable:
			fp.write(chunk.encode("ascii"));

def GZJSONLoad(filename):
	loadedData = None;
	try:
		with gzip.open(filename, 'rb') as fp:
			loadedData = json.loads(fp.read().decode("ascii"));
	except:
		pass;
	return loadedData;

WOSReference = {
	"FN":"File Name",
	"VR":"Version Number",
	"PT":"Publication Type",
	"AU":"Authors",
	"AF":"Author Full Name",
	"CA":"Group Authors",
	"TI":"Document Title",
	"ED":"Editors",
	"SO":"Publication Name",
	"SE":"Book Series Title",
	"BS":"Book Series Subtitle",
	"LA":"Language",
	"DT":"Document Type",
	"CT":"Conference Title",
	"CY":"Conference Date",
	"HO":"Conference Host",
	"CL":"Conference Location",
	"SP":"Conference Sponsors",
	"DE":"Author Keywords",
	"ID":"Keywords Plus",
	"AB":"Abstract",
	"C1":"Author Address",
	"RP":"Reprint Address",
	"EM":"E-mail Address",
	"FU":"Funding Agency and Grant Number",
	"FX":"Funding Text",
	"CR":"Cited References",
	"NR":"Cited Reference Count",
	"TC":"Times Cited",
	"PU":"Publisher",
	"PI":"Publisher City",
	"PA":"Publisher Address",
	"SC":"Subject Category",
	"SN":"ISSN",
	"BN":"ISBN",
	"J9":"29-Character Source Abbreviation",
	"JI":"ISO Source Abbreviation",
	"PD":"Publication Date",
	"PY":"Year Published",
	"VL":"Volume",
	"IS":"Issue",
	"PN":"Part Number",
	"SU":"Supplement",
	"SI":"Special Issue",
	"BP":"Beginning Page",
	"EP":"Ending Page",
	"AR":"Article Number",
	"PG":"Page Count",
	"DI":"Digital Object Identifier (DOI)",
	"SC":"Subject Category",
	"GA":"Document Delivery Number",
	"UT":"Unique Article Identifier",
	"ER":"End of Record",
	"EF":"End of File",
	"WC":"JCR Category",
	"IMPACT" : "Impact factor",
	"5YEARIMPACT": "5-year impact factor",
	"RANK":"Rank in category",
	"MAXRANK":"Journals in category",
	"NORMRANK":"Normalized rank in category",
	"CITATIONS": "References",
	"Index": "Network index",
	"ADDRESS": "Address information"
}




def WOSPrintEntry(entry):
	"""
	Print an entry coveying an article.
	
	Args:
		entry (dict): The article data to be printed .
	"""
	for key in entry:
		if(len(entry[key])>0 and key in WOSReference):
			print("%s : %s"%(WOSReference[key],entry[key]));

def WOSParseOutput(urlContents):
	"""
	Parse a WOS datafile as input and returns the articles information.
	
	Args:
		entry (stream): The data stream to be parsed.

	Returns:
		list: The data containg the parsed information of the articles.

	"""
	lines = urlContents.replace("\r","").split("\n");
	entries = [];
	if(len(lines)>0):
		header = lines[0].split("\t");
		if(len(header)>4):
			for lineIndex in range(1,len(lines)):
				line = lines[lineIndex];
				if(len(line)>0):
					entry = {};
					lineSplit = line.split("\t");
					if(len(lineSplit)>len(header)):
						lineSplit = lineSplit[0:-1];
					if(len(lineSplit)>0):
						for attributeIndex in range(0,len(lineSplit)):
							attributeName = header[attributeIndex];
							attribute = lineSplit[attributeIndex];
							entry[attributeName] = attribute;
						entries.append(entry);
	return entries;

class WOSGetter(object):
	"""Initializes a WOS object and obtain a SID if not provided.

	Args:
		SSID (str, optional): Set the WOS section ID.
		verbose (bool, optional): Enable verbose mode.

	Attributes:
		SSID (str): The WOS section ID.
		verbose (int): Exception error code.
		journal (int): Information of journals saved.
		timeout (int): Default timeout for trying again (in secs).

	"""
	def __init__(self,SSID=None, verbose = False, errorVerbose = False, requestTrials=10, processTrials=3):
		super(WOSGetter, self).__init__()
		self.requestTrials = requestTrials;
		self.processTrials = processTrials;
		self.timeout = 60;#30 secs
		self.verbose = verbose;
		self.errorVerbose = errorVerbose
		self.WOSURL = "http://apps-webofknowledge.ez67.periodicos.capes.gov.br";
		if(SSID==None):
			self.updateSSID();
		else:
			self.SSID = SSID;
		self.journals = {};

	def getURLContents(self,url):
		k = 0;
		theContents = None;
		while((theContents is None) and k<self.requestTrials):
			k+=1;
			try:
				theContents = dorequest(url,timeout=self.timeout).decode("utf-8-sig")
				#fURL = urllib.request.urlopen(url,timeout=self.timeout)
				#theContents = fURL.read().decode("utf-8-sig");
				#fURL.close();
			except urllib.error.URLError as e:
				theContents=None;
				if(self.errorVerbose):
					sys.stdout.write("\n##"+("="*40)+"\n");
					sys.stdout.write("##%r (%s) url = %s\n"%(e,str(sys.exc_info()[0]),url));
					traceback.print_exc(file=sys.stdout);
					sys.stdout.write("\n##"+("="*40)+"\n");
					sys.stdout.flush();
			except:
				theContents=None;
				if(self.errorVerbose):
					sys.stdout.write("\n##"+("="*40)+"\n");
					sys.stdout.write("##A general exception occurred (%s). Could not process URL: %s\n"%(str(sys.exc_info()[0]),url));
					traceback.print_exc(file=sys.stdout);
					sys.stdout.write("\n##"+("="*40)+"\n");
					sys.stdout.flush();
		if(theContents is None):
			sys.stdout.write("\n!!WARNING: m trials reached. Could not process URL: %s\n"%url);
			sys.stdout.flush();
		return theContents;

	def pyQueryFromURL(self,url):
		theContents = self.getURLContents(url);
		if(theContents is not None):
			return pq(theContents);
		else:
			return None;

	def updateSSID(self):
		"""Obtains and sets a new section ID from WOS to this object.
		"""
		SSID = None;
		k=0;
		while (SSID==None and k<self.processTrials):
			d = self.pyQueryFromURL(self.WOSURL);
			if(d is not None):
				SSID = d("input#SID").val();
			k+=1;
		if(SSID==None):
			raise ValueError('Maximum tries to obtain SID reached.');
		if(self.verbose):
			print("Got SID: "+SSID);
		self.SSID = SSID;
		return SSID;

	def getCitations(self,WOScode):
		"""gets the references of an article defined by the wosCode.

		Args:
			WOScode (str): The WOS code used to retrieve the information.

		Returns:
			list: The list of references of the requested article.

		"""
		entries = [];
		wosURL = self.WOSURL+"/CitedRefList.do?product=WOS&search_mode=CitedRefList&SID="+self.SSID+"&colName=WOS&parentProduct=WOSrecid="+WOScode+"&UT="+WOScode;
		d = self.pyQueryFromURL(wosURL);

		#print(wosURL);
		entryCount = d('#hitCount\\.top:first').text();
		if(entryCount!="" and entryCount is not None):
			#print("entryCount = " + entryCount);
			qid = d('input[name$="qid"]:first').val();
			#print("qid = " + qid);
			resultsCount = int(entryCount);
			if(resultsCount>100000):
				if(self.verbose):
					print("%d found but only 100000 entries will be considered."%resultsCount);
				resultsCount = 100000;
			for x in range(0,resultsCount,500):
				fromString = str(x+1);
				toString = str(min(resultsCount,x+500));
				if(self.verbose):
					print("Getting cited papers between %s-%s"%(fromString,toString))
				downloadURL = self.WOSURL+"/OutboundService.do?action=go&viewType=fullRecord&product=WOS&mark_id=WOS&colName=WOS&search_mode=CitedRefList&locale=en_US&sortBy=CAU.A;CW.A;CY.D;CV.D;CG.A&mode=CitedRefList-outputService&qid="+qid+"&SID="+self.SSID+"&format=saveToFile&filters=filters=PMID+USAGEIND+AUTHORSIDENTIFIERS+ACCESSION_NUM+FUNDING+SUBJECT_CATEGORY+JCR_CATEGORY+LANG+IDS+PAGEC+SABBR+CITREFC+ISSN+PUBINFO+KEYWORDS+CITTIMES+ADDRS+CONFERENCE_SPONSORS+DOCTYPE+ABSTRACT+CONFERENCE_INFO+SOURCE+TITLE+AUTHORS++&selectedIds=&mark_to="+toString+"&mark_from="+fromString+"&count_new_items_marked=0&value(record_select_type)=range&markFrom="+fromString+"&markTo="+toString+"&fields_selection=PMID+USAGEIND+AUTHORSIDENTIFIERS+ACCESSION_NUM+FUNDING+SUBJECT_CATEGORY+JCR_CATEGORY+LANG+IDS+PAGEC+SABBR+CITREFC+ISSN+PUBINFO+KEYWORDS+CITTIMES+ADDRS+CONFERENCE_SPONSORS+DOCTYPE+ABSTRACT+CONFERENCE_INFO+SOURCE+TITLE+AUTHORS++&bib_fields_option=++&rurl=&save_options=tabWindowsUTF8";
				#print(downloadURL);
				downloadContents = self.getURLContents(downloadURL);
				entries += WOSParseOutput(downloadContents);
		return entries;

	def getPaper(self,WOScode):
		"""gets complete information of an article defined by the wosCode.

		Args:
			WOScode (str): The WOS code used to retrieve the information.

		Returns:
			dict: The information of the requested article.

		"""
		wosURL = self.WOSURL+"/CitedFullRecord.do?product=WOS&colName=WOS&SID="+self.SSID+"&search_mode=CitedFullRecord&isickref="+WOScode;
		
		d = self.pyQueryFromURL(wosURL);
		
		qid = d('input[name$="qid"]:first').val();
		qidNext = qid;
		#pq(url="http://apps.webofknowledge.com/AutoSave_UA_output.do?action=saveForm&SID="+self.SSID+"&product=UA&search_mode=output&value(saveToMenuDefault)=other&product=UA&search_mode=output");
		#pq(url="http://ets.webofknowledge.com/ETS/ets.do?refineString=null&displayUsageInfo=true&qid="+qidNext+"&mark_to=1&fileOpt=tabWinUTF8&displayCitedRefs=true&totalMarked=1&SID="+self.SSID+"&product=UA&mark_from=1&parentQid="+qid+"&displayTimesCited=true&sortBy=null&timeSpan=null&UserIDForSaveToRID=null&action=saveToFile&colName=WOS&filters=PMID%20USAGEIND%20AUTHORSIDENTIFIERS%20ACCESSION_NUM%20FUNDING%20SUBJECT_CATEGORY%20JCR_CATEGORY%20LANG%20IDS%20PAGEC%20SABBR%20CITREFC%20ISSN%20PUBINFO%20KEYWORDS%20CITTIMES%20ADDRS%20CONFERENCE_SPONSORS%20DOCTYPE%20ABSTRACT%20CONFERENCE_INFO%20SOURCE%20TITLE%20AUTHORS%20%20&excludeEventConfig=ExcludeIfFromFullRecPage");
		if(self.verbose):
			print("qid = " + qid);

		downloadURL = self.WOSURL+"/OutboundService.do?action=go&&&marked_list_candidates=1&excludeEventConfig=ExcludeIfFromFullRecPage&displayCitedRefs=true&displayTimesCited=true&displayUsageInfo=true&viewType=fullRecord&product=WOS&mark_id=WOS&colName=WOS&search_mode=CitedFullRecord&locale=en_US&recordID="+WOScode+"&view_name=WOS-CitedFullRecord-fullRecord&mode=OpenOutputService&qid="+qid+"&SID="+self.SSID+"&format=saveToFile&filters=PMID+USAGEIND+AUTHORSIDENTIFIERS+ACCESSION_NUM+FUNDING+SUBJECT_CATEGORY+JCR_CATEGORY+LANG+IDS+PAGEC+SABBR+CITREFC+ISSN+PUBINFO+KEYWORDS+CITTIMES+ADDRS+CONFERENCE_SPONSORS+DOCTYPE+ABSTRACT+CONFERENCE_INFO+SOURCE+TITLE+AUTHORS++&mark_to=1&mark_from=1&count_new_items_marked=0&use_two_ets=false&IncitesEntitled=yes&fields_selection=PMID+USAGEIND+AUTHORSIDENTIFIERS+ACCESSION_NUM+FUNDING+SUBJECT_CATEGORY+JCR_CATEGORY+LANG+IDS+PAGEC+SABBR+CITREFC+ISSN+PUBINFO+KEYWORDS+CITTIMES+ADDRS+CONFERENCE_SPONSORS+DOCTYPE+ABSTRACT+CONFERENCE_INFO+SOURCE+TITLE+AUTHORS++&save_options=tabWinUTF8";
		#print(downloadURL);

		downloadContents = self.getURLContents(downloadURL);
		entries = WOSParseOutput(downloadContents);

		if(len(entries)>0):
			if(entries[0]["J9"]):
				impactFactor = d("#tmp_ifactor_1 td:first").text();
				impactFactor5y = d("#tmp_ifactor_1 td:nth-child(2)").text();
				JCRRank = [[d(lineElement).text() for lineElement in d("td",domElement)] for domElement in d("#tmp_category_1 tr")];
				if(len(JCRRank)>1):
					JCRRank = JCRRank[1:];
				self.journals[entries[0]["J9"]] = {
					"IMPACT" : impactFactor,
					"5YEARIMPACT" : impactFactor5y,
					"RANK" : JCRRank
				}
			return entries[0];
		else:
			return None;

	def query(self,wosQuery,timeperiod = None, continueIfLimitReached = False):
		"""gets a list of articles returned by the requested query.

		Args:
			wosQuery (str): A WOS query.
			timeperiod (tuple,optional): A time range between 1900 and the current year.
			continueIfLimitReached (bool,optional): Continues even if the limit of 100000 articles is attained. In this case, only 100000 articles are returned.

		Returns:
			list: The list of articles returned by the query.

		"""
		wosQuery = urllib.parse.quote(wosQuery);
		rangeType = "Range+Selection"
		startYearString = "2010";
		endYearString = "2016";

		if(timeperiod!=None and len(timeperiod)>1):
			startYearString = str(timeperiod[0]);
			endYearString = str(timeperiod[1]);
			rangeType = "Year+Range";

		queryURL = self.WOSURL+"/WOS_AdvancedSearch.do?product=WOS&search_mode=AdvancedSearch&SID="+self.SSID+"&input_invalid_notice=Search+Error:+Please+enter+a+search+term.&input_invalid_notice_limits=+<br/>Note:+Fields+displayed+in+scrolling+boxes+must+be+combined+with+at+least+one+other+search+field.&action=search&replaceSetId=&goToPageLoc=SearchHistoryTableBanner&value(input1)="+wosQuery+"&value(searchOp)=search&x=89&y=435&value(select2)=LA&value(input2)=&value(select3)=DT&value(input3)=&value(limitCount)=14&limitStatus=collapsed&ss_lemmatization=On&ss_spellchecking=Suggest&SinceLastVisit_UTC=&SinceLastVisit_DATE=&range=ALL&period="+rangeType+"&startYear="+startYearString+"&endYear="+endYearString+"&editions=SCI&editions=SSCI&editions=AHCI&editions=ISTP&editions=ISSHP&editions=ESCI&update_back2search_link_param=yes&ss_query_language=&rs_sort_by=LC.D;PY.D;AU.A.en;SO.A.en;VL.D;PG.A"
		
		d = self.pyQueryFromURL(queryURL);
		print(queryURL);
		queryAElement = d(".historyResults:first a");
		resultsCount = int(queryAElement.text().replace(",",""));
		if(resultsCount>100000):
			resultsCount = 100000;
			if(continueIfLimitReached):
				return None;
			if(self.verbose):
				print("%d found but only 100000 entries will be considered."%resultsCount);
		entries = [];
		if(self.verbose):
			print("-- Results: %d --"%resultsCount);
		if(resultsCount>0):
			resultsURL = self.WOSURL+queryAElement.attr("href");
			if(self.verbose):
				print(resultsURL);
			
			d = self.pyQueryFromURL(resultsURL);
		
			updatedResultsURL = resultsURL+"&page=1&action=changePageSize&pageSize=50";
			if(self.verbose):
				print(updatedResultsURL);
			
			d = self.pyQueryFromURL(updatedResultsURL);
		
			qid = d('input[name$="qid"]:first').val();
			for x in range(0,resultsCount,500):
				fromString = str(x+1);
				toString = str(min(resultsCount,x+500));
				if(self.verbose):
					print("Getting papers between %s-%s"%(fromString,toString))
				downloadURL = self.WOSURL+"/OutboundService.do?action=go&&selectedIds=&displayCitedRefs=true&displayTimesCited=true&displayUsageInfo=true&viewType=summary&product=WOS&mark_id=WOS&colName=WOS&search_mode=AdvancedSearch&locale=en_US&view_name=WOS-summary&sortBy=LC.D;PY.D;AU.A.en;SO.A.en;VL.D;PG.A&mode=OpenOutputService&qid="+qid+"&SID="+self.SSID+"&format=saveToFile&filters=PMID+USAGEIND+AUTHORSIDENTIFIERS+ACCESSION_NUM+FUNDING+SUBJECT_CATEGORY+JCR_CATEGORY+LANG+IDS+PAGEC+SABBR+CITREFC+ISSN+PUBINFO+KEYWORDS+CITTIMES+ADDRS+CONFERENCE_SPONSORS+DOCTYPE+ABSTRACT+CONFERENCE_INFO+SOURCE+TITLE+AUTHORS++&mark_to="+toString+"&mark_from="+fromString+"&queryNatural="+wosQuery+"&count_new_items_marked=0&use_two_ets=false&IncitesEntitled=yes&value(record_select_type)=range&markFrom="+fromString+"&markTo="+toString+"&fields_selection=PMID+USAGEIND+AUTHORSIDENTIFIERS+ACCESSION_NUM+FUNDING+SUBJECT_CATEGORY+JCR_CATEGORY+LANG+IDS+PAGEC+SABBR+CITREFC+ISSN+PUBINFO+KEYWORDS+CITTIMES+ADDRS+CONFERENCE_SPONSORS+DOCTYPE+ABSTRACT+CONFERENCE_INFO+SOURCE+TITLE+AUTHORS++&save_options=tabWinUTF8";
				downloadContents = self.getURLContents(downloadURL);
				entries += WOSParseOutput(downloadContents);
		return entries;

def WOSGetAggregatedCitations(wosAggregatedEntry,jobIndex,jobCount,countToAggregate,filename,forceUpdate=False):
	"""requests the citations of. TODO: COMPLETE THIS
	
	Args:
		wosAggregatedEntry (dict): A dictionary entry.
		jobIndex (int): The index of this job.
		jobCount (int): The number of jobs.
		countToAggregate (int): The number of agregated results for each job.
		filename (str): A prefixname to be used on temporary files.
		
	Returns:
		list: The list of articles with references.

	"""
	tempFilename = "%s-%d_citations.tmp/aggregated_%d.json.gz"%(filename,countToAggregate,jobIndex);
	
	wosAggregatedResults = GZJSONLoad(tempFilename);
	
	if(wosAggregatedResults is None or len(wosAggregatedResults)==0):
		wosAggregatedResults = [];
		wosGetter = None;
		sys.stdout.write("s%d"%jobIndex);
		sys.stdout.flush();
		wosGetter = WOSGetter();
		for wosEntry in wosAggregatedEntry:
			if(not forceUpdate and "CITATIONS" in wosEntry):
				wosAggregatedResults.append(wosEntry);
				continue;
			processed = False;
			paperEntry = None;
			citations = [];
			currentPaperCode = wosEntry["UT"];
			trialsA=0;
			while((not processed) and (trialsA<wosGetter.processTrials)):
				trialsA+=1;
				try:
					paperEntry = wosEntry;
					if("AB" not in paperEntry or "TI" not in paperEntry):
						retrievedPaperEntry = wosGetter.getPaper(currentPaperCode);
						if(retrievedPaperEntry):
							paperEntry.update(retrievedPaperEntry);
					citationsEntries = wosGetter.getCitations(currentPaperCode);
					citations = [citationEntry["UT"] for citationEntry in citationsEntries if "UT" in citationEntry];
					processed=True;
				except:
					sys.stdout.write("X(%s, %s)%d\n#"%(str(sys.exc_info()[0]),currentPaperCode,jobIndex));
					traceback.print_exc(file=sys.stdout);
					sys.stdout.flush();
				if(processed):
					wosGetter.updateSSID();
				if(not processed):
					sys.stdout.write("\n#WARNING: could not download citations for %s."%(currentPaperCode));
					sys.stdout.flush();	
				sys.stdout.write(".");
				sys.stdout.flush();
			paperEntry["CITATIONS"] = citations;
			wosAggregatedResults.append(paperEntry);
		GZJSONDump(wosAggregatedResults,tempFilename);
	sys.stdout.write("\n#Finished processing: %d entries (Job:%d/%d)\n"%(len(wosAggregatedEntry),jobIndex,jobCount));
	sys.stdout.flush();
	return wosAggregatedResults;

def WOSAggregateEntries(wosEntries,maximumAggregateCount):
	"""requests the citations of. TODO: COMPLETE THIS
	
	Args:
		wosAggregatedEntry (str): A WOS query.
		jobIndex (int): A time range between 1900 and the current year.
		jobCount (int): Continues even if the limit of 100000 articles is attained. In this case, only 100000 articles are returned.
		
	Returns:
		list: The list of articles with references.

	"""
	wosAggregatedEntries = [];
	currentAggregatedEntries = [];
	for entry in wosEntries:
		currentAggregatedEntries.append(entry);
		if(len(currentAggregatedEntries)>=maximumAggregateCount):
			wosAggregatedEntries.append(currentAggregatedEntries);
			currentAggregatedEntries = [];
	if(len(currentAggregatedEntries)>0):
		wosAggregatedEntries.append(currentAggregatedEntries);
	return wosAggregatedEntries;

def WOSGetArticlesFromQuery(query,timeperiod):
	"""requests the citations of. TODO: COMPLETE THIS
	
	Args:
		wosAggregatedEntry (str): A WOS query.
		jobIndex (int): A time range between 1900 and the current year.
		jobCount (int): Continues even if the limit of 100000 articles is attained. In this case, only 100000 articles are returned.
		
	Returns:
		list: The list of articles with references.

	"""
	print("Getting papers from query: [%s] from %d-%d"%(query,timeperiod[0],timeperiod[1])); 
	if(timeperiod[0]>timeperiod[1] and timeperiod[0]>1900 and timeperiod[1]<=datetime.datetime.now().year):
		raise ValueError("Time period must be a range between 1900 and %d. The time period must also convey a postive range."%datetime.datetime.now().year);
	queryResults = [];
	wosGetter = WOSGetter(verbose=True);
	if(timeperiod[0]==timeperiod[1]):
		queryResults = wosGetter.query(query,timeperiod=timeperiod,continueIfLimitReached = False);
	else:
		queryResults = wosGetter.query(query,timeperiod=timeperiod, continueIfLimitReached = True);
		if(queryResults is None):
			midYear = math.floor((timeperiod[0]+timeperiod[1])*0.5);
			queryResultsFirst = WOSGetArticlesFromQuery(query,[timeperiod[0],midYear]);
			queryResultsLast = WOSGetArticlesFromQuery(query,[midYear+1,timeperiod[1]]);
			queryResults = queryResultsFirst+queryResultsLast;
	return queryResults;


def WOSArticlesExpandCited(wosArticles):
	codesSet = set();
	neighArticles = [];
	for article in wosArticles:
		if(article["UT"] not in codesSet):
			codesSet.add(article["UT"]);
			neighArticles.append(article);
			if("CITATIONS" in article):
				for cited in article["CITATIONS"]:
					if(cited not in codesSet):
						codesSet.add(cited);
						neighArticles.append({"UT":cited});
	return neighArticles;


