import json
import time
import requests
from bs4 import BeautifulSoup
from multiprocessing.dummy import Pool as ThreadPool 

json_filename = 'papers_plos_data_time_series2_filtered.json'
json_file = json.loads(open(json_filename,'r').read())

def get_subj_areas(url):
	content1 = requests.get(url).content
	soup1 = BeautifulSoup(content1, 'html.parser')

	try:
		subs = []
		for a in soup1.find('ul', {'id':'subjectList'}).find_all('a'):
			subs.append(a.text)
		return subs
	except:
		return []
	return subs

def save(data,filename):
	with open(filename, 'w') as f:
		json.dump(data, f)

def chunks_subj_areas(arg):
	idx,dois = arg[0],arg[1]
	json_file = {}
	for doi in dois:
		url = 'https://journals.plos.org/plosone/article?id='+doi
		subs = get_subj_areas(url)
		print(subs)
		json_file[doi] = subs
		time.sleep(6)
	save(json_file,'subj_areas_'+str(idx)+'.json')

dois = list(json_file.keys())
n = int(len(dois)/10)
chunks = [(j,dois[i:i + n]) for j,i in enumerate(range(0, len(dois), n))]

chunks = chunks[3:]
for i,dois in chunks:
	print(i,len(dois))
pool = ThreadPool(7)
results = pool.map(chunks_subj_areas, chunks)
