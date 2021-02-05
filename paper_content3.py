import json,time
from tqdm import tqdm,trange
import urllib.request
from bs4 import BeautifulSoup
# from read_file import load_data
from multiprocessing import Pool as ThreadPool
import glob

def get_content(doi):
	try:
		url = 'https://journals.plos.org/plosone/article?id=' + doi
		with urllib.request.urlopen(url) as response:
			html = response.read()

		soup = BeautifulSoup(html, 'html.parser')
		
		file = open('pone_content4/'+doi.replace('/','_')+'.html','w')
		file.write(soup.prettify())
		file.close()
	except:
		print(doi)

dois_file = open('dois_pone_complete.txt','r').read()
dois = dois_file.split('\n')[:-1]
dois = set(dois)

pone_content = glob.glob('pone_content_all/*.html') + glob.glob('pone_content4/*.html')
invalid = set()

for doi in pone_content:
	invalid.add(doi.split('/')[1].replace("_","/")[:-5].lower())
	
valid = dois - invalid

print(len(dois),len(invalid))
print(len(valid))
# print(valid)
pool = ThreadPool(100) 
results = pool.map(get_content, valid)


		# title Abstract

		# mydivs = soup.findAll('div')
		# for div in mydivs: 
		#	 if (div["class"] == "article-text"):
		#		 print(div)

		# article_text = soup.find("div", {"class": "article-text"})