import json,time
import glob
from bs4 import BeautifulSoup
from multiprocessing import Pool as ThreadPool
from tqdm.contrib.concurrent import process_map 


def format_paper_to_json(file):
	try:
		json_content = dict()
		with open(file,'r',encoding="utf8") as html_file:
			soup = BeautifulSoup(html_file, 'html.parser')

			# for script in soup(['sub','span']):
			# 	# print(script)
			# 	script.extract()

		
			title = soup.find("h1", {"id": "artTitle"})
			json_content['title'] = title.get_text().strip()

			abstract = soup.find("div", {"class": "abstract toc-section"})
			abstract = abstract.get_text(separator=' ').replace('\n',' ')
			abstract = ' '.join(abstract.split())
			json_content['abstract'] = abstract
			
		
			body = soup.find("div", {"id": "artText"})
			body = body.findChildren("div", {"class": "section toc-section"},recursive=False)
			
			json_content['body'] = dict()
			for child in body:
				section_title = child.find('h2').get_text().strip()
				section_text = ''
				for paragraph in child(['p']):
					text = paragraph.get_text(separator=' ').split()
					text = [word for word in text if len(word)>0]
					text = ' '.join(text)
					section_text += text + '\n'
				json_content['body'][section_title] = section_text
				# print(section_text)

		
		doi = file.split('\\')[1].replace('_','\\')[:-5]
		json_doi = {doi:json_content}
		output_file = 'pone_json_processed_sections\\'+file[len(header):-5] + '.json'
		with open(output_file, 'w') as out:
			json.dump(json_doi, out)
		del json_doi
		del json_content
		del abstract
		del title
		del body
	except Exception as inst:
		# print(type(inst))
		# print(inst.args)
		# print(inst)
		print(file)


header = 'pone_content_all/'

if __name__ == '__main__':    
	filenames = glob.glob(header+'*.html')
	# filenames = set(filenames)


	# pone_content = glob.glob('pone_json_processed_sections/*.json')
	# invalid = set()

	# k = len('pone_json_processed_sections/')

	# for doi in pone_content:
	# 	invalid.add(header+doi[k:-5]+'.html')

	# valid = filenames - invalid

	# print(len(filenames),len(invalid),len(valid))
	# filenames = filenames[:10]
	# filenames = ['pone_content_all/10.1371_journal.pone.0118637.html']
	# 
	pool = ThreadPool(30) 
	results = pool.map(format_paper_to_json, filenames)

	# for filename in filenames:
	# 	print(filename)
	# 	format_paper_to_json(filename)
	# 	