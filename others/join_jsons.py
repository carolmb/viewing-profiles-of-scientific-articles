import glob
import json
import WOSGetter

def merge_jsons(filenames):

	merged_json = {}

	for filename in filenames:
		content = open(filename,'r').read()
		content_json = json.loads(content)

		for key,item in content_json.items():
			c = 0
			for key1,item1 in item.items():
				merged_json[key] = item1
				c += 1
			if c > 1:
				print('ERROR: see key',key)

	return merged_json

def preprocess_time_serie(views):
	total = 0
	cumulative_views = []
	dates = []
	for date in views:
		dates.append(str(int(date['year'])+int(date['month'])/12))
		total += int(date['pdf_views']) + int(date['xml_views']) + int(date['html_views'])
		cumulative_views.append(str(total))

	return {'months':dates,'views':cumulative_views}


def filter(merged_json):
	wosArticles = WOSGetter.GZJSONLoad("wosPlosOne2016_citations.json.gz")

	DOIs = [article["DI"] for article in wosArticles];

	filtered_jsons = {}

	c = 0
	for i,doi in enumerate(DOIs):
		if doi != '':
			try:
				views = merged_json[doi]['views']
				filtered_jsons[doi] = {'time_series':preprocess_time_serie(views), 'infos':wosArticles[i]}
			
				c += 1
			except:
				pass
		else:
			print('Article without DOI in WOS',i)

	print('matchs',c*100/len(DOIs))
	return filtered_jsons

def load(filename):
	content = open(filename,'r').read()
	return json.loads(content)

def save(data,filename):
	with open(filename,'w') as f:
		json.dump(data,f)

def format2work(merged_jsons,filename):
	output = open(filename,'w')

	for doi,item in merged_jsons.items():
		# print(item)
		output.write(doi+'\n')
		output.write(','.join(item['time_series']['months'])+'\n')
		output.write(','.join(item['time_series']['views'])+'\n')

	output.close()

# filenames = glob.glob('papers_time_series/*2.json')
# merged_jsons = merge_jsons(filenames)
# save(merged_jsons,'papers_plos_data_time_series2_no_filter.json')

# merged_jsons = load('papers_plos_data_time_series2_no_filter.json')
# merged_jsons = filter(merged_jsons)
# save(merged_jsons,'papers_plos_data_time_series2_filtered.json')

# merged_jsons = load('papers_plos_data_time_series2_filtered.json')
# format2work(merged_jsons,'plos_one_2019.txt')