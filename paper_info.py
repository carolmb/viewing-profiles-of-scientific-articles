import glob
import shutil
import json 

files = glob.glob('pone_json_processed_sections/*.json')

files = files[-100000:]
print(len(files))

print()

# i = 0
titles = set()
for file in files:
	with open(file,'r') as data:
		paper = json.load(data)
		key = list(paper.keys())[0]
		for title,content in paper[key]['body'].items():
			titles.add(title)
			if title == 'Development of PARMA3.0':
				print(key)
				for title,content in paper[key]['body'].items():
					print(title)


	# i+=1

	
# print(titles)

# dois = set()
# for file in files:
# 	d = file.split('/')[1].replace('_','/')[:-5]
# 	# print(d)
# 	dois.add(d)


# output = open('dois_pone_processed_sections.txt','w')
# for doi in dois:
# 	output.write(doi+'\n')
# output.close()



# file = 'data/plos_one_2019.txt'

# content = open(file,'r').read().split('\n')[:-1]

# N = len(content)
# dois = []
# for i in range(0,N,3):
# 	dois.append(content[i])

# pone_dois_file = 'dois_pone_complete.txt'
# out = open(pone_dois_file,'w')
# for doi in dois:
# 	out.write(doi+'\n')
# out.close()


# files1 = glob.glob('pone_content_all/*.html')

# files1 = set(files1)

# test = set()
# for file in files1:
# 	bla = file.split('/')[1].replace('_','/')[:-5]
# 	test.add(bla)

# dois_file = open('dois_pone.txt','r').read()
# dois = dois_file.split('\n')[:-1]
# dois = set(dois)

# print(dois-test)


# files2 = glob.glob('pone_content4/*.html')
# for file in files2:
# 	shutil.copy(file,'pone_content_all/')
