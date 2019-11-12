# import WOSGetter

# wosArticles = WOSGetter.GZJSONLoad("wosPlosOne2016_citations.json.gz");
# print(len(wosArticles))

# completeTimeSeries = WOSGetter.GZJSONLoad("plosone2016_hits.json.gz")
# print(len(completeTimeSeries))

# h = len(wosArticles)//2

# ns = [0,-1]
# for i in ns:
# 	print(i)
# 	print(wosArticles[i]['DI'])
# 	print(completeTimeSeries[i]['total'][0])

# # i = 8001

# for i in range(7999,8003):
# 	print(i)
# 	print(wosArticles[i]['DI'],wosArticles[i]['AU'])
# 	try:
# 		print(completeTimeSeries[i]['total'][0])
# 	except:
# 		print(completeTimeSeries[i])
# 	print()

# for i in range(11000,11006):
# 	print(i)
# 	print(wosArticles[i]['DI'])
# 	try:
# 		print(completeTimeSeries[i]['total'][0])
# 	except:
# 		print(completeTimeSeries[i])
# 	print()

# for i in range(28898,29003):
# 	print(i)
# 	print(wosArticles[i]['DI'])
# 	try:
# 		print(completeTimeSeries[i]['total'][0])
# 	except:
# 		print(completeTimeSeries[i])
# 	print()

# # for i,article in enumerate(completeTimeSeries[:-1]):
# # 	if article == completeTimeSeries[i+1]:
# # 		print(i+1)


import matplotlib.pyplot as plt
import numpy as np
import json
import glob

files = glob.glob('subj_*.json')
json_files = []
for f in files:
	json_files.append(json.loads(open(f,'r').read()))

subs = []
for j in json_files:
	for doi,s in j.items():
		for w in s:
			if not w.strip():
				continue
			subs.append(w)

unique,count = np.unique(subs,return_counts=True)
idx = np.argsort(-count)
unique = unique[idx]
count = count[idx]

print(unique[:10])

plt.figure(figsize=(6,12))
plt.barh(unique[:100],count[:100])
# plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('out.png')

