import read_file
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

data = read_file.load_data()
freq = defaultdict(lambda:[])
for sample in data:
	N = len(sample[1])
	delta_t = sample[3][-1] - sample[3][0]
	freq[N].append(delta_t)

for k,v in freq.items():
	mean = np.mean(v)
	std = np.std(v)
	plt.figure(figsize=(5,3))
	n = plt.hist(v)
	plt.text(9.5,0.78*max(n[0]), "$\mu$ = %.2f\n$\sigma$ = %.2f" % (mean,std), fontsize=16, bbox=dict(edgecolor='black', facecolor='none',alpha=0.5))
	plt.xlim(0,13)
	plt.xlabel('lifetime',fontsize=16)
	# plt.title(k)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.tight_layout()
	plt.savefig('dist_delta_t_by_breaks_%d.pdf'%k)
