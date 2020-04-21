import read_file
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from collections import defaultdict

data = read_file.load_data()
data = read_file.filter_outliers(data)
# freq1 = defaultdict(lambda:[])
freq = []
for sample in data:
	N = len(sample[1])
	delta_t = sample[3][-1] - sample[3][0]
	# freq1[N].append((delta_t,sample[-2][-1]))
	views = sample[-2][-1]
	if views == 0:
		continue
	freq.append((delta_t,views))

v = np.asarray(freq)
X = v[:,0]
Y = v[:,1]
p = pearsonr(X,Y)[0]
print(p)
# mean = np.mean(v)
# std = np.std(v)
plt.figure(figsize=(5,4))
df = pd.DataFrame({'lifetime':X,'views':Y})
g = sns.jointplot(x="lifetime", y="views", data=df, kind="kde");

g.ax_joint.set_xlabel('lifetime',fontsize=16)
g.ax_joint.set_ylabel('views',fontsize=16)
# g.ax_joint.set_xscale('log')
g.ax_joint.set_yscale('log')
# n = plt.hist(v)
# plt.text(9.5,0.78*max(n[0]), "$\mu$ = %.2f\n$\sigma$ = %.2f" % (mean,std), fontsize=16, bbox=dict(edgecolor='black', facecolor='none',alpha=0.5))
# plt.xlim(0,13)
# plt.xlabel('lifetime',fontsize=16)
# plt.ylabel('views',fontsize=16)
# plt.title(k)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
# plt.savefig('dist_delta_t_by_breaks_%d.pdf'%k)
plt.savefig('lifetime_views.pdf')
