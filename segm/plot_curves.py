import pandas
import matplotlib.pyplot as plt

plt.figure(figsize=(7,5))
idxs = [16,127,144,169,174,183,188]
colors = ['tab:blue','tab:red','tab:orange','tab:purple','tab:green','tab:brown','tab:pink','tab:grey']
j = 0
for i in idxs:
	df = pandas.read_csv('data %d .csv' % i)
	plt.plot(df['x'].tolist(),df['y'].tolist(),c=colors[j])
	j += 1

plt.xlabel('time',fontsize=14)
plt.ylabel('views',fontsize=14)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('examples.pdf')