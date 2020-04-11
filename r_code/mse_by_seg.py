import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

f_mse = 'mse_curves_error.txt'
f_seg = 'segmented_curves_error.txt'

mse = [np.sqrt(float(e)) for e in open(f_mse,'r').readlines()]
seg = open(f_seg,'r').readlines()
N = len(seg)
print(N)
seg = [len(seg[i+1].split(' ')) for i in range(0,N,4)]

print(len(mse))
print(len(seg))

seg_dist = defaultdict(lambda:[])
for s,e in zip(seg,mse):
    seg_dist[s].append(e)

fig,ax = plt.subplots(2,2,sharex=True)
for k,v in seg_dist.items():
    i = (k-2)//2
    j = (k-2)%2
    ax[i][j].hist(v)
    ax[i][j].set_title(k)

    for tick in ax[i][j].get_xticklabels():
        tick.set_rotation(45)
    if i == 1:
        ax[i][j].set_xlabel('RMSE')
    if j == 0:
        ax[i][j].set_ylabel('occurrences')

plt.tight_layout()
plt.savefig('mse_by_seg.pdf')
