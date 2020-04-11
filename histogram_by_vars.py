import matplotlib.pyplot as plt
from read_file import select_original_breakpoints
from collections import defaultdict

N = 5
slopes,intervals = select_original_breakpoints(N)

slopes_hist = defaultdict(lambda:[])
intervals_hist = defaultdict(lambda:[])

for sample in slopes:
    for i in range(N):
        slopes_hist[i].append(sample[i])

for sample in intervals:
    for i in range(N):
        intervals_hist[i].append(sample[i])

fig,axs = plt.subplots(5,2,figsize=(6.5,10),sharex='col')
for i in range(N):
    axs[i][0].hist(slopes_hist[i])
    axs[i][0].set_xlabel('$\\alpha_%d$'%(i+1))
    axs[i][0].set_ylabel('occurrences')
for i in range(N):
    axs[i][1].hist(intervals_hist[i])
    axs[i][1].set_xlabel('$l_%d$'%(i+1))

plt.tight_layout()
plt.savefig('histogram_by_var.pdf')

