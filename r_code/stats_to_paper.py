import numpy as np
import matplotlib.pyplot as plt

filename = 'mse_curves.txt'

file = open(filename,'r').readlines()

mse_values = [np.sqrt(float(line)) for line in file]

for i in mse_values[:10]:
	print(i)

plt.figure(figsize=(10,5))

# bins = np.logspace(np.log10(min(mse_values)),np.log10(max(mse_values)))
plt.hist(mse_values,bins=50,density=True,range=(0.001,0.02)) #bins=bins

#plt.hist(mse_values)
# plt.yscale('log')
# plt.xscale('log')
plt.vlines(0.01,ymin=0,ymax=200,colors='red')
plt.xlabel('RMSE',fontsize=14)
plt.ylabel('occurrences',fontsize=14)
plt.savefig('rmse_original_segmented_linear_linear.pdf')
