import numpy as np
import matplotlib.pyplot as plt

filename = 'mse_curves.txt'

file = open(filename,'r').readlines()

mse_values = [float(line) for line in file]

for i in mse_values[:10]:
	print(i)

plt.figure(figsize=(10,5))

bins = np.logspace(np.log10(0.0000001),np.log10(0.04), 10)
plt.hist(mse_values,bins=bins,density=True)

plt.yscale('log')
plt.xscale('log')
plt.vlines(0.0001,ymin=0,ymax=100000,colors='red')
plt.xlabel('MSE')
plt.ylabel('occurrences')
plt.savefig('mse_original_segmented.pdf')