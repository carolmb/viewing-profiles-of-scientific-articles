import numpy as np
import matplotlib.pyplot as plt

fd = open('data/series.txt', 'r')
dataStr = fd.readlines()
fd.close()

data = []
for lineIndex in range(0, len(dataStr), 2):
	timeStr = dataStr[lineIndex]
	viewsStr = dataStr[lineIndex+1]
	
	time = np.float_(timeStr.strip().split(','))
	views = np.int_(viewsStr.strip().split(','))
	
	data.append((time, views))
	
