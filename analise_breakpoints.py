import numpy as np
import matplotlib.pyplot as plt
import scipy
import pdb
import PCA

plt.ion()

def readBreakpoints(fileName):
	'''Read breakpoints file calculated in the R program'''

	fd = open(fileName, 'r')
	dataStr = fd.readlines()
	fd.close()

	breakpoints = []
	for lineIndex in range(0, len(dataStr), 3):
		seriesIndex = int(dataStr[lineIndex])
		slopesStr = dataStr[lineIndex+1]
		xBreakStr = dataStr[lineIndex+2]
		
		slopes = np.float_(slopesStr.strip().split(' '))
		
		breaks = xBreakStr.strip()
		if len(breaks)>0:
			xBreak = np.float_(breaks.split(' '))
		else:
			xBreak = []
		
		breakpoints.append((seriesIndex, slopes, xBreak))	
	
	return breakpoints
	
def calculateBreakProps(breakpoints, data):
	'''Calculate breakpoint properties'''
	
	numBreakp = np.zeros(len(breakpoints))
	avgInterv = np.zeros(len(breakpoints))
	avgPearson = np.zeros(len(breakpoints))
	seriesDuration = np.zeros(len(breakpoints))
	intervList = []
	pearsonList = []
	xBreakNormList = []

	for i in range(len(breakpoints)):

		seriesIndex = breakpoints[i][0]
		xBreak = breakpoints[i][2]
		time = data[seriesIndex][0]
		views = data[seriesIndex][1]
		xBreakInd = np.zeros(len(xBreak)+2, dtype=np.int)
		for j,x in enumerate(xBreak):
			ind = np.argmin(np.abs(x-time))
			xBreakInd[j+1] = ind
		xBreakInd[-1] = len(views)-1

		xBreakNorm = (xBreak - time[0]) / (time[-1] - time[0])
		#xBreakNormInd = [0] + list(np.round(xBreakNorm / dt).astype(int)) + [binsTime.size]
		#timeSpentOnSlope = np.diff(xBreakNormInd)
		
		numBreakp[i] = len(xBreak) #
		xBreakAugment = [time[0]] + list(xBreak) + [time[-1]]
		interv = np.diff(xBreakAugment)/(time[-1]-time[0])
		intervList.extend(interv) #
		avgInterv[i] = np.mean(interv) #
		pearson = []
		for j in range(len(xBreakInd)-1):
			t = time[xBreakInd[j]:xBreakInd[j+1]]
			v = views[xBreakInd[j]:xBreakInd[j+1]]
			if len(t)>1:
				pearson.append(np.corrcoef(t, v)[0][1])
		pearsonList.extend(pearson)	 #
		avgPearson[i] = np.mean(pearson) #

		seriesDuration[i] = time[-1]-time[0]
		xBreakNormList.append(xBreakNorm)

	return numBreakp, avgInterv, avgPearson, intervList, pearsonList, seriesDuration, xBreakNormList
	

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
	
inds = np.random.permutation(len(data))[:200]	
fig = plt.figure(figsize=[12, 8])	
for i in range(200):
	ax = plt.subplot(10, 20, i+1)
	time, views = data[inds[i]]
	plt.plot(time-time[0], np.gradient(views))
	#lt.plot([0, time[-1]-time[0]], [0, views[-1]], 'k', lw=1, c='0.75')
	ax.xaxis.set_visible(False)
	ax.yaxis.set_visible(False)
plt.tight_layout()	
	
breakpoints = readBreakpoints('data/merged_breakpoints.txt')

meas = calculateBreakProps(breakpoints, data)
numBreakp, avgInterv, avgPearson, intervList, pearsonList, seriesDuration, xBreakNormList = meas

	
bins = np.arange(np.min(numBreakp), np.max(numBreakp)+2)
plt.figure()
plt.subplot(2, 3, 1)
plt.hist(numBreakp, bins)
plt.xlabel('Number of breakpoints')

plt.subplot(2, 3, 2)
plt.hist(intervList, 20)
plt.xlabel('Transition interval')

plt.subplot(2, 3, 3)
plt.hist(avgInterv, 20)
plt.xlabel('Avg. transition interval')

plt.subplot(2, 3, 4)
plt.hist(pearsonList, 20)
plt.xlabel('Pearson corr.')

plt.subplot(2, 3, 5)
plt.hist(avgPearson, 20)
plt.xlabel('Avg. Pearosn corr.')

all_breakNorm = [v for breakNorm in xBreakNormList for v in breakNorm]
bins = np.linspace(0, 1, 50) #np.linspace(min(all_breakNorm), max(all_breakNorm), 50)
cumulativeHist = np.zeros(len(bins)-1)
for breakNorm in all_breakNorm:
	hist, _ = np.histogram(breakNorm, bins)
	cumulativeHist += hist


allSlopes = [slope for series in breakpoints for slope in series[1]]
allSlopes = np.log(allSlopes)
#allTimes = [t for series in data for t in series[0]]
#dt = 1/12.
binsSlopes = np.linspace(min(allSlopes), max(allSlopes), 30)
dSlope = binsSlopes[1] - binsSlopes[0]
minSlope = min(allSlopes)
#binsTime = np.arange(min(allTimes)-dt/2., max(allTimes)+dt/2+0.1*dt, dt)
binsTime = np.linspace(0, 1, 100)
dt = binsTime[1]-binsTime[0]
allSlopeArr = []
for i in range(len(breakpoints)):

	seriesIndex = breakpoints[i][0]
	slopes = np.log(breakpoints[i][1])
	xBreak = breakpoints[i][2]
	time = data[seriesIndex][0]
	views = data[seriesIndex][1]
	xBreakInd = np.zeros(len(xBreak)+2, dtype=np.int)
	for j,x in enumerate(xBreak):
		ind = np.argmin(np.abs(x-time))
		xBreakInd[j+1] = ind
	xBreakInd[-1] = len(views)-1	

	xBreakNorm = (xBreak-time[0])/(time[-1]-time[0])
	xBreakNormInd = [0] + list(np.round(xBreakNorm/dt).astype(int)) + [binsTime.size]
	timeSpentOnSlope = np.diff(xBreakNormInd)
	
	slopeArr = np.zeros(len(binsSlopes))
	for j in range(len(slopes)):
		indSlope = np.round((slopes[j]-minSlope)/dSlope).astype(int)
		slopeArr[indSlope] += timeSpentOnSlope[j]

	allSlopeArr.append(slopeArr)

allSlopeArr = np.array(allSlopeArr)
Y, D, W = PCA.PCA(allSlopeArr, 2, useCov=True)	
E = 100*D/sum(D)
plt.figure()
plt.plot(Y[:,0], Y[:,1], 'o', ms=1, alpha=0.3)
plt.xlabel(r'$C_1\;(%.1f\%%)$'%E[0])
plt.ylabel(r'$C_2\;(%.1f\%%)$'%E[0])


# Plot breakpoints
# i = len(breakpoints)-1
# seriesIndex = breakpoints[i][0]
# xBreak = breakpoints[i][2]
# time = data[seriesIndex][0]
# views = data[seriesIndex][1]
# plt.figure()
# plt.plot(time, views, 'b-')	
# ylim = plt.ylim()
# plt.vlines(xBreak, ylim[0], ylim[1])
# plt.show()	