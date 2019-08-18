import numpy as np
import matplotlib.pyplot as plt
import scipy
import pdb

def mergeSegments(x, y, xBreakInd, slopeDiffThreshold):
	'''Merge lines having a slope difference smaller than slopeDiffThreshold'''

	xBreakInd_new = xBreakInd[:]

	slopes = np.zeros(len(xBreakInd_new)-1)
	for index in range(len(xBreakInd_new)-1):
		xSub = x[xBreakInd_new[index]:xBreakInd_new[index+1]+1]
		ySub = y[xBreakInd_new[index]:xBreakInd_new[index+1]+1]
		
		a, b = scipy.polyfit(xSub, ySub, 1)
		
		slopes[index] = a
	
	xBreakInd_new = list(xBreakInd_new)
	slopes = list(slopes)
	
	#print("Slopes: {}".format(slopes))
	#pdb.set_trace()
	
	# keepGoing = True
	# while keepGoing:
		# keepGoing = False
		# for index in range(len(slopes)-1):
			# slopesDiff = np.abs(slopes[index+1] - slopes[index])/slopes[index]
			# #print(slopesDiff)
			# if slopesDiff<slopeDiffThreshold:
				# keepGoing = True
				
				# xBreakInd_new.pop(index+1)
				# slopes.pop(index+1)
				
				# xSub = x[xBreakInd_new[index]:xBreakInd_new[index+1]+1]
				# ySub = y[xBreakInd_new[index]:xBreakInd_new[index+1]+1]
				# a, b = scipy.polyfit(xSub, ySub, 1)
				# slopes[index] = a
							
				# break
				
	keepGoing = True
	while keepGoing:
		if len(slopes)<=1:
			break
		slopesDiff = np.abs(np.diff(slopes))/slopes[:-1]
		index = np.argmin(slopesDiff)
		minSlopeDiff = slopesDiff[index]
		if minSlopeDiff<slopeDiffThreshold:				
				xBreakInd_new.pop(index+1)
				slopes.pop(index+1)
				
				xSub = x[xBreakInd_new[index]:xBreakInd_new[index+1]+1]
				ySub = y[xBreakInd_new[index]:xBreakInd_new[index+1]+1]
				a, b = scipy.polyfit(xSub, ySub, 1)
				slopes[index] = a
		else:
			keepGoing = False
		
				
	return np.array(xBreakInd_new), np.array(slopes)
	
def artificialViews(x, slopes, breakpoints):

	numSegments = len(slopes)
	b = np.zeros(numSegments)
	for i in range(1, numSegments):
		b[i] = (slopes[i-1]-slopes[i])*breakpoints[i-1]+b[i-1]

	y = np.zeros(len(x))
	y[0:breakpoints[0]] = slopes[0]*x[0:breakpoints[0]]+b[0]
	for i in range(1,numSegments-1):
	  y[breakpoints[i-1]:breakpoints[i]] = slopes[i]*x[breakpoints[i-1]:breakpoints[i]]+b[i]

	y[breakpoints[numSegments-2]:len(y)] = slopes[numSegments-1]*x[breakpoints[numSegments-2]:len(x)]+b[numSegments-1]

	#yr = y + np.random.normal(0, 3., len(y))	
	
	return y
	
def readBreakpoints(fileName):

	fd = open(fileName, 'r')
	dataStr = fd.readlines()
	fd.close()

	breakpoints = []
	for lineIndex in range(0, len(dataStr), 3):
		seriesIndex = int(dataStr[lineIndex])
		slopesStr = dataStr[lineIndex+1]
		xBreakStr = dataStr[lineIndex+2]
		
		slopes = np.float_(slopesStr.strip().split(' '))
		xBreak = np.float_(xBreakStr.strip().split(' '))
		
		breakpoints.append((seriesIndex, xBreak))	
	
	return breakpoints
	

slopeDiffThreshold = 0.3

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
	

breakpoints = readBreakpoints('data/breakpoints.txt')
	
newBreakPoints = []	
for breakpIndex in range(len(breakpoints)):
	if breakpIndex%1000==0:
		print(breakpIndex)
	seriesIndex = breakpoints[breakpIndex][0]
	xBreak = breakpoints[breakpIndex][1]
	time = data[seriesIndex][0]
	views = data[seriesIndex][1]
	
	xBreakInd = np.zeros(len(xBreak)+2, dtype=np.int)
	for i,x in enumerate(xBreak):
		ind = np.argmin(np.abs(x-time))
		xBreakInd[i+1] = ind
	xBreakInd[-1] = len(views)-1

	mergedBreaks, slopes = mergeSegments(time, views, xBreakInd, slopeDiffThreshold)
	
	newBreakPoints.append((seriesIndex, slopes, time[mergedBreaks[1:-1]]))
	
strData = ''
fd = open('data/merged_breakpoints.txt', 'w')
for i in range(len(newBreakPoints)):	
	index, slopes, breakp = newBreakPoints[i]
	strData += '%d\n'%index
	strData += ' '.join([str(v) for v in slopes]) + '\n' 
	strData += ' '.join([str(v) for v in breakp]) + '\n'	
			
fd.write(strData)	
fd.close()
		
	
plt.plot(time, views, 'bo')	
plt.plot(time[xBreakInd], views[xBreakInd], 'ro')
plt.plot(time[mergedBreaks], views[mergedBreaks], 'go')	
plt.show()	


## Generate artificial series
#c = [1, 2, 4, 7, 11, 16]
#xx = np.arange(0, 101)
#xb = [15, 30, 45, 60, 75]
#y = artificialViews(xx, c, xb)

## Merge slopes differences smaller than slopeDiffThreshold
#breaks = mergeSegments(xx, y, [0]+list(xb)+[len(xx)-1], 6)



breakpToAnalyze = np.linspace(10, len(newBreakPoints)-10, 12, dtype=int)
for index,breakpIndex in enumerate(breakpToAnalyze):
	seriesIndex = breakpoints[breakpIndex][0]
	xBreakOr = breakpoints[breakpIndex][1]
	time = data[seriesIndex][0]
	views = data[seriesIndex][1]
	xBreakMerg = newBreakPoints[breakpIndex][2]
	xBreakInd = np.zeros(len(xBreakOr), dtype=np.int)
	for i,x in enumerate(xBreakOr):
		ind = np.argmin(np.abs(x-time))
		xBreakInd[i] = ind
	plt.subplot(3, 4, index+1)
	plt.plot(time, views, 'bo', ms=2)	
	ylim = plt.ylim()
	plt.vlines(time[xBreakInd], ylim[0], ylim[1], color='r')
	plt.vlines(xBreakMerg, ylim[0], ylim[1], color='g')



