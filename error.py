import numpy as np
import math
import matplotlib.pyplot as plt
import glob

class Segmented:
    def __init__(self,breakpoints,slopes,min_x,max_x,min_y,max_y):
        self.breakpoints = [min_x] + breakpoints + [max_x]
        self.slopes = slopes
        
        self.y_breakpoints = [min_y]
        total_breakpoints = len(self.breakpoints)
        for i in range(1,total_breakpoints-1):
            y = slopes[i-1]*(self.breakpoints[i]-self.breakpoints[i-1]) + self.y_breakpoints[i-1]
            self.y_breakpoints.append(y)
        self.y_breakpoints.append(max_y)

    def get_y(self,x):
        total_breakpoints = len(self.breakpoints)
        for i in range(total_breakpoints-1):
            if x >= self.breakpoints[i] and x <= self.breakpoints[i+1]:
                y = self.slopes[i]*(x-self.breakpoints[i]) + self.y_breakpoints[i]
                return y

        # if x == self.breakpoints[-1]:
        #     return self.y_breakpoints[-1]
        return -1

    def error(self,X,Y):
        Y_aprox = [self.get_y(x) for x in X]
        error = [(y_ - y)**2 for y_,y in zip(Y_aprox,Y)]
        error = math.sqrt(sum(error))
        return error

    def plot(self,X,Y,save=False,name='test.pdf'):
        Y_aprox = [self.get_y(x) for x in X]
        plt.scatter(X,Y,c='green',alpha=0.4,s=20)
        plt.scatter(X,Y_aprox,c='red',alpha=0.4,s=20)
        plt.scatter(self.breakpoints[1:-1],self.y_breakpoints[1:-1],c='red',s=30)
        if save:
            plt.savefig(name,format='pdf',bbox_inches='tight')
            plt.clf()

def read_file(samples_breakpoints='data/samples_breakpoints.txt'):
    samples_breakpoints = open(samples_breakpoints,'r').read().split('\n')[:-1]
    total_series = len(samples_breakpoints)
    segmenteds = []
    data = []
    mses = []
    for i in range(0,total_series,3):
        mse = float(samples_breakpoints[i])
        # slopes = [float(n) for n in samples_breakpoints[i+1].split('\t')]
        # breakpoints = [float(n) for n in samples_breakpoints[i+2].split('\t')]
        # # X = np.asarray([float(n) for n in samples[i].split('\t')])
        # # X = ((X - min(X)) / (max(X) - min(X)))
        # # min_x = X[0]
        # # max_x = X[-1]
        # # Y = np.asarray([float(n) for n in samples[i+1].split(',')])
        # # Y = ((Y - min(Y)) / (max(Y) - min(Y)))
        # # min_y = Y[0]
        # # max_y = Y[-1]
        # segmenteds.append(Segmented(breakpoints,slopes,min_x,max_x,min_y,max_y))
        # data.append((X,Y))
        mses.append(mse)
    return segmenteds,data,mses

def get_errors(samples_breakpoints,save_plots=False):
    segmenteds,data,mses = read_file(samples='data/series.txt',samples_breakpoints=samples_breakpoints)
    i = 0
    errors = []
    for seg,(X,Y) in zip(segmenteds,data):
        if i > 1000:
            break
        i += 1
        if save_plots:
            try:
                seg.plot(X,Y,True,samples_breakpoints[:-4]+'/'+str(i)+'.pdf')
            except:
                plt.scatter(X,Y)
                plt.savefig(samples_breakpoints[:-4]+'/404/'+str(i)+'.pdf')
                plt.clf()
    print(np.mean(errors),np.std(errors))
    return errors

def subplots(errors,segmenteds,data):
    errors = np.array(errors)
    idxs = np.argsort(errors)
    i = 1
    for idx in idxs:
        plt.subplot(10,10,i)
        i += 1
        X,Y = data[idx]
        segmenteds[idx].plot(X,Y)
        if i >= 101:
            break

    plt.savefig('teste_subplots.pdf',format='pdf',bbox_inches='tight')

files = glob.glob('data/breakpoints_k*it.max*stop.if.error*.txt')
files = sorted(files)
for file in files:
    _,_,errors = read_file(samples_breakpoints=file)
    errors = [e for e in errors if e > 0]
    print(file)
    print("%.6f %.6f" % (np.mean(errors),np.std(errors)))


# STOPIFERROR é mais preciso porém dá erro mais vezes (MAIS LENTO TBM)
# DIFERENÇA PEQUENA 