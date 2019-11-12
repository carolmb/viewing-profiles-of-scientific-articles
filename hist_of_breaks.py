import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
from collections import defaultdict

from read_file import load_data

data = load_data('data/plos_one_2019_breakpoints_k4_original1_data_filtered.txt')

incr = []
decr = []
for sample in data:
    slopes = sample[1]
    breakpoints = sample[2]
    n = len(breakpoints)
    delta_time = sample[3][-1] - sample[3][0]
    begin = sample[3][0]
    for i in range(n):
        moment = begin + delta_time*breakpoints[i]
        if slopes[i+1] > slopes[i]:
            incr.append(moment)
        else:
            decr.append(moment)

incr = np.asarray(incr)
decr = np.asarray(decr)

#-----------------------------------------------------------------------------

bins = 100

range0 = (min(min(incr),min(decr))-0.1,max(max(incr),max(decr)))
hist0,bins_edges0 = np.histogram(incr,bins=bins,range=range0)
hist1,bins_edges1 = np.histogram(decr,bins=bins,range=range0)


'''
def get_weights(incr,bins_edges0,hist0,hist1):
    w0 = []
    c = 0
    for v in incr:
        idx = np.searchsorted(bins_edges0,v)
        w0.append(1/(hist0[idx-1]+hist1[idx-1]))
        if idx-1 == 1:
            c += 1
            # print(v,idx-1,bins_edges0)
    print(c)
    return w0

w0 = get_weights(incr,bins_edges0,hist0,hist1)
w1 = get_weights(decr,bins_edges1,hist0,hist1)
'''
#-----------------------------------------------------------------------------

'''
plt.figure(figsize=(12,3))
count0, bins0, ignored0 = plt.hist([incr,decr],label=['incr','decr'],bins=bins,weights=[w0,w1],range=range0)

plt.legend()
plt.savefig('percent.png')
'''

#-----------------------------------------------------------------------------

# plt.figure(figsize=(12,3))
# plt.plot(list(range(len(count0[0]))),count0[0],'-',label='incr')
# plt.plot(list(range(len(count0[1]))),count0[1],'-',label='decr')
# plt.show()

#-----------------------------------------------------------------------------

#sns.set()
#sns.distplot(incr,label='incr',bins=bins,vertical=False,norm_hist=False)
#sns.distplot(decr,label='decr',bins=bins,vertical=False,norm_hist=False)

#sns.kdeplot(incr,bw=0.15,kernel='gau',label='incr2')
#sns.kdeplot(incr,bw=0.5,label='incr2')
#sns.kdeplot(decr,bw=0.15,kernel='gau',label='decr')
#plt.legend()
#plt.savefig('original.png')


#-----------------------------------------------------------------------------

def gaussian(x, a, mu, sig):
    return a*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def plt_conv_gauss(hist0,bins_edges0,label):
    deltas = []
    N = len(hist0)
    for i in range(N):
        deltas.append(((bins_edges0[i]+bins_edges0[i+1])/2,hist0[i]))

    X = np.arange(min(bins_edges0),max(bins_edges0)+1,1/12)
    Y = np.zeros(len(X))
    
    x_gaussian = np.arange(-10,10,0.2)
    N = len(deltas)
    for i in range(N):
        y_gauss = gaussian(x_gaussian,a=deltas[i][1],mu=0,sig=0.2)
        for x,y in zip(x_gaussian,y_gauss):
            x_g = x+deltas[i][0]
            idx = np.searchsorted(X,x_g)
            if idx >= 0 and idx < len(Y):
                Y[idx] += y

    return np.asarray(X),np.asarray(Y)


plt.figure(figsize=(12,3))
X0,Y0 = plt_conv_gauss(hist0,bins_edges0,'incr')
X1,Y1 = plt_conv_gauss(hist1,bins_edges1,'decr')

plt.bar(X0-0.05,Y0,label='incr',width=0.05)
plt.bar(X1,Y1,label='decr',width=0.05)
plt.legend()
plt.tight_layout()
plt.savefig('my_convolve_gaussian_a_0_0.2.png')


# plt.hist(decr,bins=bins,range=range0,color='red')


plt.figure(figsize=(12,3))
x_gaussian = np.arange(-10,10,0.2)
gauss = gaussian(x_gaussian,1,0,0.2)
y0 = np.convolve(hist0,gauss,mode='same')
y1 = np.convolve(hist1,gauss,mode='same')
x = np.arange(bins_edges0[0],bins_edges1[-1],(bins_edges1[-1]-bins_edges1[0])/len(y0))
plt.bar(x-0.05,y0,width=0.05,label='incr')
plt.bar(x,y1,width=0.05,label='decr')
plt.legend()

plt.tight_layout()
plt.savefig('convolve_gaussian_1_0_1to12.png')

