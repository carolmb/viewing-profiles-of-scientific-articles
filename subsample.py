import numpy as np 


file = 'data/series.txt'
file = open(file,'r').read().split('\n')[:-1]
X = []
Y = []
for i,v in enumerate(file):
    if i%2 == 0:
        X.append(v)
    else:
        Y.append(v)

X = np.asarray(X)
Y = np.asarray(Y)

arr = np.arange(len(X),dtype='int')
np.random.shuffle(arr)

n = 10000
arr = arr[:n]

X = X[arr]
Y = Y[arr]

file = 'data/samples.txt'
file = open(file,'w')
for x,y in zip(X,Y):
    file.write(x+'\n'+y+'\n')
file.close()