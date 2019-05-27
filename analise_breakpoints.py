# library(plot3D)

# fileName <- "data/breakpoints_k4it.max100stop.if.errorFALSE.txt"
# conn <- file(fileName,open="r")
# linn <-readLines(conn)

# slopes <- c()
# breakpoints <- c()

# for (i in seq(1,length(linn),by=3)){
#     error_i <- as.numeric(linn[i])
#     if(error_i == 0){
#         next
#     }
#     slopes_i <- as.numeric(strsplit(linn[i+1],' ')[[1]])
#     breakpoints_i <- as.numeric(strsplit(linn[i+2],' ')[[1]])
#     slopes <- c(slopes, slopes_i)
#     breakpoints <- c(breakpoints, list(breakpoints_i))
# }

# close(conn)

# breakpoints2intervals <- function(x) {
#     interval_xs <- c()
#     interval_xs <- c(interval_xs,x[[1]][1])
    
#     for (i in seq(1,length(x[[1]])-1)){
#         d <- x[[1]][i+1]-x[[1]][i]
#         interval_xs <- c(interval_xs,d)
#     }
#     d <- 1-x[[1]][length(x[[1]])]
#     interval_xs <- c(interval_xs,d)

#     return(interval_xs)
# }

# intervals <- c()
# for(i in seq(1,length(breakpoints))){
#     intervals <- c(intervals,breakpoints2intervals(breakpoints[i]))
# }

# print(length(slopes[1:100]))
# print(length(intervals[1:100]))

# z <- table(slopes[1:1000],intervals[1:1000])

# hist3D(z=z, border="black")

# # heatmap:
# image2D(z=z, border="black")

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from collections import defaultdict

def read_file(samples_breakpoints='data/breakpoints_k4it.max100stop.if.errorFALSE.txt'):
    samples_breakpoints = open(samples_breakpoints,'r').read().split('\n')[:-1]
    total_series = len(samples_breakpoints)
    
    slopes = []
    breakpoints = []
    for i in range(0,total_series,3):
        error = float(samples_breakpoints[i])
        if error == 0:
            continue
        slopes_i = [float(n) for n in samples_breakpoints[i+1].split(' ')]
        breakpoints_i = [float(n) for n in samples_breakpoints[i+2].split(' ')]

        slopes.append(slopes_i)
        breakpoints.append(breakpoints_i)
    
    return slopes,breakpoints

def breakpoints2intervals(x):
    intervals = [x[0]]
    for i in range(len(x)-1):
        intervals.append(x[i+1]-x[i])
    intervals.append(1-x[-1])
    return intervals

def flatten(X):
    y = []
    for x in X:
        y += x
    return y

def plot_hist(x,y,labelx,labely,nx,ny):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    min_x = min(x)
    max_x = max(x)
    min_y = min(y)
    max_y = max(y)

    hist, xedges, yedges = np.histogram2d(x, y, bins=[nx,ny], range=[[min_x, max_x], [min_y, max_y]])

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = (max_x-min_x)/nx
    dy = (max_y-min_y)/ny
    dz = hist.ravel()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average',edgecolors='white')

    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.show()

    plt.imshow(hist, cmap='hot', interpolation='nearest')
    plt.xticks(range(0,nx),[str(f)[:4] for f in xedges])
    plt.yticks(range(0,ny),[str(f)[:4] for f in yedges])
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.show()

def calculate_cond_xi_xi1(X,intervalsx,var):
    prob_xi = defaultdict(lambda:0)
    prob_xi1 = defaultdict(lambda:defaultdict(lambda:0))
    total = 0
    for x in X:
        for i in range(len(x)):
            total += 1
            for j in range(len(intervalsx)-1):
                if x[i] >= intervalsx[j] and x[i] < intervalsx[j+1]:
                    prob_xi[j] += 1
                    if i+1 < len(x):
                        for k in range(len(intervalsx)-1):
                            if x[i+1] >= intervalsx[k] and x[i+1] < intervalsx[k+1]:
                                prob_xi1[j][k] += 1
                
    for i,p in prob_xi.items():
        if p > 0:
            print("p(%.4f<=%si<%.4f)=%.4f" % (intervalsx[i],var,intervalsx[i+1],p/total))

    for i,prob in prob_xi1.items():
        total = prob_xi[i]
        for j,p in prob.items():
            if p > 0:
                print("p(%.4f<=%si<%.4f|%.4f<=Xi+1<%.4f)=%.4f" % (intervalsx[i],var,intervalsx[i+1],intervalsx[j],intervalsx[j+1],p/total))
    print()

def probabilities(X,Y,nx,ny):
    minx,maxx = min(flatten(X)),max(flatten(X))
    miny,maxy = min(flatten(Y)),max(flatten(Y))

    deltax = (maxx-minx)/nx
    deltay = (maxy-miny)/ny

    intervalsx = np.arange(minx,maxx+deltax,deltax)
    intervalsy = np.arange(miny,maxy+deltay,deltay)

    calculate_cond_xi_xi1(X,intervalsx,'X')
    calculate_cond_xi_xi1(Y,intervalsy,'Y')

slopes,breakpoints = read_file()
intervals = [breakpoints2intervals(b) for b in breakpoints]
slopes = [(np.arctan(s)*57.2958).tolist() for s in slopes]

nx = 5
ny = 5
probabilities(slopes,intervals,nx,ny)

# slopes = flatten(slopes)
# intervals = flatten(intervals)
# plot_hist(slopes,intervals,'slopes','intervals',nx=nx,ny=nx)