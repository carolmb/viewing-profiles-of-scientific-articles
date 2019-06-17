import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import pearsonr

def read_file_original(filename='data/samples.txt'):
    samples_breakpoints = open(filename,'r').read().split('\n')[:-1]
    total_series = len(samples_breakpoints)
    print(total_series)
    X = []
    Y = []
    for i in range(0,total_series,2):
        xs = [float(n) for n in samples_breakpoints[i].split(',')]
        ys = [float(n) for n in samples_breakpoints[i+1].split(',')]
        # breakpoints_i.append(1.0)
        X.append(np.asarray(xs))
        Y.append(np.asarray(ys))
    
    return np.asarray(X),np.asarray(Y)


def read_file(samples_breakpoints='data/breakpoints_k4it.max100stop.if.errorFALSE.txt',n=4):
    samples_breakpoints = open(samples_breakpoints,'r').read().split('\n')[:-1]
    total_series = len(samples_breakpoints)
    print(total_series)
    slopes = []
    breakpoints = []
    idxs = []
    for i in range(0,total_series,3):
        idx = int(samples_breakpoints[i]) - 1
        
        slopes_i = [float(n) for n in samples_breakpoints[i+1].split(' ')]
        breakpoints_i = [float(n) for n in samples_breakpoints[i+2].split(' ')]
        # breakpoints_i.append(1.0)

        if len(slopes_i) == n:
            idxs.append(idx)
            slopes.append(np.asarray(slopes_i))
            breakpoints.append(np.asarray(breakpoints_i))
    
    return np.asarray(idxs),np.asarray(slopes),np.asarray(breakpoints)

def breakpoints2intervals(x):
    intervals = [x[0]]
    for i in range(len(x)-1):
        intervals.append(x[i+1]-x[i])
    intervals.append(1-x[-1])
    return intervals

def sub_plot_hist(hist,xedges,yedges,labelx,labely,nx,ny,fig_hist,fig_heat,idx,deltax,deltay):
    if fig_hist:
        ax = fig_hist.add_subplot(5,1,idx, projection='3d')

        # Construct arrays for the anchor positions of the 16 bars.
        xpos, ypos = np.meshgrid(yedges + 0.25, xedges + 0.25, indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = 0

        # Construct arrays with the dimensions for the 16 bars.
        dx = deltay
        dy = deltax
        dz = hist.ravel()

        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average',edgecolors='white')

        ax.set_xlabel(labely)
        ax.set_ylabel(labelx)

    if fig_heat:    
        ax = fig_heat.add_subplot(5,1,idx)
        ax.imshow(hist, cmap='hot', interpolation='nearest',extent=(0,nx+1,nx+1,0))
        ax.set_xticks(range(nx+1))
        ax.tick_params(axis='both',labelsize=6)
        ax.set_xticklabels([str(f)[:4] for f in xedges],rotation=60)
        ax.set_yticks(range(nx+1))
        ax.set_yticklabels([str(f)[:4] for f in yedges])
        ax.set_xlabel(labelx)
        ax.set_ylabel(labely)
    
def plot_xi_xi1(X,idx,fig,n):
    pointx = []
    pointy = []
    for x in X:
        for i in range(len(x)-1):
            pointx.append(x[i])
            pointy.append(x[i+1])
    pearson,_ = pearsonr(pointx,pointy)
    ax = fig.add_subplot(n,1,idx)
    ax.scatter(pointx,pointy,alpha=0.3)
    ax.title.set_text(str(idx) + ' (pearson=' + str(pearson)[:9]+')')

def calculate_cond_xi_xi1(X,intervalsx):
    prob_xi = defaultdict(lambda:0)
    prob_xi1 = defaultdict(lambda:defaultdict(lambda:0))
    
    for x in X:
        for j in range(len(intervalsx)-1):
            if x[0] >= intervalsx[j] and x[0] < intervalsx[j+1]:
                prob_xi[j] += 1
                for k in range(len(intervalsx)-1):
                    if x[1] > intervalsx[k] and x[1] <= intervalsx[k+1]:
                        prob_xi1[j][k] += 1
    for i,prob in prob_xi1.items():
        total = prob_xi[i]
        for j in prob.keys():
            prob[j] = prob[j]/total
    return prob_xi1

def plot_hist(X,intervalsx,nx,fig_hist,fig_heat,idx,deltax):
    
    prob_xi_xi1 = calculate_cond_xi_xi1(X,intervalsx)
    
    hist = np.zeros((len(intervalsx),len(intervalsx)))
    for j,ps in prob_xi_xi1.items():
        for k,p in ps.items():
            hist[j][k] = p
    
    sub_plot_hist(hist,intervalsx,intervalsx,'x+1','x',nx,nx,fig_hist,fig_heat,idx,deltax,deltax)
    return prob_xi_xi1,fig_hist,fig_heat

def get_i(V,intervals):
    idxs = []
    for v in V:
        for k in range(len(intervals)-1):
            if v > intervals[k] and v <= intervals[k+1]:
                idxs.append(k)
                break
    idxs = np.asarray(idxs)
    return idxs

def plot_intervals_slopes(n=4):
    # PARA HISTOGRAMA DO SLOPE POR INTERVAL
    fig_hist = plt.figure(figsize=(5,20))
    fig_heat = plt.figure(figsize=(5,20))
    for i in range(n):
        slopes_i = slopes[:,i]
        intervals_i = intervals[:,i]
        slopes_i = get_i(slopes_i,intervalsx)
        intervals_i = get_i(intervals_i,intervalsy)
        
        hist = np.zeros((len(intervalsx),len(intervalsy)))
        for x,y in zip(slopes_i,intervals_i):
            hist[x][y] += 1
        print(hist)
        sub_plot_hist(hist,intervalsy,intervalsx,'intervals'+str(i),'slopes'+str(i),nx,ny,fig_hist,fig_heat,i+1,deltay,deltax)
    fig_hist.tight_layout()
    fig_hist.savefig('slope_interval_hist.pdf',format='pdf')
    fig_heat.tight_layout()
    fig_heat.savefig('slope_interval_heat.pdf',format='pdf')

def plots(slopes,intervals,n,intervalsx,intervalsy):
    fig_slopes = plt.figure(figsize=(n,3*n))
    fig_intervals = plt.figure(figsize=(n,3*n))
    fig_slopes_intervals = plt.figure(figsize=(n,3*n))
    fig_intervals_slopes = plt.figure(figsize=(n,3*n))
    fig_hist_slopes = plt.figure(figsize=(n,3*n))
    fig_heat_slopes = plt.figure(figsize=(n,3*n))
    fig_hist_intervals = plt.figure(figsize=(n,3*n))
    fig_heat_intervals = plt.figure(figsize=(n,3*n))

    for i in range(n-1):
        slopes_i = slopes[:,i:i+2]
        print(slopes_i.shape)
        plot_xi_xi1(slopes_i,i+1,fig_slopes,n)
        
        intervals_i = intervals[:,i:i+2]
        plot_xi_xi1(intervals_i,i+1,fig_intervals,n)
        
        slopes_i_intervals_i1 = np.concatenate((slopes[:,i:i+1],intervals[:,i+1:i+2]),axis=1)
        plot_xi_xi1(slopes_i_intervals_i1,i+1,fig_slopes_intervals,n)
        
        intervals_i_slopes_i1 = np.concatenate((intervals[:,i:i+1],slopes[:,i+1:i+2]),axis=1)
        plot_xi_xi1(intervals_i_slopes_i1,i+1,fig_intervals_slopes,n)
        
        plot_hist(slopes_i,intervalsx,nx,fig_hist_slopes,fig_heat_slopes,i+1,deltax)
        plot_hist(intervals_i,intervalsy,nx,fig_hist_intervals,fig_heat_intervals,i+1,deltay)
        

    fig_slopes.suptitle('slopes')
    fig_slopes.savefig("imgs_python/slopes_"+str(n)+".pdf",format='pdf',bbox_inches='tight')

    fig_slopes.suptitle('intervals')
    fig_intervals.savefig("imgs_python/intervals_"+str(n)+".pdf",format='pdf',bbox_inches='tight')

    fig_slopes_intervals.suptitle('x(i) is slope, y(i+1) is interval')
    fig_slopes_intervals.savefig("imgs_python/slopes_intervals_"+str(n)+".pdf",format='pdf',bbox_inches='tight')

    fig_intervals_slopes.suptitle('x(i) is interval, y(i+1) is slope')
    fig_intervals_slopes.savefig("imgs_python/intervals_slopes_"+str(n)+".pdf",format='pdf',bbox_inches='tight')

    fig_hist_slopes.tight_layout()
    fig_hist_slopes.suptitle('P(x+1|x), x is slope')
    fig_hist_slopes.savefig("imgs_python/hist_slopes_"+str(n)+".pdf",format='pdf',bbox_inches='tight')

    fig_heat_slopes.tight_layout()
    fig_heat_slopes.suptitle('P(x+1|x), x is slope', y=1.08)
    fig_heat_slopes.savefig("imgs_python/heat_slopes_"+str(n)+".pdf",format='pdf',bbox_inches='tight')

    fig_hist_intervals.tight_layout()
    fig_hist_intervals.suptitle('P(x+1|x), x is intervals')
    fig_hist_intervals.savefig("imgs_python/hist_intervals_"+str(n)+".pdf",format='pdf',bbox_inches='tight')

    fig_heat_intervals.tight_layout()
    fig_heat_intervals.suptitle('P(x+1|x), x is intervals', y=1.08)
    fig_heat_intervals.savefig("imgs_python/heat_intervals_"+str(n)+".pdf",format='pdf',bbox_inches='tight')


def get_prob(X,intervalsx):
    x_by_i = defaultdict(lambda:[])
    idxs = get_i(X,intervalsx)
    for i,x in zip(idxs,X):
        x_by_i[i].append(x)
    unique,count = np.unique(idxs,return_counts=True)
    prob = defaultdict(lambda:0)
    total = len(idxs)
    for u,c in zip(unique,count):
        prob[u] = c/total
    return x_by_i,prob

def get_slope_val(nx,prob_slope1,x_by_i):
    prob = np.zeros(nx)
    for k,p in prob_slope1.items():
        prob[k] = p
    slopei = np.random.choice(np.arange(nx), p=prob)
    to_select = x_by_i[slopei]
    slopeiv = np.random.randint(0,len(to_select))
    slopev = to_select[slopeiv]
    return slopev,slopei

def generate_artificial_series(n,prob_slope1,prob_cond,x_by_i):
    series = []
    for _ in range(n):
        slope_artificial = np.zeros(4)
        slope_artificial[0],slope0_i = get_slope_val(nx,prob_slope1,x_by_i[0])

        prob = prob_cond[0][slope0_i]
        slope_artificial[1],slope1_i = get_slope_val(nx,prob,x_by_i[1])
        
        prob = prob_cond[1][slope1_i]
        slope_artificial[2],slope2_i = get_slope_val(nx,prob,x_by_i[2])
        
        prob = prob_cond[2][slope2_i]
        slope_artificial[3],_ = get_slope_val(nx,prob,x_by_i[3])
        
        series.append(slope_artificial)
    return series

def save(series_slopes,series_intervals,filename):
    f = open(filename,'w')
    for s,i in zip(series_slopes,series_intervals):
        f.write('-1\n')
        to_str = ''
        for v in s:
            to_str += str(v)+' '
        f.write(to_str[:-1]+"\n")
        to_str = ''
        for v in i:
            to_str += str(v)+' '
        f.write(to_str[:-1]+"\n")
    f.close()

def artificial_series(slopes,intervalsx):
    x_by_i = [0,0,0,0]
    x_by_i[0],prob_slope1 = get_prob(slopes[:,0],intervalsx)
    x_by_i[1],_ = get_prob(slopes[:,1],intervalsx)
    x_by_i[2],_ = get_prob(slopes[:,2],intervalsx)
    x_by_i[3],_ = get_prob(slopes[:,3],intervalsx)
    
    prob_cond = []
    prob_cond.append(calculate_cond_xi_xi1(slopes[:,0:2],intervalsx))
    prob_cond.append(calculate_cond_xi_xi1(slopes[:,1:3],intervalsx))
    prob_cond.append(calculate_cond_xi_xi1(slopes[:,2:4],intervalsx))

    series_slopes = generate_artificial_series(1000,prob_slope1,prob_cond,x_by_i)
    
    return series_slopes

if __name__ == "__main__":
    
    xs,ys = read_file_original()
    all_intervals = []
    all_n = []
    for n in [2,3,4,5]:
        
        idxs,slopes,breakpoints = read_file(n=n)
        xs_n = [xs[idx] for idx in idxs]
        print('n=',n,len(idxs))
        all_n += [n]*len(idxs)
        all_intervals += [xs[-1]-xs[0] for xs in xs_n]
        # print(np.unique(delta_xs_n,return_counts=True))
    print(pearsonr(all_n,all_intervals))

    nx = 10
    ny = 10

    minx,maxx = 0,90
    deltax = (maxx-minx)/nx
    intervalsx = np.arange(minx,maxx+deltax,deltax)

    miny,maxy = 0,1
    deltay = (maxy-miny)/ny
    intervalsy = np.arange(miny,maxy+deltay,deltay)

    '''
    for n in [2,3,4,5]:
        _,slopes,breakpoints = read_file(n=n)
        intervals = np.asarray([np.asarray(breakpoints2intervals(b)) for b in breakpoints])
        slopes = np.asarray([(np.arctan(s)*57.2958) for s in slopes])
        
        plots(slopes,intervals,n,intervalsx,intervalsy)
    '''
    '''
    ls = []
    for x in slopes:
        ls.append(len(x))
    unique,count = np.unique(ls,return_counts=True)
    for u,c in zip(unique,count):
        print(u,c)
    '''

    '''
    # plots()
    # mean_interval = np.mean(intervals.flatten())
    # artificial_series(slopes,intervalsx,mean_interval)
    
    # INTERVALO SEGUINDO PROB COND/ANGULO MEDIO DE CADA INTERVALO
    mean_slopes = np.mean(slopes,axis=0)
    articifial_xs = artificial_series(intervals,intervalsy)
    # mean_slopes = [[mean_slopes]*4]*1000
    mean_slopes = [mean_slopes.tolist()]*1000
    articifial_xs = np.asarray(articifial_xs)
    save(mean_slopes,articifial_xs,'data/artificial_intervals_slope_axis0.txt')
    
    # INTERVALO SEGUINDO PROB COND/ANGULO ALEATÓRIO DE CADA INTERVALO
    artificial_slopes = np.random.choice(slopes.flatten(),size=4000).reshape(1000,4)
    save(artificial_slopes,articifial_xs,'data/artificial_intervals_slope_random.txt')

    # TUDO ALEATORIO (qualquer eixo)
    artificial_slopes = (np.random.rand(1000,4)*maxx)
    # artificial_slopes = np.random.choice(slopes.flatten(),size=4000).reshape(1000,4)
    
    artificial_intervals = np.random.rand(1000,4)
    # artificial_intervals = np.random.choice(intervals.flatten(),size=4000).reshape(1000,4)
    save(artificial_slopes,artificial_intervals,'data/artificial_all_random.txt')
    '''