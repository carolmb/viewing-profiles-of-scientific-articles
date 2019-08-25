import numpy as np
from collections import defaultdict
from read_file import preprocess_original_breakpoints,save,select_original_breakpoints

def calculate_cond_xi_xi1(X,intervalsx,intervalsy):
    prob_xi = defaultdict(lambda:0)
    prob_xi1 = defaultdict(lambda:defaultdict(lambda:0))
    
    for x in X:
        for j in range(len(intervalsx)-1):
            if x[0] >= intervalsx[j] and x[0] < intervalsx[j+1]:
                prob_xi[j] += 1
                for k in range(len(intervalsy)-1):
                    if x[1] > intervalsy[k] and x[1] <= intervalsy[k+1]:
                        prob_xi1[j][k] += 1
                        break
    prob = np.zeros((len(intervalsy),len(intervalsx)))
    for i,prob_xj in prob_xi1.items():
        total = prob_xi[i]
        for j,p in prob_xj.items():
            prob[i][j] = p/total
    
    return prob

def get_i(V,intervals):
    idxs = []
    for v in V:
        for k in range(len(intervals)-1):
            if v > intervals[k] and v <= intervals[k+1]:
                idxs.append(k)
                break
    idxs = np.asarray(idxs)
    return idxs

def get_prob(X,intervalsx):
    idxs = get_i(X,intervalsx)
    unique,count = np.unique(idxs,return_counts=True)
    prob = np.zeros(max(unique)+1)
    total = len(idxs)
    # print('unique',unique)
    for u,c in zip(unique,count):
        prob[u] = c/total
    # print('prob',prob)
    return prob
    
def get_values_in_X_by_intervals(X,intervalsx):
    x_by_i = defaultdict(lambda:[])
    idxs = get_i(X,intervalsx)
    for i,x in zip(idxs,X):
        x_by_i[i].append(x)
    return x_by_i

def get_slope_val(prob,x_by_i):
    nx = len(prob)
    slopei = np.random.choice(np.arange(nx), p=prob)
    to_select = x_by_i[slopei]
    slopeiv = np.random.randint(0,len(to_select))
    slopev = to_select[slopeiv]
    return slopev,slopei

def generate_artificial_series(samples,prob_slope1,prob_cond,x_by_i,n):
    series = []
    for _ in range(samples):
        slope_artificial = np.zeros(n)
        slope_artificial[0],slope0_i = get_slope_val(prob_slope1,x_by_i[0])

        for i in range(n-1):
            prob = prob_cond[i][slope0_i]
            slope_artificial[i+1],slope0_i = get_slope_val(prob,x_by_i[i+1])
        
            series.append(slope_artificial)
    return series

def artificial_series(X,intervalsx,n,samples):
    x_by_i = [get_values_in_X_by_intervals(X[:,i],intervalsx) for i in range(n)]
    prob_X0 = get_prob(X[:,0],intervalsx)
    
    prob_cond = []
    for i in range(n-1):
        prob_cond.append(calculate_cond_xi_xi1(X[:,i:i+2],intervalsx,intervalsx))
    
    series_slopes = generate_artificial_series(samples,prob_X0,prob_cond,x_by_i,n)
    
    return series_slopes

def generate_artificial_data(Ns,samples,intervalsx,intervalsy,maxx,folder):

    for n in Ns:
        slopes,intervals = select_original_breakpoints(n)

        # INTERVALO SEGUINDO PROB COND/ANGULO MEDIO DE CADA INTERVALO
        mean_slopes = np.mean(slopes,axis=0)
        articifial_xs = artificial_series(intervals,intervalsy,n,samples)
        mean_slopes = [mean_slopes.tolist()]*samples
        articifial_xs = np.asarray(articifial_xs)
        save(mean_slopes,articifial_xs,'data/'+folder+'/plos_one_artificial_intervals_slope_axis0_'+str(n)+'_test.txt')

        # INTERVALO SEGUINDO PROB COND/ANGULO ALEATÓRIO DE CADA INTERVALO
        artificial_slopes = np.random.choice(slopes.flatten(),size=samples*n).reshape(samples,n)
        save(artificial_slopes,articifial_xs,'data/'+folder+'/plos_one_artificial_intervals_slope_random_'+str(n)+'_test.txt')

        # # TUDO ALEATORIO (qualquer eixo)
        artificial_slopes = (np.random.rand(samples,n)*maxx)
        artificial_intervals = np.random.rand(samples,n)
        save(artificial_slopes,artificial_intervals,'data/'+folder+'/plos_one_artificial_all_random_'+str(n)+'_test.txt')

        # SLOPES SEGUINDO PROB COND/ANGULO MEDIO DE CADA INTERVALO
        mean_intervals = np.mean(intervals,axis=0)
        articifial_slopes = artificial_series(slopes,intervalsx,n,samples)
        mean_intervals = [mean_intervals.tolist()]*samples
        mean_intervals = np.asarray(mean_intervals)
        save(articifial_slopes,mean_intervals,'data/'+folder+'/plos_one_artificial_slopes_interval_axis0_'+str(n)+'_test.txt')

        # # INTERVALO SEGUINDO PROB COND/ANGULO ALEATÓRIO DE CADA INTERVALO
        artificial_intervals = np.random.choice(intervals.flatten(),size=samples*n).reshape(samples,n)
        save(articifial_slopes,artificial_intervals,'data/'+folder+'/plos_one_artificial_slopes_interval_random_'+str(n)+'_test.txt')

        # INTERVALOS E SLOPES SEGUINDO O MODELO
        articifial_intevals = artificial_series(intervals,intervalsy,n,samples)        
        # articifial_intevals = np.asarray(articifial_intevals)
        articifial_slopes = artificial_series(slopes,intervalsx,n,samples)
        save(articifial_slopes,articifial_intevals,'data/'+folder+'/plos_one_artificial_intervals_slopes_'+str(n)+'_test.txt')