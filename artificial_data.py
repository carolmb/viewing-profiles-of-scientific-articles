import numpy as np
from collections import defaultdict
from read_file import save,select_original_breakpoints

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

def artificial_series_no_memory(X,intervalsx,n,samples):

    x_by_i = [get_values_in_X_by_intervals(X[:,i],intervalsx) for i in range(n)]
    
    X_artificial = np.zeros((samples,n))
    for i in range(n):
        prob_i = get_prob(X[:,i],intervalsx)
        for j in range(samples):
            x,_ = get_slope_val(prob_i,x_by_i[i])
            X_artificial[j][i] = x

    return X_artificial

def comb_prob(slopes,intervals,intervalsx,intervalsy):
    comb = defaultdict(lambda:defaultdict(lambda:[]))
    comb_next = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:[])))
    prob = defaultdict(lambda:defaultdict(lambda:0))
    prob_next = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:0)))
    for xs,ys in zip(slopes,intervals):
        for i,(x,y) in enumerate(zip(xs,ys)):
            x_idx = get_i([x],intervalsx)[0]
            y_idx = get_i([y],intervalsy)[0]
            comb[i][(x_idx,y_idx)].append((x,y))
            if i < len(xs)-1:
                x1_idx = get_i([xs[i+1]],intervalsx)[0]
                y1_idx = get_i([ys[i+1]],intervalsy)[0]
                comb_next[i][(x_idx,y_idx)][(x1_idx,y1_idx)].append((xs[i+1],ys[i+1]))

    for i,comb_i in comb.items():
        count = 0
        for k,v in comb_i.items():
            comb[i][k] = np.asarray(v)
            count += len(v)
        total = 0.0
        for k,v in comb_i.items():
            prob[i][k] = len(v)/count
            print(i,k,prob[i][k])
            total += prob[i][k]
        print('total',total)

        for k,v_k in comb_next[i].items():
            count = 0
            for k1,v in v_k.items():
                comb_next[i][k][k1] = np.asarray(v)
                count += len(v)
                total = 0.0
            for k1,v in v_k.items():
                prob_next[i][k][k1] = len(v)/count
                total += prob_next[i][k][k1]
            print('total',total)
    return prob,prob_next

def get_slope_inter(prob,x_by_i,y_by_i):
    nx = len(prob)

    i = np.random.choice(np.arange(nx), p=list(prob.values()))
    alpha_idx,l_idx = list(prob.keys())[i]

    # try:
    #     print('alpha_idx',alpha_idx,'x_by_i',x_by_i.keys())
    # except:
    #     print(x_by_i)

    x_to_select = x_by_i[alpha_idx]
    y_to_select = y_by_i[l_idx]

    x_idx = np.random.randint(0,len(x_to_select))
    y_idx = np.random.randint(0,len(y_to_select))
    x = x_to_select[x_idx]
    y = y_to_select[y_idx]
    return x,y

def generate_comb_artificial_data(X,Y,prob,prob_next,intervalsx,intervalsy,n,samples):
    alpha_by_i = [get_values_in_X_by_intervals(X[:,i],intervalsx) for i in range(n)]
    l_by_i = [get_values_in_X_by_intervals(Y[:,i],intervalsy) for i in range(n)]

    all_alphas = []
    all_ls = []
    for _ in range(samples):
        alphas = []
        ls = []
        for i in range(n):
            if i == 0:
                alpha,l = get_slope_inter(prob[i],alpha_by_i[i],l_by_i[i])
            else:
                alpha_idx = get_i([alpha],intervalsx)[0]
                l_idx = get_i([l],intervalsy)[0]
                #print(alpha_idx,l_idx,prob_next[i-1])
                p = prob_next[i-1][(alpha_idx,l_idx)]
                alpha,l = get_slope_inter(p,alpha_by_i[i],l_by_i[i])
            alphas.append(alpha)
            ls.append(l)
        all_alphas.append(alphas)
        all_ls.append(ls)
    return all_alphas,all_ls

def generate_artificial_data(Ns,intervalsx,intervalsy,maxx,folder):

    for n in Ns:
        slopes,intervals = select_original_breakpoints(n)
        samples = len(slopes)

        prob,prob_next = comb_prob(slopes,intervals,intervalsx,intervalsy)
        X,Y = generate_comb_artificial_data(slopes,intervals,prob,prob_next,intervalsx,intervalsy,n,samples)
        save(X,Y,'data/'+folder+'/plos_one_2019_artificial_comb_'+str(n)+'test.txt')
        continue

        # SEM MEMÓRIA
        X = artificial_series_no_memory(intervals,intervalsy,n,samples)
        Y = artificial_series_no_memory(slopes,intervalsx,n,samples)
        save(Y,X,'data/'+folder+'/plos_one_2019_artificial_no_memory_'+str(n)+'test.txt')

        # INTERVALO SEGUINDO PROB COND/ANGULO MEDIO DE CADA INTERVALO
        mean_slopes = np.mean(slopes,axis=0)
        articifial_xs = artificial_series(intervals,intervalsy,n,samples)
        mean_slopes = [mean_slopes.tolist()]*samples
        articifial_xs = np.asarray(articifial_xs)
        save(mean_slopes,articifial_xs,'data/'+folder+'/plos_one_2019_artificial_intervals_slope_axis0_'+str(n)+'_test.txt')

        # INTERVALO SEGUINDO PROB COND/ANGULO ALEATÓRIO DE CADA INTERVALO
        artificial_slopes = np.random.choice(slopes.flatten(),size=samples*n).reshape(samples,n)
        save(artificial_slopes,articifial_xs,'data/'+folder+'/plos_one_2019_artificial_intervals_slope_random_'+str(n)+'_test.txt')

        # # TUDO ALEATORIO (qualquer eixo)
        artificial_slopes = (np.random.rand(samples,n)*maxx)
        artificial_intervals = np.random.rand(samples,n)
        save(artificial_slopes,artificial_intervals,'data/'+folder+'/plos_one_2019_artificial_all_random_'+str(n)+'_test.txt')

        # SLOPES SEGUINDO PROB COND/ANGULO MEDIO DE CADA INTERVALO
        mean_intervals = np.mean(intervals,axis=0)
        articifial_slopes = artificial_series(slopes,intervalsx,n,samples)
        mean_intervals = [mean_intervals.tolist()]*samples
        mean_intervals = np.asarray(mean_intervals)
        save(articifial_slopes,mean_intervals,'data/'+folder+'/plos_one_2019_artificial_slopes_interval_axis0_'+str(n)+'_test.txt')

        # # INTERVALO SEGUINDO PROB COND/ANGULO ALEATÓRIO DE CADA INTERVALO
        artificial_intervals = np.random.choice(intervals.flatten(),size=samples*n).reshape(samples,n)
        save(articifial_slopes,artificial_intervals,'data/'+folder+'/plos_one_2019_artificial_slopes_interval_random_'+str(n)+'_test.txt')

        # INTERVALOS E SLOPES SEGUINDO O MODELO
        articifial_intevals = artificial_series(intervals,intervalsy,n,samples)        
        # articifial_intevals = np.asarray(articifial_intevals)
        articifial_slopes = artificial_series(slopes,intervalsx,n,samples)
        save(articifial_slopes,articifial_intevals,'data/'+folder+'/plos_one_2019_artificial_intervals_slopes_'+str(n)+'_test.txt')