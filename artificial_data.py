import numpy as np
from scipy import stats
from collections import defaultdict

random_state = np.random.RandomState(9)
print(random_state)


def get_i(V, intervals):
    idxs = []
    for v in V:
        for k in range(len(intervals) - 1):
            if intervals[k] < v <= intervals[k + 1]:
                idxs.append(k)
                break
    idxs = np.asarray(idxs)
    return idxs


def get_prob(X, intervalsx):
    idxs = get_i(X, intervalsx)
    unique, count = np.unique(idxs, return_counts=True)
    prob = np.zeros(max(unique) + 1)
    total = len(idxs)
    # print('unique',unique)
    for u, c in zip(unique, count):
        prob[u] = c / total
    # print('prob',prob)
    return prob


def get_values_in_X_by_intervals(X, intervalsx):
    x_by_i = defaultdict(lambda: [])
    idxs = get_i(X, intervalsx)
    for i, x in zip(idxs, X):
        x_by_i[i].append(x)
    return x_by_i


def get_slope_val(prob, x_by_i):
    nx = len(prob)
    slopei = random_state.choice(np.arange(nx), p=prob)
    to_select = x_by_i[slopei]
    slopeiv = random_state.randint(0, len(to_select))
    slopev = to_select[slopeiv]
    return slopev, slopei


def generate_artificial_series(samples, prob_x0, prob_cond, n, intervals):
    print(intervals)
    series = np.zeros((samples, n))
    X0 = prob_x0(samples, random_state).flatten()
    for i in range(samples):

        x = max(X0[i], intervals[0] + 0.001)
        x = min(x, intervals[-1])
        series[i][0] = x

        control = 0
        j = 0
        while j < n - 1:
            control += 1

            # for j in range(n-1):
            try:
                if control == 50:
                    print('sample', i)
                    print('x', x, 'intervals', intervals)
                    print('i_idx', i_idx, prob_cond[j][i_idx])

                    break
                i_idx = get_i([x], intervals)[0]
                x = max(prob_cond[j][i_idx](1, random_state)[0][0], intervals[0] + 0.001)
                x = min(x, intervals[-1])
                series[i][j + 1] = x
                j += 1
            except:
                j = 0
                x = max(X0[i], intervals[0] + 0.001)
                x = min(x, intervals[-1])

        # print(x)

    return series


def get_joint_prob(X, intervals):
    n = X.shape[1] - 1
    dist_i = defaultdict(lambda: defaultdict(lambda: []))
    for x in X:
        for i in range(n):
            i_idx = get_i([x[i]], intervals)[0]
            dist_i[i][i_idx].append(x[i + 1])
            # dist_i[i].append((x[i],x[i+1]))

    pdf = defaultdict(lambda: defaultdict(lambda: None))
    # pdf = defaultdict(lambda:None)
    for i, dist_idx in dist_i.items():
        # pdf[i] = stats.gaussian_kde(dist_idx).resample
        # continue

        for idx, dist in dist_idx.items():
            try:
                pdf[i][idx] = stats.gaussian_kde(dist).resample
            except:
                a = dist[0]
                pdf[i][idx] = lambda x, y: [[a]]
    return pdf


def artificial_series(X, intervalsx, n, samples):
    prob_X0 = stats.gaussian_kde(X[:, 0]).resample

    joint_prob = get_joint_prob(X, intervalsx)

    series_slopes = generate_artificial_series(samples, prob_X0, joint_prob, n, intervalsx)

    return series_slopes


def artificial_series_no_memory(X, intervalsx, n, samples):
    # x_by_i = [get_values_in_X_by_intervals(X[:,i],intervalsx) for i in range(n)]

    X_artificial = np.zeros((samples, n))
    for i in range(n):
        prob_i = stats.gaussian_kde(X[:, i])  # get_prob(X[:,i],intervalsx)
        pdf_samples = prob_i.resample(samples).flatten()
        for j, x in enumerate(pdf_samples):
            # print('x:',x)
            X_artificial[j][i] = x

    return X_artificial


def comb_prob(slopes, intervals, intervalsx, intervalsy):
    # kde = stats.gaussian_kde([temp[:,0],temp[:,1],temp[:,2]])

    comb = defaultdict(lambda: defaultdict(lambda: []))
    alpha_comb_next = defaultdict(lambda: defaultdict(lambda: []))
    l_comb_next = defaultdict(lambda: defaultdict(lambda: []))
    prob = defaultdict(lambda: defaultdict(lambda: 0))
    alpha_prob_next = defaultdict(lambda: defaultdict(lambda: None))
    l_prob_next = defaultdict(lambda: defaultdict(lambda: None))
    for xs, ys in zip(slopes, intervals):
        for i, (x, y) in enumerate(zip(xs, ys)):
            x_idx = get_i([x], intervalsx)[0]
            y_idx = get_i([y], intervalsy)[0]
            comb[i][(x_idx, y_idx)].append((x, y))
            if i < len(xs) - 1:
                x1_idx = get_i([xs[i + 1]], intervalsx)[0]
                y1_idx = get_i([ys[i + 1]], intervalsy)[0]
                alpha_comb_next[i][(x_idx, y_idx)].append(xs[i + 1])
                l_comb_next[i][(x_idx, y_idx)].append(ys[i + 1])

    for i, comb_i in comb.items():
        count = 0
        for k, v in comb_i.items():
            comb[i][k] = np.asarray(v)
            count += len(v)
        total = 0.0
        # print('i',i)
        for k, v in comb_i.items():
            prob[i][k] = len(v) / count
            # print(k,prob[i][k])
            total += prob[i][k]
        # print('total',total)

        for k, v_k in alpha_comb_next[i].items():
            # print(alpha_comb_next[i][k][0])
            # print(l_comb_next[i][k][0])
            try:
                alpha_prob_next[i][k] = stats.gaussian_kde(alpha_comb_next[i][k]).resample
                l_prob_next[i][k] = stats.gaussian_kde(l_comb_next[i][k]).resample
            except:
                z = alpha_comb_next[i][k][0]
                w = l_comb_next[i][k][0]
                alpha_prob_next[i][k] = lambda x, y: [[z]]
                l_prob_next[i][k] = lambda x, y: [[w]]
    return prob, alpha_prob_next, l_prob_next


def get_slope_inter(prob, x_by_i):
    nx = len(prob)

    i = random_state.choice(np.arange(nx), p=list(prob.values()))
    bucket = list(prob.keys())[i]

    # try:
    #     print('alpha_idx',alpha_idx,'x_by_i',x_by_i.keys())
    # except:
    #     print(x_by_i)

    # print(bucket,x_by_i[bucket])

    x_to_select = x_by_i[bucket]

    x_idx = random_state.randint(0, len(x_to_select))
    x = x_to_select[x_idx]
    return x


def get_slope_inter_conj(prob, x_by_i, y_by_i):
    nx = len(prob)

    i = random_state.choice(np.arange(nx), p=list(prob.values()))
    alpha_bucket, l_bucket = list(prob.keys())[i]

    # try:
    #     print('alpha_idx',alpha_idx,'x_by_i',x_by_i.keys())
    # except:
    #     print(x_by_i)

    x_to_select = x_by_i[alpha_bucket]
    y_to_select = y_by_i[l_bucket]

    x_idx = random_state.randint(0, len(x_to_select))
    x = x_to_select[x_idx]
    y_idx = random_state.randint(0, len(y_to_select))
    y = y_to_select[y_idx]
    return x, y


def generate_comb_artificial_data(X, Y, prob, alpha_prob_next, l_prob_next, intervalsx, intervalsy, n, samples):
    alpha_by_i = [get_values_in_X_by_intervals(X[:, i], intervalsx) for i in range(n)]
    l_by_i = [get_values_in_X_by_intervals(Y[:, i], intervalsy) for i in range(n)]

    all_alphas = []
    all_ls = []
    j = 0
    for count in range(samples):
        # print(count)
        alphas = []
        ls = []
        i = 0
        while i < n:
            j += 1
            if i == 0:
                alpha, l = get_slope_inter_conj(prob[i], alpha_by_i[i], l_by_i[i])
            else:

                alpha_idx = get_i([alpha], intervalsx)[0]
                l_idx = get_i([l], intervalsy)[0]

                if not (alpha_idx, l_idx) in prob[i - 1]:
                    if j == 50:
                        print('j == 50')
                        print(alpha_idx)
                        print(l_idx)
                        print(prob[i - 1])
                        print('break')
                        break

                    # print("alpha_idx:",alpha_idx,"        l_idx:",l_idx)
                    # print("alpha_prob_next",alpha_prob_next[i-1].keys())
                    # print("l_prob_next",l_prob_next[i-1].keys())
                    # print(alphas,ls)
                    i = i - 1
                    alphas = alphas[:-1]
                    ls = ls[:-1]
                    continue

                # if l_prob_next[i-1][(alpha_idx,l_idx)] == []:
                #    i = i - 1
                #    continue

                # print(alpha_idx,l_idx,prob_next[i-1])

                # p = alpha_prob_next[i-1][(alpha_idx,l_idx)]
                # alpha = get_slope_inter(p,alpha_by_i[i])
                alpha = max(min(alpha_prob_next[i - 1][(alpha_idx, l_idx)](1, random_state)[0][0], 89), 3)
                # p = l_prob_next[i-1][(alpha_idx,l_idx)]
                # l = get_slope_inter(p,l_by_i[i])
                l = min(max(l_prob_next[i - 1][(alpha_idx, l_idx)](1, random_state)[0][0], 0.01), 0.99)

                # print(alpha,l)
            i = i + 1
            alphas.append(alpha)
            ls.append(l)
        all_alphas.append(alphas)
        all_ls.append(ls)

    return all_alphas, all_ls


def generate_artificial_data(all_slopes, all_intervals, n, intervalsx, intervalsy, maxx, ):
    models = defaultdict(lambda: [])

    for slopes, intervals in zip(all_slopes, all_intervals):
        samples = len(slopes)

        print("Markov-1 multivaribles model")
        prob, alpha_prob_next, l_prob_next = comb_prob(slopes, intervals, intervalsx, intervalsy)
        X, Y = generate_comb_artificial_data(slopes, intervals, prob, alpha_prob_next, l_prob_next, intervalsx,
                                             intervalsy, n, samples)
        models['markov1_multi'].append((X, Y))
        # save(X,Y,folder+'markov1_multi_'+str(n)+'_gaussian_test.txt')

        print("No memory model")
        # SEM MEMÓRIA
        X = artificial_series_no_memory(slopes, intervalsx, n, samples)
        Y = artificial_series_no_memory(intervals, intervalsy, n, samples)
        models['no_memory'].append((X, Y))
        # save(X,Y,folder+'no_memory_'+str(n)+'_gaussian_test.txt')

        print("Null model")
        # TUDO ALEATORIO (qualquer eixo)
        X = (random_state.rand(samples, n) * maxx)
        Y = random_state.rand(samples, n)
        models['null_model'].append((X, Y))
        # save(X,Y,folder+'artificial_all_random_'+str(n)+'_test.txt')

        print("Markov-1 univariavel model")
        # INTERVALOS E SLOPES SEGUINDO O MODELO
        X = artificial_series(slopes, intervalsx, n, samples)
        Y = artificial_series(intervals, intervalsy, n, samples)
        models['markov1_uni'].append((X, Y))
        # save(X,Y,folder+'markov1_uni_'+str(n)+'_gaussian_test.txt')

    return models
