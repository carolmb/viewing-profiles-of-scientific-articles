import numpy as np

'''
lê os arquivos originais com as séries temporais
'''
def read_file_original(filename):
    samples_breakpoints = open(filename,'r').read().split('\n')[:-1]
    total_series = len(samples_breakpoints)
    X = []
    Y = []
    for i in range(0,total_series,2):
        xs = [float(n) for n in samples_breakpoints[i].split(',')]
        ys = [float(n) for n in samples_breakpoints[i+1].split(',')]
        
        X.append(np.asarray(xs))
        Y.append(np.asarray(ys))
    
    return np.asarray(X),np.asarray(Y)

'''
lê as séries de slopes/intervalos geradas artificialmente a partir de modelo
retorna: os idxs das séries originais, slopes e intervalos
'''
def read_artificial_breakpoints(filename):
    samples_breakpoints = open(filename,'r').read().split('\n')[:-1]
    total_series = len(samples_breakpoints)
    slopes = []
    breakpoints = []
    # preds = []
    idxs = []
    for i in range(0,total_series,3):
        idx = int(samples_breakpoints[i]) - 1
        
        slopes_i = [float(n) for n in samples_breakpoints[i+1].split(' ')]
        breakpoints_i = [float(n) for n in samples_breakpoints[i+2].split(' ')]
        # preds_i = [float(n) for n in samples_breakpoints[i+3].split(' ')]
        idxs.append(idx)

        slopes.append(np.asarray(slopes_i))
        breakpoints.append(np.asarray(breakpoints_i))

    return np.asarray(idxs),np.asarray(slopes),np.asarray(breakpoints)

'''
lê os arquivos gerados pelo pacote segmented R
precisa de N para pegar apenas as séries com N intervalos
retorna: os idxs das séries originais, slopes, breakpoints e as predições
'''
def read_original_breakpoints(filename,N):
    samples_breakpoints = open(filename,'r').read().split('\n')[:-1]
    total_series = len(samples_breakpoints)
    slopes = []
    breakpoints = []
    idxs = []
    preds = []
    for i in range(0,total_series,4):
        idx = int(samples_breakpoints[i]) - 1
        
        slopes_i = [float(n) for n in samples_breakpoints[i+1].split(' ')]
        breakpoints_i = [float(n) for n in samples_breakpoints[i+2].split(' ')]
        y_pred_i = [float(n) for n in samples_breakpoints[i+3].split(' ')]
        # breakpoints_i.append(1.0) ???????

        if len(slopes_i) == N:
            idxs.append(idx)
            slopes.append(np.asarray(slopes_i))
            breakpoints.append(np.asarray(breakpoints_i))
            preds.append(np.asarray(y_pred_i))
    
    return np.asarray(idxs),np.asarray(slopes),np.asarray(breakpoints),np.asarray(preds)

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
