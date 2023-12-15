import numpy as np
from scipy.signal import argrelextrema



def autocorr(x):
    var = x.var()
    mean = x.mean()
    x = x - mean

    result = np.correlate(x, x, mode='full')
    return list(result[len(result)//2:]/var/len(x))

autocorr_vec = lambda x: np.mean(list(map(autocorr, x)), 0)


def qunatile_comp_1d(data):
    local_extr_min = data[argrelextrema(data, np.less_equal, order=10)[0]]
    local_extr_max = data[argrelextrema(data, np.greater_equal, order=10)[0]]
    extr = np.concatenate([local_extr_min, local_extr_max])
    local_sort_extr = np.sort(extr)
    quantiles = [local_sort_extr.min()]

    for i in range(1, 100):
        quantiles.append(np.quantile(local_sort_extr, i/100))

    quantiles.append(local_sort_extr.max())
    
    return quantiles


def extr_quant_computation(data):
    res = []
    for i in data:
        res.append(qunatile_comp_1d(i))
        
    return np.mean(res, axis=0)