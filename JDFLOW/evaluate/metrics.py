import numpy as np
from scipy.signal import argrelextrema
from scipy.spatial.distance import jensenshannon
from scipy.stats import kstest, ks_2samp, wasserstein_distance
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from IPython.display import display


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


def w_dist_calc(real, synth):
    w_dist = []

    for i in range(len(real)):
        w_dist.append(wasserstein_distance(real[i], synth[i]))
    
    return np.mean(w_dist)


def js_calc(real, synth):
    w_dist = []

    for i in range(len(real)):
        w_dist.append(jensenshannon(real[i], synth[i]))
    
    return np.mean(w_dist)


def forecast_metrics(real_data, synth_data, synth_PAR, synth_fsde, n=31):
    train_size_init = int(len(real_data[0]) - n*test_size)

    test_size = len(real_data[0])//n - 1

    def forecasting_cv(synth, real, train_size_init, test_size, n):
        array_mse = []
        train = synth[:, :train_size_init]
        

        
        for j in range(len(real)):
            train_iter = copy.deepcopy(train[j])
            mse_iter = []
            train_size = train_size_init
            
            for _ in range(n-2):
                df = pd.DataFrame()
                df['y'] = train_iter
                df.index = pd.date_range(start='1/1/2022', periods=len(train_iter), freq='D')
                df = df.reset_index()
                df = df.rename(columns={'index': 'ds'})

                m = Prophet(daily_seasonality=True)
                m.fit(df)
                
                future = m.make_future_dataframe(periods = test_size)
                pred_synth = m.predict(future)['yhat'][-test_size:]
                            
                test_synth = synth[j, train_size : (train_size+test_size)]
                test_real = real[j, train_size : (train_size+test_size)]
                
                mse = mean_squared_error(pred_synth.values, test_real)
                
                train_iter = np.concatenate([train_iter, test_synth])
                
                train_size += test_size
        
                
                mse_iter.append(mse)
        
            
            array_mse.append(np.mean(mse_iter))
            
        return array_mse    
    
    mse_real = np.array(forecasting_cv(real_data, real_data, train_size_init, test_size, n))
    mse_jdflow = np.array(forecasting_cv(synth_data, real_data, train_size_init, test_size, n))
    mse_ff = np.array(forecasting_cv(synth_ff, real_data, train_size_init, test_size, n))
    mse_par = np.array(forecasting_cv(synth_PAR, real_data, train_size_init, test_size, n))
    mse_fsde = np.array(forecasting_cv(synth_fsde, real_data, train_size_init, test_size, n))
    
    display(pd.DataFrame([[mse_real.mean(), np.min(mse_real), np.max(mse_real)],
              [mse_jdflow.mean(), np.min(mse_jdflow), np.max(mse_jdflow)],
              [mse_ff.mean(), np.min(mse_ff), np.max(mse_ff)],
              [mse_par.mean(), np.min(mse_par), np.max(mse_par)],
              [mse_fsde.mean(), np.min(mse_fsde), np.max(mse_fsde)]],\
             columns=['Mean', 'Min', 'Max'], index = ['Real', 'JDFlow', 'Fourier Flow', 'PAR', 'fSDE']))