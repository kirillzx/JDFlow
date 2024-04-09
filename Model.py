import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy

import torch
from torch import optim
from torch.autograd import Variable, grad
from torch.autograd.functional import jacobian
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize, brute
from functools import partial
from tqdm import tqdm
import itertools


from JDFLOW.intensity_optimization import *
from JDFLOW.signature_computation import *
from JDFLOW.stochastic_processes import *
from JDFLOW.nsde_functions import *
from JDFLOW.nsde_solvers import *
from JDFLOW.jdflow import *
from JDFLOW.evaluate.metrics import *
from JDFLOW.FourierFlows import FourierFlow
from JDFLOW.evaluate.metrics import *

import warnings
warnings.filterwarnings("ignore")

plt.style.use('ggplot')




#choose any of type process or upload your data. The main rule that length of each series must be the same.
def choose_data(name='DCL', n=300, M=10, **kwargs):
    dt = 1/n
    if name == 'DCL':
        data = dclProcess(n, M, theta=kwargs['theta'], delta=kwargs['delta'], T=kwargs['T']).T
    
    elif name == 'Merton':
        data = np.array([merton_process(s0=kwargs['s0'], xiP=kwargs['xiP'], muj=kwargs['muj'], sigmaj=kwargs['sigmaj'], r=kwargs['r'],\
                        sigma=kwargs['sigmad'], n=n, T=kwargs['T']) for i in range(M)])
    
    elif name == 'GBM':
        data = np.array([geometric_BM(s0=kwargs['s0'], mu=kwargs['mu'], sigma=kwargs['sigma'], n=n, T=kwargs['T']) for i in range(M)])
        
    elif name == 'Stock':
        data = pd.read_csv(kwargs['filename'])
        data = data[['Open', 'High', 'Low', 'Close']].values.T
        n = len(data[0])
        dt = 1/n
        
    return data, dt, n
        
        
def train(data, dt, n, n_flows, epochs, time_steps):
    data_save = copy.deepcopy(torch.FloatTensor(data))
    scaler = MinMaxScaler((0, 1))
    data = scaler.fit_transform(data.T).T
    data_tensor = torch.FloatTensor(data)
    # data0 = data_tensor[0]
    
    # identify initial Merton model params
    lambda_j = 0.6
    idx_jumps = list(map(lambda x: find_jumps(x, lambda_j), data_save))
    init_intensity = estimate_init_intensity_array(data_save, idx_jumps)/dt
    jump_part, diff_part = separate_dynamics(data_save, idx_jumps)
    init_params = estimate_init_params(data_save, jump_part, diff_part, dt)
    
    # optimize params
    opt_params = optimize_params(data_save, init_params, init_intensity, dt)
    
    xiP0 = opt_params[-1]
    
    # initialize hyperparameters
    h_dim = len(data_tensor[0])
    M = data_tensor.size(0)
    # time_steps = 100
    dt = torch.FloatTensor([1/time_steps])
    # n = data_tensor.size(0)
    sig_dim = n + n**2 
    
    # train
    # n_flows = 10
    nsde_flow = JDFlow(n_flows, h_dim, M, time_steps, dt, sig_dim, xiP0)
    losses = nsde_flow.fit(data_tensor, epochs)
    nsde_flow.eval()
    
    return nsde_flow, M, scaler, losses
    
    
def inverse_preprocessing(samples, M, scaler):
    synth_data = []

    if samples.shape[0] % M == 0:
        for i in range(samples.shape[0] // M):
            synth_data.append(scaler.inverse_transform(samples[M*i:M*(i+1)].T).T)
            
    else:
        for i in range(samples.shape[0] // M):
            synth_data.append(scaler.inverse_transform(samples[M*i:M*(i+1)].T).T)
            
        last = samples.shape[0] % M
        synth_data.append(scaler.inverse_transform(np.vstack([samples[-last:], np.ones((M - last, samples.shape[1]))]).T).T[:last])
        
    synth_data = np.vstack(synth_data)
    
    return synth_data

    
def sample(nsde_flow, n_samples, M, scaler):
    samples = nsde_flow.sample(n_samples)
    synth_data = inverse_preprocessing(samples, M, scaler)
    
    return synth_data
    
    
def objective_fun(z, data, dt, n):
    n_flows, epochs, time_steps = z
    nsde_flow, M, scaler, losses = train(data, dt, n, n_flows, epochs, time_steps)
    
    synth_data = sample(nsde_flow, n_samples=len(data), M=M, scaler=scaler)
    
    return w_dist_calc(synth_data, data)
    
def tuning_hyperparams(data, dt, n):
    bounds = (slice(2, 12, 2), slice(700, 1100, 100), slice(5, 130, 40))

    result = brute(partial(objective_fun, data=data, dt=dt, n=n), bounds, full_output=True)
    # minimize(lambda x: objective_fun(data, dt, n, x[0], x[1], x[2]))
    params = np.around(result[0]).astype(int)
    
    return params
    
    
    