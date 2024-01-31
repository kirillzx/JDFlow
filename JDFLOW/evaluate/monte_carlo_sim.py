import pandas as pd
import numpy as np

import torch
from tqdm import tqdm
from torch import optim
from torch.autograd import Variable, grad
from torch.autograd.functional import jacobian
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from JDFLOW.intensity_optimization import *
from JDFLOW.signature_computation import *
from JDFLOW.stochastic_processes import *
from JDFLOW.nsde_functions import *
from JDFLOW.nsde_solvers import *
from JDFLOW.jdflow import *
from JDFLOW.evaluate.metrics import *
from JDFLOW.FourierFlows import FourierFlow
from JDFLOW.FSDE import train
from JDFLOW.evaluate.metrics import *
from Model import *


def monte_carlo_train(data, N, dt, n, n_flows, epochs, time_steps):
    res_arr = []
    for _ in range(N):
        nsde_flow, M, scaler, _ = train(data, dt, n, n_flows, epochs, time_steps)
        synth_data = sample(nsde_flow, n_samples=len(data), M=M, scaler=scaler)
        
        res_arr.append(synth_data)
        
    return np.mean(res_arr, axis=0)


def monte_carlo_sample(N, n_samples, nsde_flow, M, scaler):
    res_arr = []
    for _ in range(N):
        synth_data = sample(nsde_flow, n_samples=n_samples, M=M, scaler=scaler)
        
        res_arr.append(synth_data)
        
    return np.mean(res_arr, axis=0)