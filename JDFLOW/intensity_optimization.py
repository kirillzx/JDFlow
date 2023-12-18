import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.special import factorial
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')



def log_return(data):
    # data = data + 1e-3
    return np.log(data[1:]) - np.log(data[:-1])

def find_jumps(data: torch.Tensor, ths: float = 0.1):
    # jumps = torch.abs(data[:-1] - data[1:])
    jumps = torch.abs(log_return(data))
    idx_jumps = torch.where(jumps > ths)[0]
    
    return idx_jumps


def plot_jumps(data_tensor, idx_jumps):
    plt.subplots(figsize=(10, 5), dpi=100)

    for i in range(len(data_tensor)):
        plt.plot(data_tensor[i])
        plt.scatter(idx_jumps[i], data_tensor[i][idx_jumps[i]], color='black')
        plt.xlabel('Time')
        plt.ylabel('Values')
        
    plt.show()
    
    
def calc_mean_jump(data: list, idx: list):
    jumps = []
    for i in range(len(idx)):
        jumps.append(torch.mean(data[i][idx[i]]))
        
    jumps = torch.stack(jumps)
        
    return torch.mean(jumps[~jumps.isnan()])

def estimate_init_intensity(data, idx_jumps) -> float:
    return len(idx_jumps) / len(data)

def estimate_init_intensity_array(data: torch.Tensor, idx_jumps: list) -> np.array:
    return np.array(list(map(lambda x,y: estimate_init_intensity(x, y), data, idx_jumps)))


def separate_dynamics(data: list, idx: list):
    jumps = []
    diff = []
    for i in range(len(idx)):
        jumps.append(data[i][idx[i]])
        diff.append(data[i][np.setdiff1d(np.arange(len(data[i])), idx[i])])
        
    return jumps, diff



def estimate_init_mu(data, dt):
    if data.size(0) == 1:
        return (2 * torch.mean(data) + torch.var(data) * dt) / (2 * dt)
    else:
        return (2 * torch.mean(data, dim=1) + torch.var(data, dim=1) * dt) / (2 * dt)
    
def estimate_init_var(data, dt):
    if data.size(0) == 1:
        return torch.var(data) / dt
    else:
        return torch.var(data, dim=1) / dt



def estimate_init_mu_j(data, dt, mu_hat, var_hat):
    if data.size(0) == 1:
        return torch.mean(data) - (mu_hat - var_hat/2) * dt
    else:
        return torch.mean(data, dim=1) - (mu_hat - var_hat/2) * dt
    
def estimate_init_var_j(data, dt, var_hat):
    if data.size(0) == 1:
        return torch.var(data) - var_hat * dt
    else:
        return torch.var(data, dim=1) - var_hat * dt
    
    
    
def estimate_init_params(data, jump_part, diff_part, dt):
    params = []
    for i in range(len(data)):
        r_j = log_return(jump_part[i]).view(1, -1)
        r_d = log_return(diff_part[i]).view(1, -1)
        
        mu_hat = estimate_init_mu(r_d, dt)
        var_hat = estimate_init_var(r_d, dt)
        
        mu_j_hat = estimate_init_mu_j(r_j, dt, mu_hat, var_hat)
        var_j_hat = estimate_init_var_j(r_j, dt, var_hat)
        
        params.append([mu_hat, var_hat, mu_j_hat, var_j_hat])
        
    return np.array(params)


def likelihood_mjd(data, params, intensity, dt):
    like = 0
    for i in range(len(data)):
        for k in range(3):
            mu = (params[0] - params[1]/2) * dt + params[2] * k
            var = params[1] * dt + params[3] * k
            l = (intensity * dt)**k
            
            like += np.log( (l * np.exp(-intensity * dt) / factorial(k))  * (np.exp(-(data[i] - mu)**2 / (2 * var)) / np.sqrt(2 * np.pi * var)) )
            
    return like


def optimize_params(data, params, intensity, dt):
    opt_params = []
    for i in range(len(data)):
        optimization = minimize(lambda x: -likelihood_mjd(log_return(data[i]), [x[0], x[1], x[2], x[3]], x[4], dt),\
                                                    x0=np.hstack([params[i], intensity[i]]))
        opt_params.append(optimization.x)
        
    return np.mean(np.array(opt_params), 0)