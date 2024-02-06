import numpy as np
import torch

def ornstein_uhlenbeck_process(n, m, mu, theta, sigma, x0):
    res = []

    T = 1
    
    for j in range(m):
        x = np.zeros(n + 1)
        s = np.zeros(n + 1)
        time = np.zeros(n + 1)
        
        # x[0] = np.random.normal(0, 0.1)
        x[0] = x0
        dt = T/float(n)

        for t in range(n):
            x[t+1] = x[t] + theta*(mu - x[t]) * dt + sigma * np.sqrt(dt) * np.random.normal(loc=0, scale=1)
            time[t+1] = time[t] + dt

        res.append(x)
    
    return np.array(res)


def likelihood_OU(data, mu, theta, sigma, x0):
    s = 0
    for i in range(len(data)):
        t = i+1
        s += torch.log( 1 / (torch.sqrt(torch.tensor(2*torch.pi)) * torch.sqrt(torch.tensor(sigma**2 * (1 - torch.exp(torch.tensor(-2*theta*t)))/(2*theta)))) ) +\
            - 1/(2 * (sigma**2 * (1 - torch.exp(torch.tensor(-2*theta*t)))/(2*theta))) * (data[i] - mu - (x0-mu)*torch.exp(torch.tensor(-theta*t)))**2
            
    return s

def dclProcess(N, M, theta=1, delta=2, T=1):

    Z1 = np.random.normal(0.0, 1.0, [M, N])
    X = np.zeros([M, N + 1])

    X[:, 0] = np.random.normal(0.0, 0.2, M)

    time = np.zeros([N+1])
    dt = T / float(N)
    
    for i in range(0, N):

        X[:,i+1] = X[:, i] - 1/theta * X[:,i] * dt + np.sqrt((1 - (X[:, i])**2)/(theta * (delta + 1))) * np.sqrt(dt) * Z1[:,i]
            
        if (X[:,i+1] > 1).any():
            X[np.where(X[:,i+1] >= 1)[0], i+1] = 0.9999

        if (X[:,i+1] < -1).any():
            X[np.where(X[:,i+1] <= -1)[0], i+1] = -0.9999 
            
        time[i+1] = time[i] + dt

    return X.T

def merton_process(s0, xiP, muj, sigmaj, r, sigma, n, T):
        time = np.zeros(n + 1)
        dt = T/float(n)
        
        z = np.random.normal(0.0, 1.0, n + 1)
        zj = np.random.normal(muj, sigmaj, n + 1)
        poisson_distr = np.random.poisson(xiP * dt, n + 1)
        
        x = np.zeros(n + 1)
        s = np.zeros(n + 1)
        
        s[0] = s0
        x[0] = np.log(s0)
        
        EeJ = np.exp(muj + 0.5 * sigmaj**2)

        for t in range(n):
            x[t+1] = x[t] + (r - xiP * (EeJ - 1) - 0.5 * sigma**2) * dt +\
                sigma * np.sqrt(dt) * z[t] + zj[t] * poisson_distr[t]
            
            time[t+1] = time[t] + dt
            
        s = np.exp(x)
        
        return s
    
def geometric_BM(s0, mu, sigma, n, T):
        x = np.zeros(n + 1)
        s = np.zeros(n + 1)
        time = np.zeros(n + 1)
        
        x[0] = np.log(s0)
        dt = T/float(n)
        
        for t in range(n):
            x[t+1] = x[t] + (mu - (sigma**2)/2) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1)
            time[t+1] = time[t] + dt
            
        s = np.exp(x)
        
        return s