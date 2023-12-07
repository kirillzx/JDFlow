import torch
import numpy as np


'''
Forward Functions
'''

def sdeint(drift, diffusion, dt, x0, n, h_dim, w_dim):
    wiener_process = torch.sqrt(dt) * torch.FloatTensor(np.random.normal(0, 1, size=(n, 1, h_dim)))
    solution = []
    diff_array = []
    solution.append(x0)
    t = torch.FloatTensor([0])
    
    if t.size(0) != x0.size(0):
        t = t.repeat(x0.size(0)).view(-1, 1)
        
        for i in range(n - 1):
            t_emb = torch.cat([t, torch.sin(t), torch.cos(t)], axis=1)
            x_next = solution[i] + drift(t_emb, solution[i]) * dt + diffusion(t_emb, solution[i]) * wiener_process[i] 
            # x_next = solution[i] + drift(t, solution[i]) * dt + torch.matmul(diffusion(t, solution[i]), wiener_process[i].T).T
            t = t + dt
            solution.append(x_next)
            diff_array.append(diffusion(t_emb, solution[i]))
            
    else:
        for i in range(n - 1):
            t_emb = torch.cat([t, torch.sin(t), torch.cos(t)], axis=0).view(1, -1)
            x_next = solution[i] + drift(t_emb, solution[i]) * dt + diffusion(t_emb, solution[i]) * wiener_process[i] 
            # x_next = solution[i] + drift(t, solution[i]) * dt + torch.matmul(diffusion(t, solution[i]), wiener_process[i].T).T
            t = t + dt
            solution.append(x_next)
            
        
    if h_dim == 1:
        solution = torch.tensor(solution, requires_grad=True).view(-1, 1)
        # solutions = solutions + solution
        
    else:
        solution = torch.stack(solution, 1).squeeze()
        # solutions = solutions + solution
            
    return solution, diff_array

def sdeint_V(drift, diffusion, dt, x0, n, h_dim, set_A):
    wiener_process = torch.sqrt(dt) * torch.FloatTensor(np.random.normal(0, 1, size=(n, 1, h_dim)))
        
    solution = []
    solution.append(x0)
    t = torch.FloatTensor([0])
    
    for i in range(n - 1):
        x_next = solution[i] + drift(t, solution[i], set_A[i]) * dt + diffusion(t, solution[i], set_A[i]) * wiener_process[i] 
        t = t + dt
        solution.append(x_next)

    if h_dim == 1:
        solution = torch.tensor(solution, requires_grad=True).view(-1, 1)
        # print(torch.stack(solution).view(-1, 1))
        
    else:
        solution = torch.stack(solution, 1).squeeze()
            
    return solution



def sdeint_jump(drift, diffusion, jump, dt, x0, n, h_dim, xiP):
    # xiP = xi()
    wiener_process = torch.sqrt(dt) * torch.FloatTensor(np.random.normal(0, 1, size=(n, 1, h_dim)))
    poisson_distr = torch.FloatTensor(torch.poisson(torch.ones((n, 1, h_dim)) * xiP * dt))
     # poisson_distr = torch.FloatTensor(np.random.poisson(xiP * dt, size=(n, 1, h_dim)))
    solution = []
    solution.append(x0)
    t = torch.FloatTensor([0])
   
    
    # muj = 0
    # sigmaj = 1
    # zJ = np.random.normal(muj, sigmaj, n + 1)
    t = t.repeat(x0.size(0)).view(-1, 1)
    
    for i in range(n - 1):
        t_emb = torch.cat([t, torch.sin(t), torch.cos(t)], axis=1)
        x_next = solution[i] + drift(t_emb, solution[i]) * dt + diffusion(t_emb, solution[i]) * wiener_process[i] +\
            jump(t_emb, solution[i]) * poisson_distr[i]
            
        t = t + dt
        solution.append(x_next)
        
    if h_dim == 1:
        solution = torch.tensor(solution, requires_grad=True).view(-1, 1)
        
    else:
        solution = torch.stack(solution, 1).squeeze()
            
    return solution


def sdeint_jump_1d(drift, diffusion, jump, dt, x0, n):
    solution = []
    solution.append(x0)
    t = torch.FloatTensor([0])
    t = t.repeat(len(x0)).view(-1, 1)
    
    xiP = 1
    
    for i in range(n-1):
        t_emb = torch.cat([t, torch.sin(t), torch.cos(t)], axis=1)
        x_next = solution[i] + drift(t_emb, solution[i]) * dt + diffusion(t_emb, solution[i]) * torch.sqrt(dt) * torch.randn(1) +\
                                jump(t_emb, solution[i]) * torch.FloatTensor(np.random.poisson(xiP * dt))
        solution.append(x_next)
        t = t + dt
                                
    return solution

# def sdeint_jump(drift, diffusion, jump, dt, x0, n, h_dim):
#     wiener_process = torch.sqrt(dt) * torch.FloatTensor(np.random.normal(0, 1, size=(n, 1, h_dim)))
    
#     solution = []
#     solution.append(x0)
#     t = torch.FloatTensor([0])
#     xiP = 1
#     poisson_distr = torch.FloatTensor(np.random.poisson(xiP * dt, size=(n, 1, h_dim)))
#     # muj = 0
#     # sigmaj = 1
#     # zJ = np.random.normal(muj, sigmaj, n + 1)
    
#     for i in range(n - 1):
#         x_next = solution[i] + drift(t, solution[i]) * dt + diffusion(t, solution[i]) * wiener_process[i] +\
#             jump(t, solution[i]) * poisson_distr[i]
            
#         t = t + dt
#         solution.append(x_next)
        
#     if h_dim == 1:
#         solution = torch.FloatTensor(solution).view(-1, 1)
        
#     else:
#         solution = torch.stack(solution, 1).squeeze()
            
#     return solution


'''
Inverse Functions
'''

def sdeint_inverse(drift, diffusion, dt, x, n, h_dim, w_dim):
    wiener_process = torch.sqrt(dt) * torch.FloatTensor(np.random.normal(0, 1, size=(n, 1, h_dim)))
    # solution = torch.zeros(n, h_dim)
    # solution[-1] = x
    
    t = torch.FloatTensor([1])
    if t.size(0) != x.size(0):
        t = t.repeat(x.size(0)).view(-1, 1)
        solution = torch.zeros(n, x.size(0), h_dim)
        solution[-1] = x
        
        for i in range(n-1, 0, -1):
            t_emb = torch.cat([t, torch.sin(t), torch.cos(t)], axis=1)
            x_prev = solution[i] - drift(t_emb, solution[i]) * dt - diffusion(t_emb, solution[i]) * wiener_process[i]
            # x_prev = solution[i].view(1, h_dim) - drift(t, solution[i].view(1, h_dim)) * dt - torch.matmul(diffusion(t, solution[i].view(1, h_dim)), wiener_process[i].T).T
            t = t - dt

            solution[i-1] = x_prev
    
    else:
        solution = torch.zeros(n, h_dim)
        solution[-1] = x
        
        for i in range(n-1, 0, -1):
            t_emb = torch.cat([t, torch.sin(t), torch.cos(t)], axis=0).view(1, -1)
            x_prev = solution[i].view(1, h_dim) - drift(t_emb, solution[i].view(1, h_dim)) * dt - diffusion(t_emb, solution[i].view(1, h_dim)) * wiener_process[i]
            # x_prev = solution[i].view(1, h_dim) - drift(t, solution[i].view(1, h_dim)) * dt - torch.matmul(diffusion(t, solution[i].view(1, h_dim)), wiener_process[i].T).T
            t = t - dt
                
            solution[i-1] = x_prev
            
    return solution[0]

def sdeint_jump_inverse(drift, diffusion, jump, dt, x, n, h_dim, xiP):
    wiener_process = torch.sqrt(dt) * torch.FloatTensor(np.random.normal(0, 1, size=(n, 1, h_dim)))
    # poisson_distr = torch.FloatTensor(np.random.poisson(xiP * dt, size=(n, 1, h_dim)))
    poisson_distr = torch.FloatTensor(torch.poisson(torch.ones((n, 1, h_dim)) * xiP * dt))
    
    solution = torch.zeros(n, x.size(0), h_dim)
    solution[-1] = x
    
    t = torch.FloatTensor([1])
    t = t.repeat(x.size(0)).view(-1, 1)
    solution = torch.zeros(n, x.size(0), h_dim)
    solution[-1] = x

    for i in range(n-1, 0, -1):
        t_emb = torch.cat([t, torch.sin(t), torch.cos(t)], axis=1)
        
        x_prev = solution[i] - drift(t_emb, solution[i]) * dt - diffusion(t_emb, solution[i]) * wiener_process[i] -\
            jump(t_emb, solution[i]) * poisson_distr[i]
        t = t - dt
            
        solution[i-1] = x_prev
            
    return solution[0]