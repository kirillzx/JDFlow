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


class extract_tensor(nn.Module):
    def forward(self,x):
        tensor, _ = x
        return tensor
    
class DiffMap(nn.Module):
    def __init__(self, time_steps, h_dim, M, dt, permute):
        super(DiffMap, self).__init__()
    
        self.h_dim = h_dim
        self.laten_dim = M
        self.time_steps = time_steps
        self.dt = dt

        hidden = 2**7
        self.v_dim = M
        self.d = h_dim // 2
        self.r = M // 2
        self.permute = permute
        
        self.in_size = self.d
        self.out_size = self.h_dim - self.d

    
        if self.permute:
            self.in_size, self.out_size = self.out_size, self.in_size

        
        self.fc_mu = nn.Sequential(
            # nn.LSTM(self.in_size, self.in_size, num_layers=1, batch_first=True),
            # extract_tensor(),
            # nn.Sigmoid(),
            nn.Linear(self.in_size, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, self.out_size)
        )
        self.fc_sig = nn.Sequential(
            nn.LSTM(self.in_size, self.in_size, num_layers=1, batch_first=True),
            extract_tensor(),
            nn.Sigmoid(),
            nn.Linear(self.in_size, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, self.out_size)
        )
        self.nu = nn.Sequential(
            nn.Linear(M, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, M)
        )
   
        self.p = nn.Sequential(
            nn.Linear(self.time_steps, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, self.in_size)
        )
        self.tanh = nn.Tanh()
        
        self.f = nn.Sequential(nn.Linear(self.in_size, hidden),
                               nn.Sigmoid(),
                               nn.Linear(hidden, self.out_size))
        
        self.base_dist = MultivariateNormal(torch.zeros(h_dim), torch.eye(h_dim))

    def forward(self, x, vt):
        x1, x2 = x[:, :self.d], x[:, self.d:]
    
        if self.permute:
            x2, x1 = x1, x2
        
        p_hat = self.tanh(self.p(vt.T))
        z1 = x1 + p_hat
        sig = self.fc_sig(z1)
        z2 = self.fc_mu(z1) * self.f(self.nu(z1.T).T) + x2 * torch.exp(sig)
        
        if self.permute:
            z2, z1 = z1, z2 
 
        z = torch.cat([z1, z2], axis=1)
        
        log_pz = self.base_dist.log_prob(z)
        log_jacob = sig.sum(-1)

        return z, log_pz, log_jacob
    
        

    def inverse(self, z, vt):
        z1, z2 = z[:, :self.d], z[:, self.d:]
        
        if self.permute:
            z2, z1 = z1, z2
            
        x2 = (z2 - self.fc_mu(z1) * self.f(self.nu(z1.T).T)) * torch.exp(-self.fc_sig(z1))
        
        p_hat = self.tanh(self.p(vt.T))
        x1 = z1 - p_hat
      
        if self.permute:
            x2, x1 = x1, x2
        
        x = torch.cat([x1, x2], axis=1)

        return x



class JDFlow(nn.Module):
    def __init__(self, n_flows, h_dim, M, time_steps, dt, sig_dim, xiP0):
        super(JDFlow, self).__init__()
        
        self.h_dim = h_dim
        self.permute = [True if i % 2 else False for i in range(n_flows)]

        self.bijections = nn.ModuleList(
            [DiffMap(time_steps, h_dim, M, dt, self.permute[i]) for i in range(n_flows)]
        )
        
        self.xiP = xiP0
        
        self.time_steps = time_steps
        self.dt = dt
        self.v_dim = M
        self.psi = Operator_F(self.time_steps, self.h_dim)
        self.zeta = Operator_F(self.h_dim, self.h_dim)
        
        self.drift = Drift(self.v_dim)
        self.diffusion = Diffusion(self.v_dim, 1)
        self.jump = Jump(self.v_dim)
        self.phi = Phi(self.v_dim, M, M)
        self.sigmoid_sig = nn.Sigmoid()

    def forward(self, x):
        log_jacobs = []
        
        wt = torch.randn((1, self.v_dim))
        v0 = self.phi(wt, x[:, 0].view(1, x.size(0)))
        self.vt = sdeint_jump(self.drift, self.diffusion, self.jump, self.dt, v0, self.time_steps, self.v_dim, self.xiP)
        
        self.non_linearity = torch.exp(-self.psi(self.vt.T))
        x = self.non_linearity * x + self.zeta(self.non_linearity)
        

        for bijection in self.bijections:
            x, log_pz, lj = bijection(x, self.vt)

            log_jacobs.append(lj)

        return x, log_pz, sum(log_jacobs)

    def inverse(self, z):
        
        wt = torch.randn((1, self.v_dim))
        v0 = self.phi(wt, z[:, 0].view(1, z.size(0)))
        self.vt = sdeint_jump(self.drift, self.diffusion, self.jump, self.dt, v0, self.time_steps, self.v_dim, self.xiP)

        for bijection in reversed(self.bijections):
            z = bijection.inverse(z, self.vt)
            
        self.non_linearity = torch.exp(-self.psi(self.vt.T))
        z = (z - self.zeta(self.non_linearity)) / self.non_linearity


        return z
    
    def fit(self, X, epochs=1000, learning_rate=1e-3):
        optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[400], gamma=0.3)
       
        losses = []
        mse = nn.MSELoss()
        epochs = tqdm(range(epochs))

        for _ in epochs:
            _, log_pz, log_jacob = self.forward(X)
            
            # p_Z = MultivariateNormal(torch.zeros(self.h_dim), torch.eye(self.h_dim))
            # z = p_Z.rsample(sample_shape=(X.size(0),))
            # synth_x = self.inverse(z)
            
            # signature = (stack_signatures(compute_path_signature_torch(synth_x, level_threshold=2)[1:]))
            # signature_true = (stack_signatures(compute_path_signature_torch(X, level_threshold=2)[1:]))
            
            loss_likelihood = (-log_pz - log_jacob).mean() 
          
            # loss_sig = mse(signature, signature_true)
            # loss_init_values = torch.mean((X[:, 0] - synth_x[:, 0])**2)
            loss_sig = loss_likelihood
            loss_init_values = loss_likelihood
            
            total_norm = 0.0
            for p in self.parameters():
                param_norm = p.norm(2)
                total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
        
            loss = loss_likelihood + total_norm 
            # + 0.01 * loss_sig + loss_init_values + total_norm
            

            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()

            epochs.set_description(f'Loss: {round(loss.item(), 4)}')
            losses.append([loss_likelihood.item(), loss_init_values.item(), loss_sig.item()])
            
        return np.array(losses)
            
    
    def sample(self, n_samples):
        samples_array = []
            
        if n_samples % self.v_dim == 0:
            for i in range(n_samples // self.v_dim):
                p_Z = MultivariateNormal(torch.zeros(self.h_dim), torch.eye(self.h_dim))
                z = p_Z.rsample(sample_shape=(self.v_dim,))

                X_sample = self.inverse(z).detach().numpy()
                samples_array.append(X_sample)
                
            samples_array = np.vstack(samples_array)
                
        else:
            for i in range(n_samples // self.v_dim + 1):
                p_Z = MultivariateNormal(torch.zeros(self.h_dim), torch.eye(self.h_dim))
                z = p_Z.rsample(sample_shape=(self.v_dim,))

                X_sample = self.inverse(z).detach().numpy()
                samples_array.append(X_sample)
              
            samples_array = np.vstack(samples_array)[:n_samples]

        return samples_array