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
        hidden_v = 2**7
        self.d = h_dim // 2
        self.permute = permute
        
        self.in_size = self.d
        self.out_size = self.h_dim - self.d

        if self.permute:
            self.in_size, self.out_size = self.out_size, self.in_size

        
        self.fc_mu = nn.Sequential(
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
            nn.Linear(hidden, self.out_size),
        )
        self.nu = nn.Sequential(
            nn.Linear(M, hidden_v),
            nn.Sigmoid(),
            nn.Linear(hidden_v, hidden_v),
            nn.Sigmoid(),
            nn.Linear(hidden_v, M)
        )
   
        self.p = nn.Sequential(
            nn.Linear(self.time_steps, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, self.in_size)
        )
        self.tanh = nn.Tanh()
        
        self.f = nn.Sequential(
            # nn.LSTM(self.time_steps, self.time_steps, num_layers=1, batch_first=True),
            # extract_tensor(),
            # nn.Sigmoid(),
            nn.Linear(self.in_size, hidden_v),
            nn.Sigmoid(),
            nn.Linear(hidden_v, self.out_size))
        
        self.base_dist = MultivariateNormal(torch.zeros(h_dim), torch.eye(h_dim))

    def forward(self, x, vt):
        x1, x2 = x[:, :self.d], x[:, self.d:]
    
        if self.permute:
            x2, x1 = x1, x2
        
        p_hat = self.tanh(self.p(vt.T))
        z1 = x1 + p_hat
        sig = self.fc_sig(z1)
        
        z2 = self.fc_mu(z1) * self.f(self.nu(z1.T).T) + x2 * torch.exp(sig)
        
        # * self.f(self.nu(z1.T).T)
        # self.nu(vt)
       
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
            
        x2 = ( z2 - self.fc_mu(z1) * self.f(self.nu(z1.T).T) ) * torch.exp(-self.fc_sig(z1))
        
        # * self.f(self.nu(z1.T).T)
        # * self.f(self.nu(vt).T)
        # * self.f(self.nu(z1.T).T)
        # * self.f(vt.T)

      
        p_hat = self.tanh(self.p(vt.T))
        x1 = z1 - p_hat
        
        if self.permute:
            x2, x1 = x1, x2
        
        x = torch.cat([x1, x2], axis=1)

        return x



class JDFlow(nn.Module):
    def __init__(self, n_flows, h_dim, M, time_steps, dt, xiP0):
        super(JDFlow, self).__init__()
        
        self.h_dim = h_dim
        self.permute = [True if i % 2 else False for i in range(n_flows)]

        self.bijections = nn.ModuleList(
            [DiffMap(time_steps, h_dim, M, dt, self.permute[i]) for i in range(n_flows)]
        )
        
        self.xiP = xiP0
        
        self.time_steps = time_steps
        self.dt = dt
        self.M = M
        self.psi = Operator_F(self.time_steps, self.h_dim)
        self.zeta = Operator_F(self.h_dim, self.h_dim)
        
        self.drift = Drift(self.M)
        self.diffusion = Diffusion(self.M, 1)
        self.jump = Jump(self.M)
        self.phi = Phi(self.M, M, M)
        

    def forward(self, x):
        log_jacobs = []
        
        wt = torch.randn((1, self.M))
        v0 = self.phi(wt, x[:, 0].view(1, x.size(0)))
        self.vt = sdeint_jump(self.drift, self.diffusion, self.jump, self.dt, v0, self.time_steps, self.M, self.xiP)
        
        # self.vt, dt, n = choose_data('Merton', n=self.time_steps - 1, M=self.M, s0=2, xiP=self.xiP, muj=0.1, sigmaj=0.2, r=0.04, sigmad=1, T=1)
        # self.vt = torch.FloatTensor(self.vt).T
        
        self.non_linearity = torch.exp(-self.psi(self.vt.T))
        x = self.non_linearity * x + self.zeta(self.non_linearity)
      

        for bijection in self.bijections:
            x, log_pz, lj = bijection(x, self.vt)

            log_jacobs.append(lj)

        return x, log_pz, sum(log_jacobs), self.vt

    def inverse(self, z):
        wt = torch.randn((1, self.M))
        v0 = self.phi(wt, z[:, 0].view(1, z.size(0)))
        self.vt = sdeint_jump(self.drift, self.diffusion, self.jump, self.dt, v0, self.time_steps, self.M, self.xiP) 
        
        # self.vt2, dt, n = choose_data('Merton', n=self.time_steps - 1, M=self.M, s0=data_save[:, 0].mean().numpy(), xiP=self.xiP, muj=0.1, sigmaj=0.2, r=0.04, sigmad=0.8, T=1)
        # self.vt2 = torch.FloatTensor(self.vt2).T
        # self.vt2 = torch.rand_like(self.vt) 
        
        # self.vt = 0.5*(self.vt + self.vt2)

        for bijection in reversed(self.bijections):
            z = bijection.inverse(z, self.vt)
            
        self.non_linearity = torch.exp(-self.psi(self.vt.T))
        z = (z - self.zeta(self.non_linearity)) / self.non_linearity

        return z
    
    def fit(self, X, epochs=1000, learning_rate=1e-3):
        optim = torch.optim.Adam(self.bijections.parameters(), lr=learning_rate)
        optim_psi = torch.optim.Adam(self.psi.parameters(), lr=learning_rate)
        optim_zeta = torch.optim.Adam(self.zeta.parameters(), lr=learning_rate)
        optim_phi = torch.optim.Adam(self.phi.parameters(), lr=learning_rate)
        
        optim_drift = torch.optim.Adam(self.drift.parameters(), lr=3e-4)
        optim_diff = torch.optim.Adam(self.diffusion.parameters(), lr=3e-4)
        optim_jump = torch.optim.Adam(self.jump.parameters(), lr=3e-4)
        
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[400, 900], gamma=0.3)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)
       
        losses = []
        epochs = tqdm(range(epochs))
 
        for _ in epochs:
                _, log_pz, log_jacob, vt = self.forward(X)
                
                # p_Z = MultivariateNormal(torch.zeros(self.time_steps-1), torch.eye(self.time_steps-1))
                # z = p_Z.rsample(sample_shape=(self.time_steps,))
                # synth_x = self.inverse(z)
                
                # signature = (stack_signatures(compute_path_signature_torch(synth_x, level_threshold=2)[1:]))
                # signature_true = (stack_signatures(compute_path_signature_torch(X, level_threshold=2)[1:]))
                
                loss_likelihood = (-log_pz - log_jacob).mean() 
            
                # loss_sig = mse(signature, signature_true)
                # loss_init_values = torch.mean((X[:, 0] - synth_x[:, 0])**2)
                loss_sig = loss_likelihood
                loss_init_values = loss_likelihood
                
                total_norm = 0.0
                for p in self.bijections.parameters():
                    param_norm = p.norm(2)
                    total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
            
                
                loss = loss_likelihood + total_norm
                # + 0.01 * loss_sig + loss_init_values + total_norm
                

                optim.zero_grad()
                optim_drift.zero_grad()
                optim_diff.zero_grad()
                optim_jump.zero_grad()
                optim_psi.zero_grad()
                optim_zeta.zero_grad()
                optim_phi.zero_grad()
                
                loss.backward()
                
                optim.step()
                optim_drift.step()
                optim_diff.step()
                optim_jump.step()
                optim_psi.step()
                optim_zeta.step()
                optim_phi.step()
                
                scheduler.step()

                epochs.set_description(f'Loss: {round(loss.item(), 4)}')
                losses.append([loss_likelihood.item(), loss_init_values.item(), loss_sig.item()])
            
        return np.array(losses)
            
    
    def sample(self, n_samples):
        samples_array = []
        p_Z = MultivariateNormal(torch.zeros(self.h_dim), torch.eye(self.h_dim))
            
        if n_samples % self.M == 0:
            for i in range(n_samples // self.M):
                z = p_Z.rsample(sample_shape=(self.M,))

                X_sample = self.inverse(z).detach().numpy()
                samples_array.append(X_sample)
                
            samples_array = np.vstack(samples_array)
                
        else:
            for i in range(n_samples // self.M + 1):
                z = p_Z.rsample(sample_shape=(self.M,))

                X_sample = self.inverse(z).detach().numpy()
                samples_array.append(X_sample)
              
            samples_array = np.vstack(samples_array)[:n_samples]

        return samples_array    
    
    
    
    
    
    
    
# class DiffMap(nn.Module):
#     def __init__(self, time_steps, h_dim, M, dt, permute):
#         super(DiffMap, self).__init__()
    
#         self.h_dim = h_dim
#         self.laten_dim = M
#         self.time_steps = time_steps
#         self.dt = dt

#         hidden = 2**7
#         self.v_dim = M
#         self.d = h_dim // 2
#         self.r = M // 2
#         self.permute = permute
        
#         self.in_size = self.d
#         self.out_size = self.h_dim - self.d

    
#         if self.permute:
#             self.in_size, self.out_size = self.out_size, self.in_size

        
#         self.fc_mu = nn.Sequential(
#             # nn.LSTM(self.in_size, self.in_size, num_layers=1, batch_first=True),
#             # extract_tensor(),
#             # nn.Sigmoid(),
#             nn.Linear(self.in_size, hidden),
#             nn.Sigmoid(),
#             nn.Linear(hidden, hidden),
#             nn.Sigmoid(),
#             nn.Linear(hidden, self.out_size)
#         )
#         self.fc_sig = nn.Sequential(
#             nn.LSTM(self.in_size, self.in_size, num_layers=1, batch_first=True),
#             extract_tensor(),
#             nn.Sigmoid(),
#             nn.Linear(self.in_size, hidden),
#             nn.Sigmoid(),
#             nn.Linear(hidden, hidden),
#             nn.Sigmoid(),
#             nn.Linear(hidden, self.out_size)
#         )
#         self.nu = nn.Sequential(
#             nn.Linear(M, hidden),
#             nn.Sigmoid(),
#             nn.Linear(hidden, hidden),
#             nn.Sigmoid(),
#             nn.Linear(hidden, M)
#         )
   
#         self.p = nn.Sequential(
#             nn.Linear(self.time_steps, hidden),
#             nn.Sigmoid(),
#             nn.Linear(hidden, hidden),
#             nn.Sigmoid(),
#             nn.Linear(hidden, self.in_size)
#         )
#         self.tanh = nn.Tanh()
        
#         self.f = nn.Sequential(nn.Linear(self.in_size, hidden),
#                                nn.Sigmoid(),
#                                nn.Linear(hidden, self.out_size))
        
#         self.base_dist = MultivariateNormal(torch.zeros(h_dim), torch.eye(h_dim))

#     def forward(self, x, vt):
#         x1, x2 = x[:, :self.d], x[:, self.d:]
    
#         if self.permute:
#             x2, x1 = x1, x2
        
#         p_hat = self.tanh(self.p(vt.T))
#         z1 = x1 + p_hat
#         sig = self.fc_sig(z1)
#         z2 = self.fc_mu(z1) * self.f(self.nu(z1.T).T) + x2 * torch.exp(sig)
        
#         if self.permute:
#             z2, z1 = z1, z2 
 
#         z = torch.cat([z1, z2], axis=1)
        
#         log_pz = self.base_dist.log_prob(z)
#         log_jacob = sig.sum(-1)

#         return z, log_pz, log_jacob
    
        

#     def inverse(self, z, vt):
#         z1, z2 = z[:, :self.d], z[:, self.d:]
        
#         if self.permute:
#             z2, z1 = z1, z2
            
#         x2 = (z2 - self.fc_mu(z1) * self.f(self.nu(z1.T).T)) * torch.exp(-self.fc_sig(z1))
        
#         p_hat = self.tanh(self.p(vt.T))
#         x1 = z1 - p_hat
      
#         if self.permute:
#             x2, x1 = x1, x2
        
#         x = torch.cat([x1, x2], axis=1)

#         return x



# class JDFlow(nn.Module):
#     def __init__(self, n_flows, h_dim, M, time_steps, dt, sig_dim, xiP0):
#         super(JDFlow, self).__init__()
        
#         self.h_dim = h_dim
#         self.permute = [True if i % 2 else False for i in range(n_flows)]

#         self.bijections = nn.ModuleList(
#             [DiffMap(time_steps, h_dim, M, dt, self.permute[i]) for i in range(n_flows)]
#         )
        
#         self.xiP = xiP0
        
#         self.time_steps = time_steps
#         self.dt = dt
#         self.v_dim = M
#         self.psi = Operator_F(self.time_steps, self.h_dim)
#         self.zeta = Operator_F(self.h_dim, self.h_dim)
        
#         self.drift = Drift(self.v_dim)
#         self.diffusion = Diffusion(self.v_dim, 1)
#         self.jump = Jump(self.v_dim)
#         self.phi = Phi(self.v_dim, M, M)
#         self.sigmoid_sig = nn.Sigmoid()

#     def forward(self, x):
#         log_jacobs = []
        
#         wt = torch.randn((1, self.v_dim))
#         v0 = self.phi(wt, x[:, 0].view(1, x.size(0)))
#         self.vt = sdeint_jump(self.drift, self.diffusion, self.jump, self.dt, v0, self.time_steps, self.v_dim, self.xiP)
        
#         self.non_linearity = torch.exp(-self.psi(self.vt.T))
#         x = self.non_linearity * x + self.zeta(self.non_linearity)
        

#         for bijection in self.bijections:
#             x, log_pz, lj = bijection(x, self.vt)

#             log_jacobs.append(lj)

#         return x, log_pz, sum(log_jacobs)

#     def inverse(self, z):
        
#         wt = torch.randn((1, self.v_dim))
#         v0 = self.phi(wt, z[:, 0].view(1, z.size(0)))
#         self.vt = sdeint_jump(self.drift, self.diffusion, self.jump, self.dt, v0, self.time_steps, self.v_dim, self.xiP)

#         for bijection in reversed(self.bijections):
#             z = bijection.inverse(z, self.vt)
            
#         self.non_linearity = torch.exp(-self.psi(self.vt.T))
#         z = (z - self.zeta(self.non_linearity)) / self.non_linearity


#         return z
    
#     def fit(self, X, epochs=1000, learning_rate=1e-3):
#         optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
#         scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[400, 900], gamma=0.3)
       
#         losses = []
#         mse = nn.MSELoss()
#         epochs = tqdm(range(epochs))

#         for _ in epochs:
#             _, log_pz, log_jacob = self.forward(X)
            
#             loss_likelihood = (-log_pz - log_jacob).mean() 
          
#             # loss_sig = mse(signature, signature_true)
#             # loss_init_values = torch.mean((X[:, 0] - synth_x[:, 0])**2)
#             loss_sig = loss_likelihood
#             loss_init_values = loss_likelihood
            
#             total_norm = 0.0
#             for p in self.parameters():
#                 param_norm = p.norm(2)
#                 total_norm += param_norm.item() ** 2
#                 total_norm = total_norm ** (1. / 2)
        
#             loss = loss_likelihood + total_norm 
#             # + 0.01 * loss_sig + loss_init_values + total_norm
            

#             optim.zero_grad()
#             loss.backward()
#             optim.step()
#             scheduler.step()

#             epochs.set_description(f'Loss: {round(loss.item(), 4)}')
#             losses.append([loss_likelihood.item(), loss_init_values.item(), loss_sig.item()])
            
#         return np.array(losses)
            
    
#     def sample(self, n_samples):
#         samples_array = []
            
#         if n_samples % self.v_dim == 0:
#             for i in range(n_samples // self.v_dim):
#                 p_Z = MultivariateNormal(torch.zeros(self.h_dim), torch.eye(self.h_dim))
#                 z = p_Z.rsample(sample_shape=(self.v_dim,))

#                 X_sample = self.inverse(z).detach().numpy()
#                 samples_array.append(X_sample)
                
#             samples_array = np.vstack(samples_array)
                
#         else:
#             for i in range(n_samples // self.v_dim + 1):
#                 p_Z = MultivariateNormal(torch.zeros(self.h_dim), torch.eye(self.h_dim))
#                 z = p_Z.rsample(sample_shape=(self.v_dim,))

#                 X_sample = self.inverse(z).detach().numpy()
#                 samples_array.append(X_sample)
              
#             samples_array = np.vstack(samples_array)[:n_samples]

#         return samples_array