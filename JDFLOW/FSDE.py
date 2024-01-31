from FractionalSDE.utils.neural_net import LatentFSDEfunc, LatentODEfunc, GeneratorRNN
from FractionalSDE.utils.neural_net import LatentSDEfunc, latent_dim, batch_dim, nhidden_rnn
from FractionalSDE.utils.utils import RunningAverageMeter, log_normal_pdf, normal_kl, calculate_log_likelihood
from FractionalSDE.utils.plots import plot_generated_paths, plot_original_path, plot_hist
from FractionalSDE.utils.utils import save_csv, tensor_to_numpy
import argparse
from torchdiffeq import odeint
from torchsde import sdeint
from FractionalSDE.utils.fsde_solver import fsdeint
import torch
import numpy as np
from torch import optim


def train_fsde(data, method, epochs):
    batch_dim = 5
    device='cpu'
    train_data = data.to(device) 
    train_ts = torch.tensor(np.arange(len(train_data)))
 
    
    if method == "RNN":
        rnn = GeneratorRNN().to(device)
        params = list(rnn.parameters()) 
    elif method == "SDE":
        func_SDE = LatentSDEfunc().to(device)
        params = list(func_SDE.parameters()) 
    elif method == "fSDE":
        func_fSDE = LatentFSDEfunc().to(device)
        params = (list(func_fSDE.parameters())) 

    optimizer = optim.Adam(params, lr=1e-3)
    
    for itr in range(1, epochs + 1):
        optimizer.zero_grad()
  
        z0 = torch.zeros(batch_dim, latent_dim) + train_data[0, 0]
        
        
        if method == "RNN":
            h = torch.randn(train_data.size(0), batch_dim, nhidden_rnn)
            z = z0
            pred_return = torch.zeros(batch_dim, latent_dim)
            for k in range(train_data.size(0)-1):        
                z, h_out = rnn(z, h[k])
                pred_return = torch.cat((pred_return, h_out), dim=1)
            pred_return = torch.cumsum(pred_return.unsqueeze(-1), dim=1)
            pred_z = torch.zeros(batch_dim, train_data.size(0), latent_dim) + train_data[0, 0] - pred_return
        elif method == "SDE":
            # dimension of sdeint is (t_size, batch_size, latent_size)
            pred_z = sdeint(func_SDE, z0, train_ts) #.permute(1, 0, 2)
        elif method == "fSDE":
            # dimension of fsdeint is (batch_size, t_size, latent_size)
            pred_z = fsdeint(func_fSDE, 0.7, z0, train_ts) #.permute(0, 2, 1)
    

        with torch.autograd.set_detect_anomaly(True):
            loss = - calculate_log_likelihood(pred_z[:,:,0], train_data[:,0])
        
            reg_lambda = 0
            reg = torch.tensor(0.) 
            for param in params:
                reg += torch.norm(param, 1)
            loss += reg_lambda * reg

            loss.backward()
            optimizer.step()
        
        if itr%100==0:
            print("Iter: {}, Log Likelihood: {:.4f}, Regularization: {:.4f}".format(itr, -loss, reg))        
    # print(f'Training complete after {itr} iters.\n')
    
    
    # Generation of sample paths
    with torch.no_grad():
        x0 = torch.zeros(batch_dim, latent_dim) + train_data[0, 0]

        
        if method == 'RNN':
            h = torch.randn(train_data.size(0), batch_dim, nhidden_rnn)
            x = x0
            return_pred = torch.zeros(batch_dim, latent_dim)
            for k in range(train_data.size(0)-1):        
                x, h_out = rnn(x, h[k])
                return_pred = torch.cat((return_pred, h_out), dim=1)
            return_pred = torch.cumsum(return_pred.unsqueeze(-1), dim=1)
            xs_gen = torch.zeros(batch_dim, train_data.size(0), latent_dim) + train_data[0, 0] - return_pred
        elif method == 'SDE':
            xs_gen = sdeint(func_SDE, x0, torch.tensor(np.arange(len(train_data)))) #.permute(1, 0, 2)
        elif method == 'fSDE':
            xs_gen = fsdeint(func_fSDE, 0.7, x0, torch.tensor(np.arange(len(train_data))))
        
        # plot_original_path(data_name, ts_total, data_total)
        # plot_generated_paths(min([args.num_paths, batch_dim]), data_name, method, ts_total, data_total, xs_gen)
        xs_gen_np = tensor_to_numpy(xs_gen[:,:,0]) 
        # save_csv(data_name, method, ts_total_str, data_total.reshape(-1), xs_gen_np)
        # plot_hist(data_name, method, xs_gen_np[0], train_data)
    return xs_gen_np