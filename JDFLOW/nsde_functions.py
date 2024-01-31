import torch
import torch.nn as nn
import torch.nn.functional as F

class Phi(nn.Module):
    def __init__(self, z_dim, h_dim, data_dim):
        super(Phi, self).__init__()
        
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.data_dim = data_dim
        
        self.model = nn.Sequential(nn.Linear(self.z_dim, 2**7),
                                   nn.Sigmoid(),
                                   nn.Linear(2**7, 2**7),
                                   nn.Sigmoid(),
                                   nn.Linear(2**7, self.h_dim))
        
        self.linear = nn.Sequential(nn.Linear(self.data_dim, self.h_dim),
                                    nn.Sigmoid(),
                                    nn.Linear(self.h_dim, self.h_dim))
        
    def forward(self, x, x0):
        out = self.model(x)
        out_x0 = self.linear(x0)
        
        return out + out_x0


class Drift(nn.Module):
    def __init__(self, h_dim):
        super(Drift, self).__init__()

        self.h_dim = h_dim

        self.model = nn.Sequential(nn.Linear(self.h_dim, 2**6),
                                   nn.Sigmoid(),
                                #    nn.LayerNorm(2**7),
                                   nn.Linear(2**6, 2**7),
                                   nn.Sigmoid(),
                                #    nn.LayerNorm(2**7),
                                   nn.Linear(2**7, 2**6))
        
        self.t_nn = nn.Sequential(nn.Linear(3, 2**5),
                                  nn.Sigmoid(),
                                #   nn.LayerNorm(2**6),
                                  nn.Linear(2**5, 2**6))
        
        self.proj = nn.Linear(2**7, self.h_dim)

    def forward(self, t, x):
        out_x = self.model(x)
        out_t = self.t_nn(t)
        out = torch.cat([out_t.view(-1, 2**6), out_x], axis=1)
        out = self.proj(out)
        
        return out
    
class Diffusion(nn.Module):
    def __init__(self, h_dim, w_dim):
        super(Diffusion, self).__init__()

        self.h_dim = h_dim
        self.w_dim = w_dim

        self.model = nn.Sequential(nn.Linear(self.h_dim, 2**6),
                                   nn.Sigmoid(),
                                #    nn.LayerNorm(2**7),
                                   nn.Linear(2**6, 2**7),
                                   nn.Sigmoid(),
                                #    nn.LayerNorm(2**7),
                                   nn.Linear(2**7, 2**6))
        
        self.t_nn = nn.Sequential(nn.Linear(3, 2**5),
                                  nn.Sigmoid(),
                                #   nn.LayerNorm(2**6),
                                  nn.Linear(2**5, 2**6))
        
        self.proj = nn.Linear(2**7, self.h_dim * self.w_dim)

    def forward(self, t, x):
        out_x = self.model(x)
        out_t = self.t_nn(t)
        out = torch.cat([out_t.view(-1, 2**6), out_x], axis=1)
        out = self.proj(out)
        
        if self.w_dim == 1:
            return out
        
        else:
            return out.view(self.h_dim, self.w_dim)
    
class Jump(nn.Module):
    def __init__(self, h_dim):
        super(Jump, self).__init__()

        self.h_dim = h_dim

        self.model = nn.Sequential(nn.Linear(self.h_dim, 2**6),
                                   nn.Sigmoid(),
                                #    nn.LayerNorm(2**7),
                                   nn.Linear(2**6, 2**7),
                                   nn.Sigmoid(),
                                #    nn.LayerNorm(2**7),
                                   nn.Linear(2**7, 2**6))
        
        self.t_nn = nn.Sequential(nn.Linear(3, 2**5),
                                  nn.Sigmoid(),
                                #   nn.LayerNorm(2**6),
                                  nn.Linear(2**5, 2**6))
        
        self.proj = nn.Linear(2**7, self.h_dim)

    def forward(self, t, x):
        out_x = self.model(x)
        out_t = self.t_nn(t)
        out = torch.cat([out_t.view(-1, 2**6), out_x], axis=1)
        out = self.proj(out)
        
        return out  
    
class Mu_V(nn.Module):
    def __init__(self, h_dim, h_dim_v, data_dim):
        super(Mu_V, self).__init__()

        self.h_dim = h_dim
        self.h_dim_v = h_dim_v
        self.data_dim = data_dim

        self.model = nn.Sequential(nn.Linear(self.h_dim, 2**7),
                                   nn.Tanh(),
                                #    nn.LayerNorm(2**7),
                                   nn.Linear(2**7, 2**7),
                                   nn.Tanh(),
                                #    nn.LayerNorm(2**7),
                                   nn.Linear(2**7, 2**6))
        
        self.t_nn = nn.Sequential(nn.Linear(3, 2**6),
                                  nn.Tanh(),
                                #   nn.LayerNorm(2**6),
                                  nn.Linear(2**6, 2**6))
        
        self.a_nn = nn.Sequential(nn.Linear(self.data_dim, 2**6),
                                  nn.Tanh(),
                                #   nn.LayerNorm(2**6),
                                  nn.Linear(2**6, 2**6))
        
        self.proj = nn.Linear(3 * 2**6, self.h_dim_v)

    def forward(self, t, x, a):
        out_x = self.model(x)
        out_t = self.t_nn(torch.cat([t, torch.sin(t), torch.cos(t)], axis=0).view(1, -1))
        out_a = self.a_nn(a)
        out = torch.cat([out_t.view(1, 2**6), out_x.view(1, -1), out_a.view(1, 2**6)], axis=1)
        out = self.proj(out)
        
        return out
    
class Sigma_V(nn.Module):
    def __init__(self, h_dim, h_dim_v, data_dim):
        super(Sigma_V, self).__init__()

        self.h_dim = h_dim
        self.h_dim_v = h_dim_v
        self.data_dim = data_dim

        self.model = nn.Sequential(nn.Linear(self.h_dim, 2**7),
                                   nn.Tanh(),
                                #    nn.LayerNorm(2**7),
                                   nn.Linear(2**7, 2**7),
                                   nn.Tanh(),
                                #    nn.LayerNorm(2**7),
                                   nn.Linear(2**7, 2**6))
        
        self.t_nn = nn.Sequential(nn.Linear(3, 2**6),
                                  nn.Tanh(),
                                #   nn.LayerNorm(2**6),
                                  nn.Linear(2**6, 2**6))
        
        self.a_nn = nn.Sequential(nn.Linear(self.data_dim, 2**6),
                                  nn.Tanh(),
                                #   nn.LayerNorm(2**6),
                                  nn.Linear(2**6, 2**6))
        
        self.proj = nn.Linear(3 * 2**6, self.h_dim_v)

    def forward(self, t, x, a):
        out_x = self.model(x)
        out_t = self.t_nn(torch.cat([t, torch.sin(t), torch.cos(t)], axis=0).view(1, -1))
        out_a = self.a_nn(a)
        out = torch.cat([out_t.view(1, 2**6), out_x.view(1, -1), out_a.view(1, 2**6)], axis=1)
        out = self.proj(out)
        
        return out
    
class Operator_F(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Operator_F, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.model = nn.Sequential(nn.Linear(self.in_dim, 2**7),
                                   nn.Sigmoid(),
                                   nn.Linear(2**7, 2**7),
                                   nn.Sigmoid(),
                                   nn.Linear(2**7, self.out_dim),
                                   nn.Sigmoid())
        
        
        # self.rnn = nn.LSTM(h_dim, h_dim, 5, bidirectional=False)

    def forward(self, x):
        # out, _ = self.rnn(x)
        out = self.model(x)
        return out
    
class Operator_xi(nn.Module):
    def __init__(self, h_dim, data_dim):
        super(Operator_F, self).__init__()

        self.h_dim = h_dim
        self.data_dim = data_dim

        self.model = nn.Sequential(nn.Linear(self.h_dim, 2**7),
                                   nn.Sigmoid(),
                                   nn.Linear(2**7, 2**7),
                                   nn.Sigmoid(),
                                   nn.Linear(2**7, 2**6))
        
        self.t_nn = nn.Sequential(nn.Linear(1, 2**5),
                                   nn.Sigmoid(),
                                   nn.Linear(2**5, 2**6),
                                   nn.Sigmoid(),
                                   nn.Linear(2**6, 2**6))
        
        # self.rnn = nn.LSTM(h_dim, h_dim, 5, bidirectional=False)
        self.proj = nn.Linear(2 * 2**6, self.h_dim_v)

    def forward(self, x, t):
        out_x = self.model(x)
        out_t = self.t_nn(torch.cat([t, torch.sin(t), torch.cos(t)], axis=0).view(1, -1))
        out = torch.cat([out_t.view(1, 2**6), out_x.view(1, -1)], axis=1)
        out = self.proj(out)
        
        return out
    
class Operator_G(nn.Module):
    def __init__(self, h_dim, data_dim):
        super(Operator_G, self).__init__()

        self.h_dim = h_dim
        self.data_dim = data_dim

        self.model = nn.Sequential(nn.Linear(self.h_dim, 2**7),
                                   nn.ReLU(),
                                   nn.Linear(2**7, 2**7),
                                   nn.ReLU(),
                                   nn.Linear(2**7, self.data_dim))

    def forward(self, x):
        out = self.model(x)
        return out
    
    
class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = F.softmax(p.view(-1, p.size(-1))), F.softmax(q.view(-1, q.size(-1)))
        m = (0.5 * (p + q)).log()
        
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))
    
    
def calc_windows_eq_size(data, size=10):
    k = len(data)//size
    windows = []

    for i in range(size):
        windows.append(data[i*k : (i+1)*k])
        
    windows = np.mean(np.array(windows), 1)
    
    return torch.FloatTensor(windows)