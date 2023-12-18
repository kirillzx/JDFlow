from __future__ import absolute_import, division, print_function

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import torch.nn as nn
from tqdm import tqdm


def flip(x, dim):

    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[
        :, getattr(torch.arange(x.size(1) - 1, -1, -1), ("cpu", "cuda")[x.is_cuda])().long(), :
    ]
    return x.view(xsize)


def reconstruct_DFT(x, component="real"):

    if component == "real":
        x_rec = torch.cat([x[0, :], flip(x[0, :], dim=0)[1:]], dim=0)

    elif component == "imag":
        x_rec = torch.cat([x[1, :], -1 * flip(x[1, :], dim=0)[1:]], dim=0)

    return x_rec


class DFT(nn.Module):

    def __init__(self, N_fft=100):

        super(DFT, self).__init__()

        self.N_fft = N_fft
        self.crop_size = int(self.N_fft / 2) + 1
        base_mu, base_cov = torch.zeros(self.crop_size * 2), torch.eye(self.crop_size * 2)
        self.base_dist = MultivariateNormal(base_mu, base_cov)

    def forward(self, x):

        if len(x.shape) == 1:

            x = x.reshape((1, -1))

        x_numpy = x.detach().float()
        X_fft = [np.fft.fftshift(np.fft.fft(x_numpy[k, :])) for k in range(x.shape[0])]
        X_fft_train = np.array(
            [
                np.array(
                    [np.real(X_fft[k])[: self.crop_size] / self.N_fft, np.imag(X_fft[k])[: self.crop_size] / self.N_fft]
                )
                for k in range(len(X_fft))
            ]
        )
        x_fft = torch.from_numpy(X_fft_train).float()

        log_pz = self.base_dist.log_prob(x_fft.view(-1, x_fft.shape[1] * x_fft.shape[2]))
        log_jacob = 0

        return x_fft, log_pz, log_jacob

    def inverse(self, x):


        x_numpy = x.view((-1, 2, self.crop_size))

        x_numpy_r = [
            reconstruct_DFT(x_numpy[u, :, :], component="real").detach().numpy() for u in range(x_numpy.shape[0])
        ]
        x_numpy_i = [
            reconstruct_DFT(x_numpy[u, :, :], component="imag").detach().numpy() for u in range(x_numpy.shape[0])
        ]

        x_ifft = [
            self.N_fft * np.real(np.fft.ifft(np.fft.ifftshift(x_numpy_r[u] + 1j * x_numpy_i[u])))
            for u in range(x_numpy.shape[0])
        ]
        x_ifft_out = torch.from_numpy(np.array(x_ifft)).float()

        return x_ifft_out



class SpectralFilter(nn.Module):
    def __init__(self, d, k, FFT, hidden, flip=False, RNN=False):

        super().__init__()

        self.d, self.k = d, k

        if FFT:

            self.out_size = self.d - self.k + 1
            self.pz_size = self.d + 1
            self.in_size = self.k

        else:

            self.out_size = self.d - self.k
            self.pz_size = self.d
            self.in_size = self.k

        if flip:

            self.in_size, self.out_size = self.out_size, self.in_size

        self.sig_net = nn.Sequential(  # RNN(mode="RNN", HIDDEN_UNITS=20, INPUT_SIZE=1,),
            nn.Linear(self.in_size, hidden),
            nn.Sigmoid(),  # nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),  # nn.Tanh(),
            nn.Linear(hidden, self.out_size),
        )

        self.mu_net = nn.Sequential(  # RNN(mode="RNN", HIDDEN_UNITS=20, INPUT_SIZE=1,),
            nn.Linear(self.in_size, hidden),
            nn.Sigmoid(),  # nn.LeakyReLU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),  # nn.Tanh(),
            nn.Linear(hidden, self.out_size),
        )

        base_mu, base_cov = torch.zeros(self.pz_size), torch.eye(self.pz_size)
        self.base_dist = MultivariateNormal(base_mu, base_cov)

    def forward(self, x, flip=False):
        x1, x2 = x[:, : self.k], x[:, self.k :]

        if flip:

            x2, x1 = x1, x2

        # forward

        sig = self.sig_net(x1).view(-1, self.out_size)
        z1, z2 = x1, x2 * torch.exp(sig) + self.mu_net(x1).view(-1, self.out_size)

        if flip:

            z2, z1 = z1, z2

        z_hat = torch.cat([z1, z2], dim=-1)

        log_pz = self.base_dist.log_prob(z_hat)
        log_jacob = sig.sum(-1)
    

        return z_hat, log_pz, log_jacob

    def inverse(self, Z, flip=False):

        z1, z2 = Z[:, : self.k], Z[:, self.k :]

        if flip:
            z2, z1 = z1, z2

        x1 = z1

        sig_in = self.sig_net(z1).view(-1, self.out_size)
        x2 = (z2 - self.mu_net(z1).view(-1, self.out_size)) * torch.exp(-sig_in)

        if flip:

            x2, x1 = x1, x2

        return torch.cat([x1, x2], -1)


class FourierFlow(nn.Module):
    def __init__(self, hidden, fft_size, n_flows, FFT=True, flip=True, normalize=False):

        super().__init__()

        self.d = fft_size
        self.k = int(fft_size / 2) + 1
        self.fft_size = fft_size
        self.FFT = FFT
        self.normalize = normalize

        if flip:

            self.flips = [True if i % 2 else False for i in range(n_flows)]

        else:

            self.flips = [False for i in range(n_flows)]

        self.bijectors = nn.ModuleList(
            [SpectralFilter(self.d, self.k, self.FFT, hidden=hidden, flip=self.flips[_]) for _ in range(n_flows)]
        )

        self.FourierTransform = DFT(N_fft=self.fft_size)

    def forward(self, x):

        if self.FFT:

            x = self.FourierTransform(x)[0] + 1e-5

            if self.normalize:
                x = (x - self.fft_mean) / self.fft_std
      
            x = x.view(-1, self.d + 1)
          

        log_jacobs = []

        for bijector, f in zip(self.bijectors, self.flips):

            x, log_pz, lj = bijector(x, flip=f)

            log_jacobs.append(lj)

        return x, log_pz, sum(log_jacobs)

    def inverse(self, z):

        for bijector, f in zip(reversed(self.bijectors), reversed(self.flips)):

            z = bijector.inverse(z, flip=f)

        if self.FFT:

            if self.normalize:
                z = z * self.fft_std.view(-1, self.d + 1) + self.fft_mean.view(-1, self.d + 1)

            z = self.FourierTransform.inverse(z)

        return z.detach().numpy()

    def fit(self, X, epochs=500, learning_rate=1e-3, display_step=100):

        X_train = torch.from_numpy(np.array(X)).float()

        # for normalizing the spectral transforms
        X_train_spectral = self.FourierTransform(X_train)[0]
        self.fft_mean = torch.mean(X_train_spectral, dim=0)
        self.fft_std = torch.std(X_train_spectral, dim=0)

        self.d = X_train.shape[1]
        self.k = int(np.floor(X_train.shape[1] / 2))

        optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.999)

        epochs = tqdm(range(epochs))

        for _ in epochs:

            optim.zero_grad()

            _, log_pz, log_jacob = self(X_train)
            loss = (-log_pz - log_jacob).mean()

            loss.backward()
            optim.step()
            scheduler.step()

            epochs.set_description(f'Loss: {loss.item()}')

    def sample(self, n_samples):

        if self.FFT:

            mu, cov = torch.zeros(self.d + 1), torch.eye(self.d + 1)

        else:

            mu, cov = torch.zeros(self.d), torch.eye(self.d)

        p_Z = MultivariateNormal(mu, cov)
        z = p_Z.rsample(sample_shape=(n_samples,))

        X_sample = self.inverse(z)

        return X_sample