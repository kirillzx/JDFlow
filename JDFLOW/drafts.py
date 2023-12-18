class NSDE:
    def __init__(self, h_dim, dt, time_steps):
        super(NSDE, self).__init__()
        
        self.h_dim = h_dim
        self.time_steps = time_steps
        self.dt = dt
        
        self.drift = Drift(h_dim)
        self.diffusion = Diffusion(h_dim, 1)
        self.f = Operator_F(h_dim, h_dim)
        
    def forward(self, x):
        xi0 = x
        if xi0.shape[0] == 1:
            xi = sdeint(self.drift, self.diffusion, self.dt, xi0, self.time_steps, self.h_dim, 1)
            xi = xi.view(1, xi.shape[0], xi.shape[1])
            
            z = torch.randn_like(x)
            x0 = sdeint_inverse(self.drift, self.diffusion, self.dt, z, self.time_steps, self.h_dim, 1).detach()
            s_tilde = self.f(x0.view(1, self.h_dim))
            s_tilde = s_tilde.view(1, s_tilde.shape[0], s_tilde.shape[1])
            
            return xi, s_tilde, xi0.view(1, xi0.shape[0], xi0.shape[1])
    
        else:
            xi = sdeint(self.drift, self.diffusion, self.dt, xi0, self.time_steps, self.h_dim, 1)
            
            z = torch.randn_like(x)
            x0 = sdeint_inverse(self.drift, self.diffusion, self.dt, z, self.time_steps, self.h_dim, 1).detach()
            s_tilde = self.f(x0)
            
            return xi, s_tilde, xi0
            
    def inverse(self, z):
        s_tilde = self.f(sdeint_inverse(self.drift, self.diffusion, self.dt, z, self.time_steps, self.h_dim, 1).detach()).detach().cpu()
        
        return s_tilde
        
    
    def fit(self, X, epochs=200):
        optim_drift = optim.Adam(self.drift.parameters(), lr=1e-3)
        optim_diffusion = optim.Adam(self.diffusion.parameters(), lr=1e-3)
        optim_f = optim.Adam(self.f.parameters(), lr=1e-3, betas=(0.9, 0.999), amsgrad=True)

        epochs = tqdm(range(epochs))
        distr = MultivariateNormal(torch.zeros(self.h_dim), torch.eye(self.h_dim))

        if self.h_dim == 1:
            for _ in epochs:
                mse = nn.MSELoss()

                xi, s_tilde, u = self.forward(X)
                loss = - torch.mean(distr.log_prob(xi[:, -1])) + mse(s_tilde[:, 0], u[:, 0])


                optim_drift.zero_grad()
                optim_diffusion.zero_grad()
                optim_f.zero_grad()

                loss.backward()
                
                optim_drift.step()
                optim_diffusion.step()
                optim_f.step()

                epochs.set_description(f'Loss: {round(loss.item(), 4)}')

        else:
            for _ in epochs:
                mse = nn.MSELoss()
                jsd = JSD()

                xi, s_tilde, u = self.forward(X)
                loss = -torch.mean(distr.log_prob(xi[:, -1])) + mse(s_tilde, u)
                # + torch.mean(jsd(s_tilde, u))

                optim_drift.zero_grad()
                optim_diffusion.zero_grad()
                optim_f.zero_grad()

                loss.backward()
                
                optim_drift.step()
                optim_diffusion.step()
                optim_f.step()

                epochs.set_description(f'Loss: {round(loss.item(), 4)}')
    
    
    def sample(self, n_sample):
        p_Z = MultivariateNormal(torch.zeros(self.h_dim), torch.eye(self.h_dim))
        z = p_Z.rsample(sample_shape=(n_sample,))
        
        x0 = self.inverse(z)

        return x0
    
    
class NSDE_VAE(nn.Module):
    def __init__(self, h_dim, time_steps, dt):
        super(NSDE_VAE, self).__init__()
        
        self.h_dim = h_dim
        self.laten_dim = 2**8
        self.time_steps = time_steps
        self.dt = dt
        self.hidden_lp = 2**5
        
        self.drift1 = Drift(self.laten_dim)
        self.drift2 = Drift(self.laten_dim)
        self.diffusion1 = Diffusion(self.laten_dim, 1)
        self.diffusion2 = Diffusion(self.laten_dim, 1)
        self.f = Operator_F(h_dim, h_dim)
        
        self.h_dim_v = 1
        self.z_dim = 30
        self.phi = Phi(self.z_dim, 1, 1)
        self.mu_v = Mu_V(1, self.h_dim_v, h_dim)
        self.sigma_v = Sigma_V(1, self.h_dim_v, h_dim)
        
        

        # self.mu = nn.Sequential(nn.Linear(h_dim_flat, 2**7),
        #                              nn.ReLU(),
        #                              nn.Linear(2**7, self.laten_dim))
        
        # self.log_var = nn.Sequential(nn.Linear(h_dim_flat, 2**7),
        #                              nn.ReLU(),
        #                              nn.Linear(2**7, self.laten_dim))
        
        self.encoder = nn.Sequential(nn.Linear(self.h_dim, 2**6),
                                     nn.Tanh(),
                                     nn.Linear(2**6, 2**6),
                                     nn.Tanh(),
                                     nn.Linear(2**6, self.laten_dim))
        
        # self.fc_d = nn.Linear(self.laten_dim, h_dim_flat)
        
        self.decoder = nn.Sequential(nn.Linear(self.laten_dim, 2**8),
                                     nn.Sigmoid(),
                                     nn.Linear(2**8, 2**8),
                                     nn.Sigmoid(),
                                     nn.Linear(2**8, self.h_dim))

        self.fc_mu = nn.Sequential(nn.LSTM(self.laten_dim, self.laten_dim, num_layers=1, batch_first=True),
                                   extract_tensor(),
                                   nn.Linear(self.laten_dim, 2**7),
                                   nn.ReLU(),
                                   nn.Linear(2**7, self.laten_dim))
        self.fc_logvar = nn.Sequential(nn.LSTM(self.laten_dim, self.laten_dim, num_layers=1, batch_first=True),
                                       extract_tensor(),
                                       nn.Linear(self.laten_dim, 2**7),
                                       nn.ReLU(),
                                       nn.Linear(2**7, self.laten_dim))
        # self.fc_mu = nn.Sequential(nn.Conv1d(10, 128, kernel_size=3, stride=2),
        #                            nn.ReLU(),
        #                            nn.Conv1d(128, 10, kernel_size=3, stride=2))
        # self.fc_logvar = nn.Sequential(nn.Conv1d(10, 128, kernel_size=3, stride=2),
        #                            nn.ReLU(),
        #                            nn.Conv1d(128, 10, kernel_size=3, stride=2))
        self.fc_lp = nn.Linear(self.h_dim, self.hidden_lp)
        
        
        
    def encode(self, x):
        x_lat = self.encoder(x)
        # x_lat = torch.flatten(x_lat, start_dim=0)
        
        if x.size(0) == 1:
            mu_out = sdeint(self.drift1, self.diffusion1, self.dt, x_lat, self.time_steps, self.laten_dim, 1)
            mu_out = mu_out.view(1, mu_out.shape[0], mu_out.shape[1])[:, -1]
            
            log_var_out = sdeint(self.drift2, self.diffusion2, self.dt, x_lat, self.time_steps, self.laten_dim, 1)
            log_var_out = log_var_out.view(1, log_var_out.shape[0], log_var_out.shape[1])[:, -1]
            
        else:
            mu_out = sdeint(self.drift1, self.diffusion1, self.dt, x_lat, self.time_steps, self.laten_dim, 1)
            mu_out = mu_out[:, -1]
            # mu_out = x_lat
            
            log_var_out = sdeint(self.drift2, self.diffusion2, self.dt, x_lat, self.time_steps, self.laten_dim, 1)
            log_var_out = log_var_out[:, -1]
            # log_var_out = x_lat
        
        # mu_out = self.mu(x_lat)
        # log_var_out = self.log_var(x_lat)
        
        log_var_out = self.fc_logvar(log_var_out)
        mu_out = self.fc_mu(mu_out)
        
        std = torch.exp(0.5 * log_var_out)
        eps = torch.randn_like(std)
        return eps * std + mu_out, mu_out, log_var_out
    
    def decode(self, z):
        # z = self.fc_d(z)
        # z = z.view(-1, self.h_dim)
        out = self.decoder(z)
        return out
    
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        out = self.decode(z)
        
        return out, mu, logvar
    
    def sample(self, n_samples, set_A, x0):
        if x0.size(0) == 1:
            p_Z = MultivariateNormal(torch.zeros(self.laten_dim), torch.eye(self.laten_dim))
            z = p_Z.rsample(sample_shape=(n_samples,))
            out_array = []
            
            for i in range(n_samples):
                u0 = self.phi(torch.randn(self.z_dim), x0)
                latent_process = sdeint_V(self.mu_v, self.sigma_v, self.dt, u0, self.h_dim, self.h_dim_v, set_A[0]).view(1, self.h_dim)
    
                out = self.f(torch.cat([self.decode(z[i].view(1, self.laten_dim)), latent_process], dim=1)).detach().cpu().numpy()
                out_array.append(out)

            # out = self.decode(z).detach().cpu().numpy()
            
            return np.array(out_array)
        
        else:
            p_Z = MultivariateNormal(torch.zeros(self.laten_dim), torch.eye(self.laten_dim))
            z = p_Z.rsample(sample_shape=(n_samples,))
            out_array = []
            # u0 = self.phi(torch.randn(size=(n_samples, self.z_dim)), x0)
            
            for i in range(n_samples):
                # latent_process = sdeint_V(self.mu_v, self.sigma_v, self.dt, u0[i], self.h_dim, self.h_dim_v, set_A[i])
                # latent_process_emb = self.fc_lp(latent_process.T).detach()

                # out = self.f(torch.cat([self.decode(z[i].view(1, self.laten_dim)), latent_process_emb], dim=1)).detach().cpu().numpy()
                # out = self.f(self.decode(z[i].view(1, self.laten_dim))).detach().cpu().numpy()
                out = self.decode(z[i].view(1, self.laten_dim)).detach().cpu().numpy()
                out_array.append(out)

            # out = self.decode(z).detach().cpu().numpy()
            
            return np.array(out_array)

    def fit(self, X, set_A, x0, epochs=100):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        optim_drift1 = optim.Adam(self.drift1.parameters(), lr=3e-4)
        optim_diffusion1 = optim.Adam(self.diffusion1.parameters(), lr=3e-4)
        optim_drift2 = optim.Adam(self.drift2.parameters(), lr=3e-4)
        optim_diffusion2 = optim.Adam(self.diffusion2.parameters(), lr=3e-4)
        optim_f = optim.Adam(self.f.parameters(), lr=3e-4)
        
        optim_phi = optim.Adam(self.phi.parameters(), lr=3e-4)
        optim_mu_v = optim.Adam(self.mu_v.parameters(), lr=1e-3)
        optim_sigma_v = optim.Adam(self.sigma_v.parameters(), lr=1e-3)
        
        epochs = tqdm(range(epochs))
        mse = nn.MSELoss()
        mse_v = nn.MSELoss()
        
        loss_array_v = []
        loss_array = []
        logvar_array = []
        
        if X.size(0) == 1:
            for _ in epochs:
                u0 = self.phi(torch.randn(self.z_dim), x0)
                latent_process = sdeint_V(self.mu_v, self.sigma_v, self.dt, u0, self.h_dim, self.h_dim_v, set_A[0]).view(1, self.h_dim)
                
                loss_v = mse_v(latent_process, X[0]) + (u0 - X[0][0])**2
                    
                
                x_rec, mu, logvar = self.forward(X)
                # print(x_rec.size(), latent_process.size())
                x_rec = self.f(torch.cat([x_rec, latent_process], dim=1))
                kl_div = torch.sum(0.5 * torch.sum(-1 - logvar + mu ** 2 + logvar.exp(), dim=1))
                
                loss =  mse(x_rec, X) + kl_div


                optimizer.zero_grad()
                optim_drift1.zero_grad()
                optim_diffusion1.zero_grad()
                optim_drift2.zero_grad()
                optim_diffusion2.zero_grad()
                optim_f.zero_grad()
                optim_phi.zero_grad()
                optim_mu_v.zero_grad()
                optim_sigma_v.zero_grad()
                    
                loss.backward()
                loss_v.backward()
                
                optimizer.step()
                optim_drift1.step()
                optim_diffusion1.step()
                optim_drift2.step()
                optim_diffusion2.step()
                optim_f.step()
                optim_phi.step()
                optim_mu_v.step()
                optim_sigma_v.step()
                
                scheduler.step()

                epochs.set_description(f'Loss: {round(loss.item(), 8)}')
                
                loss_array_v.append(loss_v.item())
                loss_array.append(loss.item())
                logvar_array.append(logvar)
                
        else:
            for _ in epochs:
                # v_array = torch.zeros(size=(X.size(0), self.h_dim, 1))

                # for i in range(X.size(0)):
                #     u0 = self.phi(torch.randn(size=(X.size(0), self.z_dim)), x0)
                #     v_sol = sdeint_V(self.mu_v, self.sigma_v, self.dt, u0[i], self.h_dim, self.h_dim_v, set_A[i])
                #     v_array[i] = v_sol
                    
                # latent_process = v_array.squeeze(2)
             
                
                # loss_v = mse_v(latent_process, X) + 0.1 * torch.mean((u0 - x0)**2)
                
                x_rec, mu, logvar = self.forward(X)
                # latent_process_emb = self.fc_lp(latent_process).detach()
           
                # x_rec = self.f(torch.cat([x_rec, latent_process_emb], dim=1))
                # x_rec = self.f(x_rec)
    
    
                kl_div = torch.mean(torch.sum(0.5 * (-1 - logvar + mu ** 2 + logvar.exp()), dim=1))
                rec_loss = torch.mean((x_rec - X)**2)
                
                loss = rec_loss + 5 * kl_div


                optimizer.zero_grad()
                optim_drift1.zero_grad()
                optim_diffusion1.zero_grad()
                optim_drift2.zero_grad()
                optim_diffusion2.zero_grad()
                optim_f.zero_grad()
                optim_phi.zero_grad()
                optim_mu_v.zero_grad()
                optim_sigma_v.zero_grad()
                    
                # loss_v.backward(retain_graph=False)
                loss.backward()
                
                optimizer.step()
                optim_drift1.step()
                optim_diffusion1.step()
                optim_drift2.step()
                optim_diffusion2.step()
                optim_f.step()
                optim_phi.step()
                optim_mu_v.step()
                optim_sigma_v.step()
                
                scheduler.step()

                epochs.set_description(f'Loss: {round(loss.item(), 8)}')
                
                # loss_array_v.append(loss_v.item())
                loss_array_v.append(1)
                loss_array.append([kl_div.item(), rec_loss.item()])
                logvar_array.append([mu, logvar])
                
        return loss_array, loss_array_v, logvar_array
            
            
# class NSDE_VAE(nn.Module):
#     def __init__(self, h_dim, h_dim_flat):
#         super(NSDE_VAE, self).__init__()
        
#         self.h_dim = h_dim
#         self.laten_dim = 2**9
        
#         self.mu = nn.Sequential(nn.Linear(h_dim_flat, 2**7),
#                                      nn.ReLU(),
#                                      nn.Linear(2**7, self.laten_dim))
        
#         self.log_var = nn.Sequential(nn.Linear(h_dim_flat, 2**7),
#                                      nn.ReLU(),
#                                      nn.Linear(2**7, self.laten_dim))
        
#         self.encoder = nn.Sequential(nn.Linear(self.h_dim, 2**7),
#                                      nn.ReLU(),
#                                      nn.Linear(2**7, 2**7),
#                                      nn.ReLU(),
#                                      nn.Linear(2**7, self.h_dim))
        
#         self.fc_d = nn.Linear(self.laten_dim, h_dim_flat)
        
#         self.decoder = nn.Sequential(nn.Linear(self.h_dim, 2**7),
#                                      nn.ReLU(),
#                                      nn.Linear(2**7, 2**7),
#                                      nn.ReLU(),
#                                      nn.Linear(2**7, self.h_dim))
        
#     def encode(self, x):
#         x_lat = self.encoder(x)
#         x_lat = torch.flatten(x_lat, start_dim=0)
        
#         mu_out = self.mu(x_lat)
#         log_var_out = self.log_var(x_lat)
        
#         std = torch.exp(0.5 * log_var_out)
#         eps = torch.randn_like(std)
#         return eps * std + mu_out, mu_out, log_var_out
    
#     def decode(self, z):
#         z = self.fc_d(z)
#         z = z.view(-1, self.h_dim)
#         out = self.decoder(z)
#         return out
    
#     def forward(self, x):
#         z, mu, logvar = self.encode(x)
#         out = self.decode(z)
        
#         return out, mu, logvar
    
#     def sample(self, n_samples):
#         p_Z = MultivariateNormal(torch.zeros(self.laten_dim), torch.eye(self.laten_dim))
#         z = p_Z.rsample(sample_shape=(n_samples,))

#         out = self.decode(z)
        
#         return out.detach().cpu().numpy()

#     def fit(self, X, epochs=100):
#         optimizer = optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999), amsgrad=True)
#         epochs = tqdm(range(epochs))
#         mse = nn.MSELoss()
        
#         for _ in epochs:
#             x_rec, mu, logvar = self.forward(X)
            
#             kl_div = torch.mean(0.5 * torch.sum(-1 - logvar + mu ** 2 + logvar.exp(), dim=0))
           
#             loss = mse(x_rec, X) + kl_div


#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             epochs.set_description(f'Loss: {round(loss.item(), 4)}')




# class DiffMap(nn.Module):
#     def __init__(self, hidden, in_size, out_size):

#         super(DiffMap, self).__init__()
        
#         self.in_size = in_size
#         self.out_size = out_size
        

#         self.sig_net = nn.Sequential(
#             # nn.LSTM(self.in_size, self.in_size, num_layers=1, batch_first=True),
#             # extract_tensor(),
#             # nn.Sigmoid(),
#             nn.Linear(self.in_size, hidden),
#             nn.Sigmoid(),
#             nn.Linear(hidden, hidden),
#             nn.Sigmoid(),
#             nn.Linear(hidden, self.out_size)
#         )

#         self.mu_net = nn.Sequential(
#             # nn.LSTM(self.in_size, self.in_size, num_layers=1, batch_first=True),
#             # extract_tensor(),
#             # nn.Sigmoid(),
#             nn.Linear(self.in_size, hidden),
#             nn.Sigmoid(),
#             nn.Linear(hidden, hidden),
#             nn.Sigmoid(),
#             nn.Linear(hidden, self.out_size)
#         )

#         base_mu, base_cov = torch.zeros(out_size), torch.eye(out_size)
#         self.base_dist = MultivariateNormal(base_mu, base_cov)

#     def forward(self, x):
#         sig = self.sig_net(x)

#         z = x * torch.exp(sig) + self.mu_net(x)
#         log_pz = self.base_dist.log_prob(z)
#         log_jacob = sig.sum(-1)
        
#         return z, log_pz, log_jacob

#     def inverse(self, z):
#         sig_in = self.sig_net(z)

#         x = (z - self.mu_net(z)) * torch.exp(-sig_in)

#         return x




# class Flow(nn.Module):
#     def __init__(self, hidden, n_flows, in_size, out_size):
#         super(Flow, self).__init__()
        
#         self.in_size = in_size
#         self.out_size = out_size

#         self.bijections = nn.ModuleList(
#             [DiffMap(hidden, self.in_size, self.out_size) for _ in range(n_flows)]
#         )

#     def forward(self, x):
#         log_jacobs = []

#         for bijection in self.bijections:

#             x, log_pz, lj = bijection(x)

#             log_jacobs.append(lj)

#         return x, log_pz, sum(log_jacobs)
#     def inverse(self, z):

#         for bijection in reversed(self.bijections):

#             z = bijection.inverse(z)

#         return z.detach().numpy()
    
#     def fit(self, X, epochs=200, learning_rate=1e-3):
        
#         optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
#         scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.999)

#         losses = []
        
#         epochs = tqdm(range(epochs))

#         for _ in epochs:
#             _, log_pz, log_jacob = self.forward(X)
#             loss = (-log_pz - log_jacob).mean()

#             losses.append(loss.item())

#             optim.zero_grad()
#             loss.backward()
#             optim.step()
#             scheduler.step()

#             epochs.set_description(f'Loss: {round(loss.item(), 4)}')
            
#         return losses
            
            
    
#     def sample(self, n_samples):
#         mu, cov = torch.zeros(self.out_size), torch.eye(self.out_size)
#         p_Z = MultivariateNormal(mu, cov)
#         z = p_Z.rsample(sample_shape=(n_samples,))

#         X_sample = self.inverse(z)

#         return X_sample