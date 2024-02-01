import matplotlib.pyplot as plt
from JDFLOW.signature_computation import *
from JDFLOW.evaluate.metrics import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import FastICA

def plotCollection(ax, ys, *args, **kwargs):

  ax.plot(ys, *args, **kwargs)

  if "label" in kwargs.keys():

    #remove duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
      if label not in newLabels:
        newLabels.append(label)
        newHandles.append(handle)

    plt.legend(newHandles, newLabels)
    
    
def plot_signature(data_tensor, samples, samples_ff, samples_PAR, samples_fsde, save, name):

  fig, axs = plt.subplots(2, 2, figsize=(10, 5), dpi=100)

  signature_traj = compute_path_signature_torch(torch.FloatTensor(samples), level_threshold=3)
  signature = signature_set(signature_traj)[1:]

  signature_traj = compute_path_signature_torch(data_tensor, level_threshold=3)
  signature_real = signature_set(signature_traj)[1:]

  signature_traj = compute_path_signature_torch(torch.FloatTensor(samples_ff), level_threshold=3)
  signature_ff = signature_set(signature_traj)[1:]

  signature_traj = compute_path_signature_torch(torch.FloatTensor(samples_PAR), level_threshold=3)
  signature_PAR = signature_set(signature_traj)[1:]

  signature_traj = compute_path_signature_torch(torch.FloatTensor(samples_fsde), level_threshold=3)
  signature_fsde = signature_set(signature_traj)[1:]



  axs[0, 0].hist(signature_real, bins=np.arange(-2, 2, 0.1), label='Real', color='black', alpha=0.7, density=True)
  axs[0, 0].hist(signature, bins=np.arange(-2, 2, 0.1), label='JDFlow', color='tab:red', alpha=0.7, density=True)
  axs[0, 0].set_xlabel('Values')
  axs[0, 0].set_ylabel('Frequencies')
  axs[0, 0].legend()


  axs[0, 1].hist(signature_real, bins=np.arange(-2, 2, 0.1), label='Real', color='black', alpha=0.7, density=True)
  axs[0, 1].hist(signature_ff, bins=np.arange(-2, 2, 0.1), label='Fourier Flow', color='tab:blue', alpha=0.7, density=True)
  axs[0, 1].set_xlabel('Values')
  # axs[0, 1].set_ylabel('Frequencies')
  axs[0, 1].legend()


  axs[1, 0].hist(signature_real, bins=np.arange(-2, 2, 0.1), label='Real', color='black', alpha=0.7, density=True)
  axs[1, 0].hist(signature_PAR, bins=np.arange(-2, 2, 0.1), label='PAR', color='tab:orange', alpha=0.7, density=True)
  axs[1, 0].set_xlabel('Values')
  axs[1, 0].set_ylabel('Frequencies')
  axs[1, 0].legend()


  axs[1, 1].hist(signature_real, bins=np.arange(-2, 2, 0.1), label='Real', color='black', alpha=0.7, density=True)
  axs[1, 1].hist(signature_fsde, bins=np.arange(-2, 2, 0.1), label='fSDE', color='tab:green', alpha=0.7, density=True)
  axs[1, 1].set_xlabel('Values')
  # axs[0, 0].set_ylabel('Frequencies')
  axs[1, 1].legend()

  plt.subplots_adjust(wspace=0.3)
  plt.tight_layout(pad=0.5)
  
  if save:
    plt.savefig(f'{name}.pdf', dpi=300)
    
  plt.show()
  
  
  

def plot_qq_extrema(real_data, synth_data, synth_ff, synth_PAR, synth_fsde, save, name):
  fig, axs = plt.subplots(2, 2, figsize=(10, 5), dpi=100)

  axs[0, 0].plot(extr_quant_computation(real_data), extr_quant_computation(real_data), color='black', label='Real')
  axs[0, 0].scatter(extr_quant_computation(synth_data), extr_quant_computation(real_data), color='tab:red', label='JDFlow')
  axs[0, 0].set_xlabel('Real')
  axs[0, 0].set_ylabel('Synth')
  axs[0, 0].legend()


  axs[0, 1].plot(extr_quant_computation(real_data), extr_quant_computation(real_data), color='black', label='Real')
  axs[0, 1].scatter(extr_quant_computation(synth_ff), extr_quant_computation(real_data), color='tab:blue', label='Fourier Flow')
  axs[0, 1].set_xlabel('Real')
  # axs[0, 1].set_ylabel('Synth')
  axs[0, 1].legend()


  axs[1, 0].plot(extr_quant_computation(real_data), extr_quant_computation(real_data), color='black', label='Real')
  axs[1, 0].scatter(extr_quant_computation(synth_PAR), extr_quant_computation(real_data), color='tab:orange', label='PAR')
  axs[1, 0].set_xlabel('Real')
  axs[1, 0].set_ylabel('Synth')
  axs[1, 0].legend()


  axs[1, 1].plot(extr_quant_computation(real_data), extr_quant_computation(real_data), color='black', label='Real')
  axs[1, 1].scatter(extr_quant_computation(synth_fsde), extr_quant_computation(real_data), color='tab:green', label='fSDE')
  axs[1, 1].set_xlabel('Real')
  # axs[1, 1].set_ylabel('Synth')
  axs[1, 1].legend()


  plt.tight_layout(pad=1)
  plt.subplots_adjust(wspace=0.1)
  
  if save:
    plt.savefig(f'{name}.pdf', dpi=300) 
    
  plt.show()
  
  
  
def plot_autocorr(real_data, synth_data, synth_ff, synth_PAR, synth_fsde, save, name):
    fig, axs = plt.subplots(2, 2, figsize=(10, 5), dpi=100)

    axs[0, 0].scatter(np.arange(0, len(data[0]))[::50], autocorr_vec(real_data)[::50], marker='o', label = 'Real', color='black', s=18)
    axs[0, 0].plot(autocorr_vec(synth_data), label = 'JDFlow', color='tab:red')
    axs[0, 0].set_xlabel('Lag')
    axs[0, 0].set_ylabel('Correlation')
    axs[0, 0].legend()


    axs[0, 1].scatter(np.arange(0, len(data[0]))[::50], autocorr_vec(real_data)[::50], marker='o', label = 'Real', color='black', s=18)
    axs[0, 1].plot(autocorr_vec(synth_ff), label = 'Fourier Flow', color='tab:blue')
    axs[0, 1].set_xlabel('Lag')
    axs[0, 1].set_ylabel('Correlation')
    axs[0, 1].legend()

    axs[1, 0].scatter(np.arange(0, len(data[0]))[::50], autocorr_vec(real_data)[::50], marker='o', label = 'Real', color='black', s=18)
    axs[1, 0].plot(autocorr_vec(synth_PAR), label = 'PAR', color='tab:orange')
    axs[1, 0].set_xlabel('Lag')
    axs[1, 0].set_ylabel('Correlation')
    axs[1, 0].legend()

    axs[1, 1].scatter(np.arange(0, len(data[0]))[::50], autocorr_vec(real_data)[::50], marker='o', label = 'Real', color='black', s=18)
    axs[1, 1].plot(autocorr_vec(synth_fsde), label = 'fSDE', color='tab:green')
    axs[1, 1].set_xlabel('Lag')
    axs[1, 1].set_ylabel('Correlation')
    axs[1, 1].legend()


    plt.xticks()
    plt.yticks()
    plt.tight_layout(pad=0.5)
    
    if save:
        plt.savefig(f'{name}.pdf', dpi=300)
    plt.show()
    
    
def plot_tsne(real_data, synth_data, synth_ff, synth_PAR, synth_fsde, save, name):    
    pca = TSNE(n_components=2)
    pca_real = pca.fit_transform(real_data.T)
    pca_jdflow = pca.fit_transform(synth_data.T)
    pca_PAR = pca.fit_transform(synth_PAR.T)
    pca_FF = pca.fit_transform(synth_ff.T)
    pca_fsde = pca.fit_transform(synth_fsde.T)


    fig, axs = plt.subplots(2, 2, figsize=(10, 5), dpi=100)

    axs[0, 0].scatter(pca_real[:, 0], pca_real[:, 1], marker='o', label = 'Real', color='black', s=18, alpha=0.6)
    axs[0, 0].scatter(pca_jdflow[:, 0], pca_jdflow[:, 1], label = 'JDFlow', color='tab:red', alpha=0.6)
    axs[0, 0].set_xlabel('$X_1$')
    axs[0, 0].set_ylabel('$X_2$')
    axs[0, 0].legend()


    axs[0, 1].scatter(pca_real[:, 0], pca_real[:, 1], marker='o', label = 'Real', color='black', s=18, alpha=0.6)
    axs[0, 1].scatter(pca_FF[:, 0], pca_FF[:, 1], label = 'Fourier Flow', color='tab:blue', alpha=0.6)
    axs[0, 1].set_xlabel('$X_1$')
    axs[0, 1].set_ylabel('$X_2$')
    axs[0, 1].legend()

    axs[1, 0].scatter(pca_real[:, 0], pca_real[:, 1], marker='o', label = 'Real', color='black', s=18, alpha=0.6)
    axs[1, 0].scatter(pca_PAR[:, 0], pca_PAR[:, 1], label = 'PAR', color='tab:orange', alpha=0.6)
    axs[1, 0].set_xlabel('$X_1$')
    axs[1, 0].set_ylabel('$X_2$')
    axs[1, 0].legend()

    axs[1, 1].scatter(pca_real[:, 0], pca_real[:, 1], marker='o', label = 'Real', color='black', s=18, alpha=0.6)
    axs[1, 1].scatter(pca_fsde[:, 0], pca_fsde[:, 1], label = 'fSDE', color='tab:green', alpha=0.6)
    axs[1, 1].set_xlabel('$X_1$')
    axs[1, 1].set_ylabel('$X_2$')
    axs[1, 1].legend()


    plt.xticks()
    plt.yticks()
    plt.tight_layout(pad=0.5)
    
    if save:
        plt.savefig(f'{name}.pdf', dpi=300)
        
    plt.show()
    
    
    
def plot_signature_w1_dist(data_tensor, samples, samples_ff, samples_PAR, samples_fsde, save, name):
    signature_traj = compute_path_signature_torch(data_tensor, level_threshold=3)
    signature_real = signature_set_all(signature_traj, data_tensor.size(1) - 1)

    signature_traj = compute_path_signature_torch(torch.FloatTensor(samples), level_threshold=3)
    signature = signature_set_all(signature_traj, torch.FloatTensor(samples).size(1) - 1)

    signature_traj = compute_path_signature_torch(torch.FloatTensor(samples_ff), level_threshold=3)
    signature_ff = signature_set_all(signature_traj, torch.FloatTensor(samples_ff).size(1) - 1)

    signature_traj = compute_path_signature_torch(torch.FloatTensor(samples_PAR), level_threshold=3)
    signature_PAR = signature_set_all(signature_traj, torch.FloatTensor(samples_PAR).size(1) - 1)

    signature_traj = compute_path_signature_torch(torch.FloatTensor(samples_fsde), level_threshold=3)
    signature_fsde = signature_set_all(signature_traj, torch.FloatTensor(samples_fsde).size(1) - 1)
    
    
    
    w1_dist_sig_jdflow = []
    w1_dist_sig_ff = []
    w1_dist_sig_par = []
    w1_dist_sig_fsde = []

    for i in range(len(signature_real)):
        w1_dist_sig_jdflow.append(wasserstein_distance(signature_real[i], signature[i]))
        w1_dist_sig_ff.append(wasserstein_distance(signature_real[i], signature_ff[i]))
        w1_dist_sig_par.append(wasserstein_distance(signature_real[i], signature_PAR[i]))
        w1_dist_sig_fsde.append(wasserstein_distance(signature_real[i], signature_fsde[i]))
        
        
    plt.subplots(figsize=(10, 5), dpi=100)

    plt.plot(np.arange(len(w1_dist_sig_jdflow)), [0]*len(w1_dist_sig_jdflow), color='black')
    plt.plot(w1_dist_sig_jdflow, label='JDFlow', color='tab:red', alpha=0.95)
    plt.plot(w1_dist_sig_ff, label='Fourier Flow', color='tab:blue', alpha=0.95)
    plt.plot(w1_dist_sig_par, label='PAR', color='tab:orange', alpha=0.95)
    plt.plot(w1_dist_sig_fsde, label='fSDE', color='tab:green', alpha=0.95)

    plt.xlabel('Signature cross section')
    plt.ylabel('$W_1$-dist')
    plt.legend()
    plt.tight_layout(pad=0.5)
    
    if save:
        plt.savefig(f'{name}.pdf', dpi=300)
    
    plt.show()