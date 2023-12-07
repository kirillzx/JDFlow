import numpy as np
import torch
import itertools

def compute_path_signature_torch(X, a=0, b=1, level_threshold=3):
    N = len(X)
    t = np.linspace(a, b, len(X[0]))
    dt = t[1] - t[0]
    X_t = [Xi for Xi in X]

    t = t[:-1]
    dX_t = [torch.diff(Xi_t) for Xi_t in X_t]
    X_prime_t = [dXi_t / dt for dXi_t in dX_t]
    
    signature = [[torch.ones(len(t))]]
    
    for k in range(level_threshold):
        previous_level = signature[-1]
        current_level = []
        for previous_level_integral in previous_level:
            for i in range(N):
                current_level.append(torch.cumsum(torch.FloatTensor(previous_level_integral * dX_t[i]), dim=0))
        signature.append(current_level)

    signature_terms = [list(itertools.product(*([np.arange(1, N+1).tolist()] * i)))
                       for i in range(0, level_threshold+1)]
    
    return signature

def signature_set(signature_traj) -> list:
    signature_f = []
    for i in signature_traj:
        for j in i:
            signature_f.append(j[-1])
            
    return signature_f


def compute_path_signature(X, a=0, b=1, level_threshold=3):
    N = len(X)
    t = np.linspace(a, b, len(X[0]))
    dt = t[1] - t[0]
    X_t = [Xi for Xi in X]

    t = t[:-1]
    dX_t = [np.diff(Xi_t) for Xi_t in X_t]
    X_prime_t = [dXi_t / dt for dXi_t in dX_t]
    
    signature = [[np.ones(len(t))]]
    
    for k in range(level_threshold):
        previous_level = signature[-1]
        current_level = []
        for previous_level_integral in previous_level:
            for i in range(N):
                current_level.append(np.cumsum(previous_level_integral * dX_t[i]))
        signature.append(current_level)

    signature_terms = [list(itertools.product(*([np.arange(1, N+1).tolist()] * i)))
                       for i in range(0, level_threshold+1)]
    
    return t, X_t, X_prime_t, signature, signature_terms



def signature_set(signature_traj) -> list:
    signature_f = []
    for i in signature_traj:
        for j in i:
            signature_f.append(j[-1])
            
    return signature_f


def stack_signatures(signatures: list):
    all_sig = []
    for sig in signatures:
        all_sig.append(torch.stack(sig))
        
    return torch.vstack(all_sig)