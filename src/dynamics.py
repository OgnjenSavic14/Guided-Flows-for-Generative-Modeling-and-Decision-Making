import torch

def alpha_t(t):
    return t

def sigma_t(t):
    return 1.0 - t

def dalpha_dt(t):
    return torch.ones_like(t)

def dsigma_dt(t):
    return -torch.ones_like(t)