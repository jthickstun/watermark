import torch

def gumbel_sampling(probs,pi,xi):
    return torch.argmax(xi ** (1/torch.gather(probs, 1, pi)),axis=1).unsqueeze(-1)
