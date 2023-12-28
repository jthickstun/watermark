import torch

def transform_key_func(generator,n,vocab_size,eff_vocab_size=None):
    pi = torch.randperm(vocab_size, generator=generator)
    xi = torch.rand((n,1), generator=generator)

    return xi,pi