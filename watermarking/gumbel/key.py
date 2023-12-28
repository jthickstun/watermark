import torch

def gumbel_key_func(generator,n,vocab_size,eff_vocab_size=None):
    if eff_vocab_size is None:
        eff_vocab_size = vocab_size
        
    pi = torch.arange(eff_vocab_size)
    xi = torch.rand((n,eff_vocab_size), generator=generator)

    return xi,pi