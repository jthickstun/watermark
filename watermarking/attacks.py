import torch

def substitution_attack(tokens,p,vocab_size,distribution=None):
    if distribution is None:
        distribution = lambda x : torch.ones(size=(len(tokens),vocab_size)) / vocab_size
    idx = torch.randperm(len(tokens))[:int(p*len(tokens))]
    
    new_probs = distribution(tokens)
    samples = torch.multinomial(new_probs,1).flatten()
    tokens[idx] = samples[idx]
    
    return tokens

def deletion_attack(tokens,p):
    idx = torch.randperm(len(tokens))[:int(p*len(tokens))]
    
    keep = torch.ones(len(tokens),dtype=torch.bool)
    keep[idx] = False
    tokens = tokens[keep]
        
    return tokens

def insertion_attack(tokens,p,vocab_size,distribution=None):
    if distribution is None:
        distribution = lambda x : torch.ones(size=(len(tokens),vocab_size)) / vocab_size
    idx = torch.randperm(len(tokens))[:int(p*len(tokens))]
    
    new_probs = distribution(tokens)
    samples = torch.multinomial(new_probs,1)
    for i in idx.sort(descending=True).values:
        tokens = torch.cat([tokens[:i],samples[i],tokens[i:]])
        tokens[i] = samples[i]
        
    return tokens