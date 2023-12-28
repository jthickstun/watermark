import torch
import scipy
import numpy as np

def permutation_test(tokens,vocab_size,n,k,seed,test_stat,n_runs=100,max_seed=100000):
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    test_result = test_stat(tokens=tokens,
                            n=n,
                            k=k,
                            generator=generator,
                            vocab_size=vocab_size)
    p_val = 0
    for run in range(n_runs):
        pi = torch.randperm(vocab_size)
        tokens = torch.argsort(pi)[tokens]
        
        seed = torch.randint(high=max_seed,size=(1,)).item()
        generator.manual_seed(int(seed))
        null_result = test_stat(tokens=tokens,
                                n=n,
                                k=k,
                                generator=generator,
                                vocab_size=vocab_size,
                                null=True)
        # assuming lower test values indicate presence of watermark
        p_val += (null_result <= test_result).float() / n_runs
    
    return p_val

def fast_permutation_test(tokens,vocab_size,n,k,seed,test_stat,null_results):
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    test_result = test_stat(tokens=tokens,
                            n=n,
                            k=k,
                            generator=generator,
                            vocab_size=vocab_size)
    p_val = torch.searchsorted(null_results,test_result,right=True) / len(null_results)
    return p_val

def phi(tokens,n,k,generator,key_func,vocab_size,dist,null=False,normalize=False):
    if null:
        tokens = torch.unique(tokens, return_inverse=True,sorted=False)[1]
        eff_vocab_size = torch.max(tokens) + 1
    else:
        eff_vocab_size = vocab_size

    xi,pi = key_func(generator,n,vocab_size,eff_vocab_size)
    tokens = torch.argsort(pi)[tokens]
    if normalize:
        tokens = tokens.float() / vocab_size
    
    A = adjacency(tokens,xi,dist,k)
    closest = torch.min(A,axis=1)[0]

    return torch.min(closest)

def adjacency(tokens,xi,dist,k):
    m = len(tokens)
    n = len(xi)

    A = torch.empty(size=(m-(k-1),n))
    for i in range(m-(k-1)):
        for j in range(n):
            A[i][j] = dist(tokens[i:i+k],xi[(j+torch.arange(k))%n])

    return A
