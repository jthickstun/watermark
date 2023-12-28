import sys
import numpy as np
import pyximport
pyximport.install(reload_support=True, language_level=sys.version_info[0],
                  setup_args={"include_dirs":np.get_include()})
from watermarking.gumbel.gumbel_levenshtein import gumbel_levenshtein

import torch

def gumbel_score(tokens,xi):
    xi_samp = torch.gather(xi,-1,tokens.unsqueeze(-1)).squeeze()
    return -torch.sum(torch.log(1/(1-xi_samp)))

def gumbel_edit_score(tokens,xi,gamma):
    return gumbel_levenshtein(tokens.numpy(),xi.numpy(),gamma)
