import sys
import numpy as np
import pyximport
pyximport.install(reload_support=True, language_level=sys.version_info[0],
                  setup_args={"include_dirs":np.get_include()})
from watermarking.transform.transform_levenshtein import transform_levenshtein

import torch

def transform_score(tokens,xi):
    return torch.pow(torch.linalg.norm(tokens-xi.squeeze(),ord=1),1)

def transform_edit_score(tokens,xi,gamma=1):
    return transform_levenshtein(tokens.numpy(),xi.squeeze().numpy(),gamma)
