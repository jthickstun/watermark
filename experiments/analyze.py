import torch
import torch.nn.functional as F

import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from transformers import MarianMTModel, MarianTokenizer

from collections import defaultdict
import pickle
import copy

import argparse

parser = argparse.ArgumentParser(description="Experiment Settings")
parser.add_argument('--file',default="",type=str)
args = parser.parse_args()

results = pickle.load(open(args.file,"rb"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# you can change the 1.3b descriptor to use other OPT models
# see the list here: https://huggingface.co/models?other=opt
tokenizer = AutoTokenizer.from_pretrained(results['args'].model)
model = AutoModelForCausalLM.from_pretrained(results['args'].model).to(device)

print("First example prompt:")
print(tokenizer.decode(results['prompts'][0], skip_special_tokens=True))

print("Watermarked completion to first prompt:")
print(tokenizer.decode(results['watermark']['tokens'][0], skip_special_tokens=True))
print("Regular completion:")
print(tokenizer.decode(results['null']['tokens'][0], skip_special_tokens=True))

print("Median p-value of watermarked completions for all prompts:")
print(results['watermark']['pvals'])
print("Median p-value of regular completions for all prompts:")
print(results['null']['pvals'])