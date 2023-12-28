import json

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from transformers import MarianMTModel, MarianTokenizer

from tqdm import tqdm
from collections import defaultdict
import pickle
import copy

import numpy as np
import scipy

from watermarking.generation import generate, generate_rnd
from watermarking.detection import phi,fast_permutation_test,permutation_test
from watermarking.attacks import substitution_attack,insertion_attack,deletion_attack

from watermarking.transform.score import transform_score,transform_edit_score
from watermarking.transform.sampler import transform_sampling
from watermarking.transform.key import transform_key_func

from watermarking.gumbel.score import gumbel_score,gumbel_edit_score
from watermarking.gumbel.sampler import gumbel_sampling
from watermarking.gumbel.key import gumbel_key_func

from watermarking.kirchenbauer.watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

import argparse

results = defaultdict(dict)

parser = argparse.ArgumentParser(description="Experiment Settings")

parser.add_argument('--method',default="transform",type=str)

parser.add_argument('--model',default="/nlp/scr/jthickstun/sft_v6_52k_llama_7b_regen_v7_3ep_recover",type=str)
parser.add_argument('--save',default="",type=str)
parser.add_argument('--seed',default=0,type=int)

parser.add_argument('--m',default=200,type=int)
parser.add_argument('--n',default=256,type=int)
parser.add_argument('--T',default=0,type=int)

parser.add_argument('--n_runs',default=500,type=int)
parser.add_argument('--max_seed',default=100000,type=int)
parser.add_argument('--offset', action='store_true')

parser.add_argument('--norm',default=1,type=int)
parser.add_argument('--gamma',default=0.0,type=float)
parser.add_argument('--edit', action='store_true')

parser.add_argument('--kirch_gamma',default=0.25,type=float)
parser.add_argument('--kirch_delta',default=1.0,type=float)

parser.add_argument('--rt_translate', action='store_true')
parser.add_argument('--language',default="french",type=str)

args = parser.parse_args()

results['args'] = args

torch.manual_seed(args.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
vocab_size = model.get_output_embeddings().weight.shape[0]
eff_vocab_size = vocab_size

def clip(tokens):
    eos = torch.where(tokens == 2)[0] # find instances of the EOS token
    if len(eos) > 0:
        # truncate after the first EOS token (end of the response)
        tokens = tokens[:eos[0]]

    return tokens

T = args.T
# Upper bound on the number of generated tokens
# If we don't generate a full response to the instruction, oh well
max_new_tokens = args.m
n = args.n    # watermark key sequence length

if args.rt_translate:
    if args.language == "french":
        en_ne_model_name = "Helsinki-NLP/opus-mt-tc-big-en-fr"
        en_ne_tokenizer = MarianTokenizer.from_pretrained(en_ne_model_name)
        en_ne_model = MarianMTModel.from_pretrained(en_ne_model_name).to(device)

        ne_en_model_name = "Helsinki-NLP/opus-mt-tc-big-fr-en"
        ne_en_tokenizer = MarianTokenizer.from_pretrained(ne_en_model_name)
        ne_en_model = MarianMTModel.from_pretrained(ne_en_model_name).to(device)
    elif args.language == "russian":
        en_ne_model_name = "Helsinki-NLP/opus-mt-en-ru"
        en_ne_tokenizer = MarianTokenizer.from_pretrained(en_ne_model_name)
        en_ne_model = MarianMTModel.from_pretrained(en_ne_model_name).to(device)

        ne_en_model_name = "Helsinki-NLP/opus-mt-ru-en"
        ne_en_tokenizer = MarianTokenizer.from_pretrained(ne_en_model_name)
        ne_en_model = MarianMTModel.from_pretrained(ne_en_model_name).to(device)
    else:
        raise

    def rt_translate(text):
        try:
            tokens = en_ne_tokenizer(text.split('. '), return_tensors="pt", padding=True).to(device)
            tokens = en_ne_model.generate(**tokens,max_new_tokens=52)
            french_text = ' '.join([en_ne_tokenizer.decode(t, skip_special_tokens=True) for t in tokens])

            tokens = ne_en_tokenizer(french_text.split('. '), return_tensors="pt", padding=True).to(device)
            tokens = ne_en_model.generate(**tokens,max_new_tokens=512)
            roundtrip_text = ' '.join([ne_en_tokenizer.decode(t, skip_special_tokens=True) for t in tokens])
        except:
            roundtrip_text = ""
        return roundtrip_text

seeds = torch.randint(2**32, (T,))

if args.method == "transform":
    generate_watermark = lambda prompt,seed : generate(model,
                                                       prompt,
                                                       vocab_size,
                                                       n,
                                                       max_new_tokens,
                                                       seed,
                                                       transform_key_func,
                                                       transform_sampling,
                                                       random_offset=args.offset)

    if args.edit is True:
        dist = lambda x,y : transform_edit_score(x,y,gamma=args.gamma)
    else:
        dist = lambda x,y : transform_score(x,y)
    test_stat = lambda tokens,n,k,generator,vocab_size,null=False : phi(tokens=tokens,
                                                                        n=n,
                                                                        k=k,
                                                                        generator=generator,
                                                                        key_func=transform_key_func,
                                                                        vocab_size=vocab_size,
                                                                        dist=dist,
                                                                        null=False,
                                                                        normalize=True)

elif args.method == "gumbel":
    generate_watermark = lambda prompt,seed : generate(model,
                                                       prompt,
                                                       vocab_size,
                                                       n,
                                                       max_new_tokens,
                                                       seed,
                                                       gumbel_key_func,
                                                       gumbel_sampling,
                                                       random_offset=args.offset)

    if args.edit is True:
        dist = lambda x,y : gumbel_edit_score(x,y,gamma=args.gamma)
    else:
        dist = lambda x,y : gumbel_score(x,y)
    test_stat = lambda tokens,n,k,generator,vocab_size,null=False : phi(tokens=tokens,
                                                                        n=n,
                                                                        k=k,
                                                                        generator=generator,
                                                                        key_func=gumbel_key_func,
                                                                        vocab_size=vocab_size,
                                                                        dist=dist,
                                                                        null=null,
                                                                        normalize=False)
                                                                                   
elif args.method == "kirchenbauer":
    watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                               gamma=args.kirch_gamma,
                                               delta=args.kirch_delta,
                                               seeding_scheme="simple_1")

    watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                            gamma=args.kirch_gamma, # should match original setting
                                            seeding_scheme="simple_1", # should match original setting
                                            device=model.device, # must match the original rng device type
                                            tokenizer=tokenizer,
                                            z_threshold=1.5,
                                            normalizers=[],
                                            ignore_repeated_bigrams=False)
    
    generate_watermark = lambda prompt,seed=None : model.generate(
                                                        prompt.to(model.device),
                                                        do_sample=True,
                                                        max_new_tokens=max_new_tokens,
                                                        min_new_tokens=max_new_tokens,
                                                        top_k=0,
                                                        logits_processor=LogitsProcessorList([watermark_processor])).cpu()

    def test_stat_wrapper(tokens):
        try:
            return torch.tensor(-watermark_detector.detect(tokenizer.decode(tokens, skip_special_tokens=True))['z_score'])
        except:
            return torch.tensor(0.0)
    test_stat = lambda tokens,n,k,generator,vocab_size,null=False : test_stat_wrapper(tokens)
else:
    raise

test = lambda tokens,seed : permutation_test(tokens,
                                             vocab_size,
                                             n,
                                             len(tokens),
                                             seed,
                                             test_stat,
                                             n_runs=args.n_runs)

with open("experiments/instructions/alpaca_farm_evaluation.json") as f:
    dataset = json.load(f)

# This is the prompt that the Alpaca team uses
prompt_fmt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"

print('Generating Alpaca responses to instructions...')

results['prompts'] = []
results['watermark']['tokens'] = []
results['null']['tokens'] = []

pvals_watermark = []
pvals_null = []

itm = 0
for example in tqdm(dataset):
    text = example['instruction']
    inp = example['input']
    output = example['output']

    # These are more complicated/structured instructions
    # Yann says we should skip them unless we really need more data
    if example['input'] != "":
        continue

    prompt = prompt_fmt.format(instruction=text)
    prompt = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=2048)[0]
    prompt_tokens = len(prompt)

    results['prompts'].append(copy.deepcopy(prompt))

    null_sample = generate_rnd(prompt.unsqueeze(0),max_new_tokens,model)[0,prompt_tokens:]
    watermarked_sample = generate_watermark(prompt.unsqueeze(0), seeds[itm].unsqueeze(0))[0,prompt_tokens:]

    results['null']['tokens'].append(copy.deepcopy(null_sample))
    results['watermark']['tokens'].append(copy.deepcopy(watermarked_sample))

    null_sample = clip(null_sample)
    watermarked_sample = clip(watermarked_sample)

    null_sample = tokenizer.decode(null_sample, skip_special_tokens=True)
    watermarked_sample = tokenizer.decode(watermarked_sample, skip_special_tokens=True)

    if args.rt_translate:
        null_sample = rt_translate(null_sample)
        watermarked_sample = rt_translate(watermarked_sample)
    
    null_sample = tokenizer.encode(null_sample,
                                   return_tensors='pt',
                                   truncation=True,
                                   max_length=2048)[0]
    watermarked_sample = tokenizer.encode(watermarked_sample,
                                   return_tensors='pt',
                                   truncation=True,
                                   max_length=2048)[0]
    
    if args.method == "kirchenbauer":
        pval_null = scipy.stats.norm.sf(-test_stat_wrapper(null_sample))
        pval_watermark = scipy.stats.norm.sf(-test_stat_wrapper(watermarked_sample))
    else:
        pval_null = test(null_sample, seeds[itm])
        pval_watermark = test(watermarked_sample, seeds[itm])

    pvals_null.append(pval_null)
    pvals_watermark.append(pval_watermark)

    itm += 1
    if args.T > 0 and itm >= args.T:
        break

results['watermark']['pvals'] = torch.tensor(pvals_watermark)
results['null']['pvals'] = torch.tensor(pvals_null)
results['total'] = itm

pickle.dump(results,open(args.save,"wb"))
