from time import time

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from transformers import MarianMTModel, MarianTokenizer

from datasets import load_dataset

from tqdm import tqdm
from collections import defaultdict
import pickle
import copy

import numpy as np

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

parser.add_argument('--model',default="facebook/opt-1.3b",type=str)
parser.add_argument('--save',default="",type=str)
parser.add_argument('--seed',default=0,type=int)
parser.add_argument('--batch_size',default=1,type=int)

parser.add_argument('--m',default=80,type=int)
parser.add_argument('--k',default=0,type=int)
parser.add_argument('--n',default=256,type=int)
parser.add_argument('--T',default=500,type=int)

parser.add_argument('--prompt_tokens',default=50,type=int)
parser.add_argument('--buffer_tokens',default=20,type=int)
parser.add_argument('--n_runs',default=5000,type=int)
parser.add_argument('--max_seed',default=100000,type=int)
parser.add_argument('--offset', action='store_true')

parser.add_argument('--norm',default=1,type=int)
parser.add_argument('--gamma',default=0.0,type=float)
parser.add_argument('--edit', action='store_true')

parser.add_argument('--deletion',default=0.0,type=float)
parser.add_argument('--insertion',default=0.0,type=float)
parser.add_argument('--substitution',default=0.0,type=float)

parser.add_argument('--kirch_gamma',default=0.25,type=float)
parser.add_argument('--kirch_delta',default=1.0,type=float)

parser.add_argument('--rt_translate', action='store_true')
parser.add_argument('--language',default="french",type=str)

parser.add_argument('--truncate_vocab',default=8,type=int)

args = parser.parse_args()
results['args'] = copy.deepcopy(args)

# fix the random seed for reproducibility
t0 = time()
torch.manual_seed(args.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

vocab_size = model.get_output_embeddings().weight.shape[0]
eff_vocab_size = vocab_size - args.truncate_vocab
print(f'Loaded the model (t = {time()-t0} seconds)')

dataset = load_dataset("c4", "realnewslike", split="train", streaming=True)

def corrupt(tokens):
    tokens = deletion_attack(tokens,args.deletion)
    tokens = insertion_attack(tokens,args.insertion,eff_vocab_size)
    tokens = substitution_attack(tokens,args.substitution,eff_vocab_size)

    return tokens

T = args.T                  # number of prompts/generations
n_batches = int(np.ceil(T / args.batch_size)) # number of batches
prompt_tokens = args.prompt_tokens      # minimum prompt length
new_tokens = args.m     # number of tokens to generate
buffer_tokens = args.buffer_tokens
if args.k == 0: 
    k = args.m # k is the block size (= number of tokens)
else:
    k = args.k     
n = args.n     # watermark key length

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

# this is the "key" for the watermark
# for now each generation gets its own key
seeds = torch.randint(2**32, (T,))

if args.method == "transform":
    generate_watermark = lambda prompt,seed : generate(model,
                                                       prompt,
                                                       vocab_size,
                                                       n,
                                                       new_tokens+buffer_tokens,
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
                                                       new_tokens+buffer_tokens,
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
                                                        max_new_tokens=new_tokens+buffer_tokens,
                                                        min_new_tokens=new_tokens+buffer_tokens,
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

ds_iterator = iter(dataset)

null_results = []
generator = torch.Generator()
pbar = tqdm(total=args.n_runs)
run = 0
while run < args.n_runs:
    example = next(ds_iterator)
    text = example['text']

    tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048-buffer_tokens)[0]
    if len(tokens) < prompt_tokens + new_tokens:
        continue
    tokens = tokens[-new_tokens:]

    seed = torch.randint(high=args.max_seed,size=(1,)).item()
    generator.manual_seed(int(seed))
    null_result = test_stat(tokens=tokens,
                            n=n,
                            k=k,
                            generator=generator,
                            vocab_size=vocab_size,
                            null=True)

    null_results.append(null_result)
    run += 1
    pbar.update(1)

null_results = torch.sort(torch.tensor(null_results)).values
test = lambda tokens,seed : fast_permutation_test(tokens,
                                                  vocab_size,
                                                  n,
                                                  k,
                                                  seed,
                                                  test_stat,
                                                  null_results)


t1 = time()

prompts = []
itm = 0
while itm < T:
    example = next(ds_iterator)
    text = example['text']

    tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=2048-buffer_tokens)[0]
    if len(tokens) < prompt_tokens + new_tokens:
        continue
    prompt = tokens[-(new_tokens+prompt_tokens):-new_tokens]
    prompts.append(prompt)

    itm += 1
    
prompts = torch.vstack(prompts)
results['prompts'] = copy.deepcopy(prompts)

null_samples = []
watermarked_samples = []
for batch in range(n_batches):
    idx = torch.arange(batch * args.batch_size,min(T,(batch + 1) * args.batch_size))

    null_samples.append(generate_rnd(prompts[idx],new_tokens+buffer_tokens,model)[:,prompt_tokens:])
    watermarked_samples.append(generate_watermark(prompts[idx], seeds[idx])[:,prompt_tokens:])
null_samples = torch.vstack(null_samples)
watermarked_samples = torch.vstack(watermarked_samples)

results['watermark']['tokens'] = copy.deepcopy(watermarked_samples)
results['null']['tokens'] = copy.deepcopy(null_samples)

null_samples = torch.clip(null_samples,max=eff_vocab_size-1)
watermarked_samples = torch.clip(watermarked_samples,max=eff_vocab_size-1)

print(f'Generated samples in (t = {time()-t1} seconds)')

pvals_watermark = []
pvals_null = []
pbar = tqdm(total=T)
for itm in range(T):
    null_sample = null_samples[itm]
    null_sample = corrupt(null_sample)
    null_sample = tokenizer.decode(null_sample, skip_special_tokens=True)
    if args.rt_translate:
        null_sample = rt_translate(null_sample)
    null_sample = tokenizer.encode(null_sample,
                                   return_tensors='pt',
                                   truncation=True,
                                   max_length=2048)[0]
    if len(null_sample) < new_tokens + 1:
        null_sample = torch.nn.functional.pad(null_sample,(new_tokens-len(null_sample),0),"constant",0)
    else:
        null_sample = null_sample[1:new_tokens+1]
    pval = test(null_sample, seeds[itm])
    pvals_null.append(pval)

    watermarked_sample = watermarked_samples[itm]
    watermarked_sample = corrupt(watermarked_sample)
    watermarked_sample = tokenizer.decode(watermarked_sample, skip_special_tokens=True)
    if args.rt_translate:
        watermarked_sample = rt_translate(watermarked_sample)
    watermarked_sample = tokenizer.encode(watermarked_sample,
                                          return_tensors='pt',
                                          truncation=True,
                                          max_length=2048)[0]
    if len(watermarked_sample) < new_tokens + 1:
        watermarked_sample = torch.nn.functional.pad(watermarked_sample,(new_tokens-len(watermarked_sample),0),"constant",0)
    else:
        watermarked_sample = watermarked_sample[1:new_tokens+1]
    pval = test(watermarked_sample, seeds[itm])
    pvals_watermark.append(pval)

    pbar.update(1)

pbar.close()
print(f'Ran the experiment (t = {time()-t1} seconds)')

results['watermark']['pvals'] = torch.tensor(pvals_watermark)
results['null']['pvals'] = torch.tensor(pvals_null)

pickle.dump(results,open(args.save,"wb"))
