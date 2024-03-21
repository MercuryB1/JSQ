import pdb
import torch
import torch.nn as nn
import torch
from loguru import logger
from copy import deepcopy
import os
import random
import numpy as np
import time
from loguru import logger
import math
from transformers import (
    AutoModelForCausalLM
)
from smoothquant.smooth import smooth_layer
from smoothquant.quantize import quantize_layer
from smoothquant.utils import find_layers, prepare_calibration_input, clip_matrix, generate_ss
from smoothquant.layerwrapper import WrappedGPT
global layer_num
layer_num = 0

def prune_wanda_and_smoothquant_annealing(args, device=torch.device("cuda:0")):
    logger.info(f'test_only is {args.test_only}')
    # model = deepcopy(ori_model)
    
    logger.info(f'rho is {args.rho}')
    logger.info(f'sparsity_ratio is {args.sparsity_ratio}')
    logger.info(f'nbits is {args.nbits}')
    # from transformers import LlamaTokenizer
    # tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)
    # tokenizer.save_pretrained("/mnt/nvme0/wangzining/smooth2/examples/llama-30b")
    # exit(0)
    model = build_model(args)
    # pdb.set_trace()
    CHATGLM = False
    if CHATGLM:
        logger.info('chatglm')
        exit(0)
    if hasattr(model, "transformer"):
        if hasattr(model.transformer, "embedding"):
            CHATGLM = True
            logger.info('chatglm')

    
    # model.eval()
    logger.info(f'sparsity type: {args.sparsity_type}')
    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))
    logger.info(f'set n to {prune_n}, set m to {prune_m}')

    T = 300
    Tmin = 10
    k = 50 
    t = 0
    if CHATGLM:
        layers = model.transformer.encoder.layers
    else: 
        layers = model.model.layers
    # clip_opts = [0.0001, 0.001, 0.01, 0.02, 0.05]
    clip_opts = [0.00000,0.00004,0.00005,0.00006,0.00007]
    clip_table = [2] * len(layers)
    # llama_eval(model, testloader, 'cuda')
    ppl_prev = clip(model, args, clip_opts, clip_table, device, prune_n, prune_m)
    model.save_pretrained(args.saved_path)
    # exit(0)
    ppl_new = ppl_prev
    best_ppl = ppl_new
    if args.test_only:
        logger.info(f'ppl is {best_ppl}')
        model.save_pretrained(args.saved_path)
        exit(0)
    while T >= Tmin:
        for i in range(k):
            random.seed(time.time())
            change_axis = random.randint(0, len(layers)-1)
            new_clip_table = deepcopy(clip_table)
            change_opt = random.randint(0, 4)
            while change_opt == new_clip_table[change_axis]:
                change_opt = random.randint(0, 4)
            new_clip_table[change_axis] = change_opt
            logger.info(f'new table is {new_clip_table}')
            new_model = build_model(args)
            ppl_new = clip(new_model, args, clip_opts, new_clip_table, device, prune_n, prune_m)
            # logger.info(f'new ppl is {ppl_new}')
            if ppl_new - ppl_prev < 0:
                    logger.info(f'ppl_new is {ppl_new}, ppl_prev is {ppl_prev}, ppl_new is better, transfer')
                    clip_table = new_clip_table
                    ppl_prev = ppl_new
            else:
                # metropolis principle
                logger.info(f'ppl_new is {ppl_new}, ppl_prev is {ppl_prev}, ppl_prev is better, do metropolis')
                p = math.exp(-(ppl_new - ppl_prev) * 1500/ T)
                r = np.random.uniform(low=0, high=1)
                if r <= p:
                    logger.info(f'r is {r}, p is {p}, r <= p. transfer')
                    clip_table = new_clip_table
                    ppl_prev = ppl_new
                else:
                    logger.info(f'r is {r}, p is {p}, r > p. do not transfer')
            if best_ppl > ppl_prev:
                best_ppl = ppl_prev
                logger.info(f'best ppl is {best_ppl}, now ppl is {ppl_prev}, store now model.')
                new_model.save_pretrained(args.saved_path)
                # tokenizer.save_pretrained('/mnt/disk1/wzn/smooth2/examples/save')
            else:
                logger.info(f'best ppl is {best_ppl}, now ppl is {ppl_prev}, next model.')

            logger.info(f'now T is {T}, epoch is {i}, now ppl is {ppl_prev}, now clip table is {clip_table}')
        t += 1
        T = T / (1 + t)

    # logger.info(model)
    torch.cuda.empty_cache() 



def clip(model, args, clip_opts, clip_table, device, prune_n = 0, prune_m = 0):
    
    
    logger.info("loading calibdation data")
    dataloader, testenc = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    logger.info("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
    # model.cuda()
    CHATGLM = False

    if hasattr(model, "transformer"):
        if hasattr(model.transformer, "embedding"):
            CHATGLM = True
    if CHATGLM:
        layers = model.transformer.encoder.layers
    else:
        layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        layer_name = f'model.layers.{i}'
        # quantize_layer(layer)

        subset = find_layers(layer)
        # pdb.set_trace()
        
        if '30b' in args.model and "model.layers.{i}" in model.hf_device_map:  
        #handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(
                dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        # logger.info(f"get scales of layer {i}")
        act_scales = {}

        def stat_tensor(name, tensor):
            hidden_dim = tensor.shape[-1]
            tensor = tensor.view(-1, hidden_dim).abs().detach()
            comming_max = torch.max(tensor, dim=0)[0].float().cpu()
            if name in act_scales:
                act_scales[layer_name + '.' + name] = torch.max(act_scales[name], comming_max)
            else:
                act_scales[layer_name + '.' + name] = comming_max

        def add_batch(name):
            def tmp(_, inp, out):
                inp = inp[0].data
                inp = clip_matrix(inp, args.abs, args.clip_l, clip_opts[clip_table[i]])
                stat_tensor(name, inp)
                wrapped_layers[name].add_batch(inp, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                if CHATGLM:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask, position_ids)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        for h in handles:
            h.remove()

        for name in subset:
            # logger.info(f"pruning layer {i} name {name}")
            weight = torch.abs(subset[name].weight.data)
            # pdb.set_trace()
            activation = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            # if name == 'mlp.down_proj':
            #     pt_name = f'layer{i}_activation.pt'
            #     logger.info(f'save {pt_name}')
            #     torch.save(activation, args.saved_path+'/'+pt_name)
            ss = generate_ss(wrapped_layers[name].inp_sum / wrapped_layers[name].inp_num, subset[name].weight.data)
            # logger.info(ss)
            # logger.info(weight * activation)
            W_metric = weight * activation + args.rho * ss
            W_mask = (torch.zeros_like(W_metric) == 1)

            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                # unstructured pruning
                indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

            

            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args.nsamples):
            with torch.no_grad():
                if CHATGLM:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask, position_ids)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        # logger.info(f"smoothing layer {i}")
        smooth_layer(layer_name, layer, act_scales, 0.8)

        # logger.info(f"quantizing layer {i}")
        quantize_layer(layer, nbits=args.nbits)

        inps, outs = outs, inps

    # exit(0)
    logger.info('begin eval')
    ppl = 0
    if CHATGLM:
        # return ppl
        ppl = chatglm_eval(model, testenc, device)
    else:
        # return ppl
        ppl = llama_eval(model, testenc, device)
    logger.info(f'SmoothQuant W8A8 quantized model ppl: {ppl}')
    model.cpu()
    torch.cuda.empty_cache()

    
    return ppl

@torch.no_grad()
def chatglm_eval(model, testenc, dev):
    
    # logger.info('Evaluating ...')
    model.eval()
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.encoder.layers

    # model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.transformer.embedding.word_embeddings = model.transformer.embedding.word_embeddings.to(dev)
    model.transformer.rotary_pos_emb = model.transformer.rotary_pos_emb.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, attention_mask=None, position_ids=None, kv_cache=None, use_cache=True):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = attention_mask
            cache['position_ids'] = position_ids
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            # pdb.set_trace()
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    # model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.transformer.embedding.word_embeddings = model.transformer.embedding.word_embeddings.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        # print(i)
        if '30b' in model.config._name_or_path.lower() and "model.layers.{i}" in model.hf_device_map:  
        #handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
        layer = layers[i].to(dev)
        
        # if args.nearest:
        #     subset = find_layers(layer)
        #     for name in subset:
        #         quantizer = Quantizer()
        #         quantizer.configure(
        #             args.wbits, perchannel=True, sym=False, mse=False
        #         )
        #         W = subset[name].weight.data
        #         quantizer.find_params(W, weight=True)
        #         subset[name].weight.data = quantize(
        #             W, quantizer.scale, quantizer.zero, quantizer.maxq
        #         ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask, position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    # if model.model.norm is not None:
    #     model.model.norm = model.model.norm.to(dev)
    if model.transformer.encoder.final_layernorm is not None:
        model.transformer.encoder.final_layernorm = model.transformer.encoder.final_layernorm.to(dev)
    # model.lm_head = model.lm_head.to(dev)
    model.transformer.output_layer = model.transformer.output_layer.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        # logger.info(f'{i}th sample')
        hidden_states = inps[i].unsqueeze(0)
        # if model.model.norm is not None:
        #     hidden_states = model.model.norm(hidden_states)
        if model.transformer.encoder.final_layernorm is not None:
            hidden_states = model.transformer.encoder.final_layernorm(hidden_states)
        # lm_logits = model.lm_head(hidden_states)
        lm_logits = model.transformer.output_layer(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    # logger.info(ppl.item())

    model.config.use_cache = use_cache
    return ppl



@torch.no_grad()
def llama_eval(model, testenc, dev):
    
    # logger.info('Evaluating ...')
    model.eval()
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            # pdb.set_trace()
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        # print(i)
        if '30b' in model.config._name_or_path.lower() and "model.layers.{i}" in model.hf_device_map:  
        #handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
        layer = layers[i].to(dev)
        
        # if args.nearest:
        #     subset = find_layers(layer)
        #     for name in subset:
        #         quantizer = Quantizer()
        #         quantizer.configure(
        #             args.wbits, perchannel=True, sym=False, mse=False
        #         )
        #         W = subset[name].weight.data
        #         quantizer.find_params(W, weight=True)
        #         subset[name].weight.data = quantize(
        #             W, quantizer.scale, quantizer.zero, quantizer.maxq
        #         ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        # logger.info(f'{i}th sample')
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    # logger.info(ppl.item())

    model.config.use_cache = use_cache
    return ppl



def get_c4(nsamples, seed, seqlen, model):
    from datasets import load_dataset, load_from_disk
    logger.info('load c4 datasets')
    # traindata = load_dataset(
    #     'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    # )
    # valdata = load_dataset(
    #     'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
    # )
    logger.info('load from local')
    traindata = load_from_disk('/mnt/nvme0/wangzining/hf/allenai/c4/allenai--c4/train')
    valdata = load_from_disk('/mnt/nvme0/wangzining/hf/allenai/c4/allenai--c4/validation')
    # traindata = load_dataset(
    #     '/mnt/nvme0/wangzining/huggingface/datasets/c4/en/c4-train.00000-of-01024.json.gz', split='train'
    # )
    # valdata = load_dataset(
    #     '/mnt/nvme0/wangzining/huggingface/datasets/c4/en/c4-validation.00000-of-00008.json.gz', split='validation'
    # )
    from transformers import LlamaTokenizer, AutoTokenizer
    if 'glm' in model:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, use_fast=False)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model, use_fast=False)
    # tokenizer = LlamaTokenizer.from_pretrained('baffo32/decapoda-research-llama-7B-hf', use_fast=False)
    # tokenizer.save_pretrained("/mnt/nvme0/wangzining/smooth2/examples/llama-13b")
    # exit(0)
    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc 

def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForCausalLM
    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
    model.seqlen = 2048
    return model

def build_model(args):
    model_name = args.model
    if 'glm' in model_name:
        kwargs = {'torch_dtype': torch.float16, 'device_map': 'auto', 'trust_remote_code': True}
    else:
        kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.seqlen = args.seqlen
    return model


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model=''
):
    # if 'wikitext2' in name:
    #     return get_wikitext2(nsamples, seed, seqlen, model)
    # if 'ptb' in name:
    #     if 'new' in name:
    #         return get_ptb_new(nsamples, seed, seqlen, model)
    #     return get_ptb(nsamples, seed, seqlen, model)
    if 'c4' in name:
        # if 'new' in name:
        #     return get_c4_new(nsamples, seed, seqlen, model)
        return get_c4(nsamples, seed, seqlen, model)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#             'model', type=str,
#             help='LlaMa model to load; pass location of hugginface converted checkpoint.'
#         )
#     parser.add_argument(
#         'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
#         help='Where to extract calibration data from.'
#     )
#     # parser.add_argument('--model', default="decapoda-research/llama-7b-hf", type=str, help='LLaMA model')
#     parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
#     parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
#     parser.add_argument('--seqlen', type=int, default=2048)
#     parser.add_argument('--sparsity_ratio', type=float, default=0.50, help='Sparsity level')
#     parser.add_argument("--sparsity_type", default="unstructured", type=str, choices=["unstructured", "4:8", "2:4"])
#     parser.add_argument("--prune_method", default="wanda", type=str, choices=["magnitude", "wanda", "sparsegpt"])
#     parser.add_argument("--cache_dir", default="/mnt/disk1/hg/huggingface/cache", type=str)
#     parser.add_argument('--use_variant', action="store_true",
#                         help="whether to use the wanda variant described in the appendix")
#     parser.add_argument('--save', type=str, default=None, help='Path to save results.')
#     parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
#     parser.add_argument('--clip_l', type=float, default=0.0)
#     parser.add_argument('--clip_h', type=float, default=0.001)
#     parser.add_argument('--abs', action="store_false")
#     parser.add_argument('--rho', type=float, default=0.0)
#     args = parser.parse_args()

#     prune_wanda_and_smoothquant_annealing(args)
