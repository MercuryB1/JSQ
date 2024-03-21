import copy

import torch
import torch.nn as nn
from smoothquant.fake_quant import W8A8Linear
# from transformers_modules.falcon_7b.modeling_falcon import FalconLinear
# from transformers.models.falcon.modeling_falcon import FalconLinear

def clip_matrix(matrix, abs=True, clip_l=0, clip_h=0, channel=False):
    if clip_l == 0 and clip_h == 0:
        return matrix

    if channel:
        # print("Channel wise clip!")
        matrix_flatten = matrix
        if abs:
            matrix_flatten = torch.abs(matrix)
        max_threshold = None
        min_threshold = None

        if clip_h != 0:
            max_threshold = torch.quantile(matrix_flatten[0].double(), q=1 - clip_h, dim=0)
        clipped_matrix = torch.clamp(matrix, min=-max_threshold, max=max_threshold)
        return clipped_matrix
    else:
        # if weight is not None:
        #     print("A matrix!")
        #     print(matrix.shape)
        #     A_metric = torch.abs(matrix) / torch.sqrt(torch.norm(weight, p=2, dim=0)).reshape((1, -1))
        #     A_metric = torch.clamp(A_metric, -1000, 1000)
        #     A_mask = (torch.zeros_like(A_metric) == 0)
        #     # sort_res = torch.sort(A_metric, dim=-1, stable=True)
        #     # indices = sort_res[1][:, -int(A_metric.shape[1] * clip_h):]
        #     # A_mask.scatter_(1, indices, True)
        #     matrix[A_mask] = 0
        #     print(matrix.shape)
        #     return matrix
        num_elements = matrix.numel()
        if abs:
            matrix_flatten = torch.abs(matrix).flatten()
        else:
            matrix_flatten = matrix.flatten()

        max_threshold = None
        min_threshold = None

        if clip_l != 0:
            low_index = int(clip_l * num_elements)
            min_threshold, _ = torch.topk(matrix_flatten, largest=False, k=low_index)
            min_threshold = min_threshold[-1]
        if clip_h != 0:
            high_index = int(clip_h * num_elements)
            max_threshold, _ = torch.topk(matrix_flatten, largest=True, k=high_index)
            max_threshold = max_threshold[-1]

        if abs:
            clipped_matrix = torch.clamp(matrix, -max_threshold, max_threshold)
        else:
            clipped_matrix = torch.clamp(matrix, min_threshold, max_threshold)

        return clipped_matrix


def find_layers(module, layers=[nn.Linear, W8A8Linear], name=''):
    if type(module) in layers or "FalconLinear" in module.__class__.__name__:
        return {name: module}
    else:
        pass
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def check_sparsity(model):
    CHATGLM = False
    Falcon = False
    if hasattr(model, "transformer"):
        if hasattr(model.transformer, "embedding"):
            CHATGLM = True
        elif hasattr(model.transformer, "word_embeddings"):
            Falcon = True
    if CHATGLM:
        layers = model.transformer.encoder.layers
    elif Falcon:
        layers = model.transformer.h
    else:
        layers = model.model.layers

    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count) / sub_params:.6f}")

    return float(count) / total_params


def prepare_calibration_input(model, dataloader, device):
    CHATGLM = False
    Falcon = False
    if hasattr(model, "transformer"):
        if hasattr(model.transformer, "embedding"):
            CHATGLM = True
        elif hasattr(model.transformer, "word_embeddings"):
            Falcon = True
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if CHATGLM:
        layers = model.transformer.encoder.layers
    elif Falcon:
        layers = model.transformer.h
    else:
        layers = model.model.layers

    # if Falcon:
    #     return

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device) # (128, 2048, 4096)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, *args, **kwargs):
            # print(kwargs)
            if CHATGLM:
                inps[cache['i']] = inp.transpose(0, 1)[0]
            else:
                inps[cache['i']] = inp
            cache['i'] += 1
            if CHATGLM:
                cache['attention_mask'] = args[0]
            else:
                cache['attention_mask'] = kwargs['attention_mask']
            if CHATGLM:
                cache['position_ids'] = args[1]
            elif Falcon:
                pass
            else:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            # if cache['i']==0:
            #     print(model.prepare_inputs_for_generation(batch[0].to(device)))
            #     cache['position_ids'] = model.prepare_inputs_for_generation(batch[0].to(device))['position_ids']
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    model.config.use_cache = use_cache
    return inps, outs, attention_mask, position_ids


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask == True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def cal_mse_layer(args, layer1, layer2, inps, attention_mask, position_ids, outs1=None, outs2=None):
    if outs1 is None:
        layer1_outs = []
        for i in range(args.nsamples):
            with torch.no_grad():
                layer1_outs.append(
                    layer1(inps[i].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0][0])
    else:
        layer1_outs = outs1

    if outs2 is None:
        layer2_outs = []
        for i in range(args.nsamples):
            with torch.no_grad():
                layer2_outs.append(
                    layer2(inps[i].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0][0])
    else:
        layer2_outs = outs2

    mse = 0.0
    print(layer1_outs[0].shape)
    for i in range(args.nsamples):
        device = layer2_outs[i].device
        mse += torch.nn.functional.mse_loss(layer1_outs[i].to(device), layer2_outs[i]).item()

    return mse


def generate_ss(activation, weight):
    cin, cout = weight.shape
    ss = torch.zeros_like(weight)
    for i in range(cout):
        # print(weight.shape)
        w = copy.deepcopy(weight)
        w[:, i] = 0
        out = activation @ (w.t())
        max_values, _ = torch.max(out, dim=0)
        min_values, _ = torch.min(out, dim=0)
        row_ss = (max_values - min_values)
        ss[:, i] = row_ss
    ss = torch.where(torch.isinf(ss), torch.tensor(100), ss)
    return ss

def generate_ss2(activation, weight):
    out = activation @ (weight.t())
    max_values, _ = torch.max(out, dim=0)
    min_values, _ = torch.min(out, dim=0)
    row_ss = (max_values - min_values).reshape((-1, 1))
    # print(weight.shape)
    # print(row_ss.shape)
    # ss = torch.where(torch.isinf(ss), torch.tensor(100), ss)
    return row_ss
