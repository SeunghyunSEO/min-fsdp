'''
https://github.com/karpathy/nanoGPT/blob/master/model.py
'''

import os
import math
import random
import numpy as np
from copy import deepcopy
from typing import List, Dict
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch_profiler_utils import ContextManagers, get_torch_profiler


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_dist():
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    print(f"rank: {rank}, world size: {world_size}")
    return rank, world_size

def print_message_with_master_process(rank, message):
    if rank==0:
        print(message)

class Attention(nn.Module):
    def __init__(self, hidden, nhead, bias=False):
        super(Attention, self).__init__()
        assert hidden % nhead == 0, "hidden size should be divisible by nhead"
        self.dhead = hidden // nhead
        self.q_proj = nn.Linear(hidden, hidden, bias=bias)
        self.k_proj = nn.Linear(hidden, hidden, bias=bias)
        self.v_proj = nn.Linear(hidden, hidden, bias=bias)
        self.o_proj = nn.Linear(hidden, hidden, bias=bias)

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, -1, self.dhead).transpose(1, 2).contiguous() # B, nhead, T, dhead
        k = self.k_proj(x).view(B, T, -1, self.dhead).transpose(1, 2).contiguous() # B, nhead, T, dhead
        v = self.v_proj(x).view(B, T, -1, self.dhead).transpose(1, 2).contiguous() # B, nhead, T, dhead
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        x = x.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(x)

class MLP(nn.Module):
    def __init__(self, hidden, bias=False):
        super(MLP, self).__init__()
        self.ffn1 = nn.Linear(hidden, 4*hidden, bias)
        self.act = nn.GELU()
        self.ffn2 = nn.Linear(4*hidden, hidden, bias)

    def forward(self, x):
        return self.ffn2(self.act(self.ffn1(x)))

class LayerNorm(nn.Module):
    def __init__(self, hidden, bias=False):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.bias = nn.Parameter(torch.zeros(hidden)) if bias else None

    def forward(self, x):
        return F.layer_norm(x.float(), self.weight.shape, self.weight, self.bias, 1e-5).type_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, hidden, nhead, bias=False):
        super(ResidualBlock, self).__init__()
        self.ln1 = LayerNorm(hidden, bias)
        self.attn = Attention(hidden, nhead, bias)
        self.ln2 = LayerNorm(hidden, bias)
        self.mlp = MLP(hidden, bias)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

class Transformer(nn.Module):
    def __init__(self, vocab_size, block_size, hidden, nhead, nlayer, bias=False):
        super(Transformer, self).__init__()
        assert bias == False, "currently bias is not supported"
        self.vocab_size = vocab_size
        self.nhead = nhead
        self.model = nn.ModuleDict(
            dict(
                wte = nn.Embedding(vocab_size, hidden),
                wpe = nn.Embedding(block_size, hidden),
                h = nn.ModuleList([ResidualBlock(hidden, nhead, bias) for _ in range(nlayer)]),
                ln = LayerNorm(hidden, bias=bias),
            )
        )
        self.lm_head = nn.Linear(hidden, vocab_size, bias=bias)
        self.model.wte.weight = self.lm_head.weight

    def compute_loss(self, z, y):
        z = z[..., :-1, :].contiguous()
        y = y[..., 1:].contiguous()
        return F.cross_entropy(z.view(-1, self.vocab_size), y.view(-1))

    def forward(self, x): 
        y = x
        B, T = x.size()

        pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        x = self.model.wte(x) + self.model.wpe(pos)
        for block in self.model.h:
            x = block(x)
        x = self.model.ln(x)

        z = self.lm_head(x).float() # upcasted logit
        return compute_loss(z, y)

class g(torch.autograd.Function):
    def forward(ctx, x):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x
    def backward(ctx, gradient):
        return gradient

class f(torch.autograd.Function):
    def forward(ctx, x):
        return x
    def backward(ctx, gradient):
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        return gradient

def rowwise_forward(self, x):
    bias = self.bias if self.bias else None
    x = F.linear(x, self.weight, bias)
    return g.apply(x)

def colwise_forward(self, x):
    bias = self.bias if self.bias else None
    x = f.apply(x)
    return F.linear(x, self.weight, bias)

class LossParallel(torch.autograd.Function):
    '''
    it's not exactly same vocab parallel from megatron

    https://github.com/NVIDIA/Megatron-LM/blob/e8f8e63f13a074f7e35d72c8bfb3e1168cd84e8e/megatron/core/models/common/embeddings/language_model_embedding.py#L48
    https://github.com/NVIDIA/Megatron-LM/blob/e8f8e63f13a074f7e35d72c8bfb3e1168cd84e8e/megatron/core/models/gpt/gpt_model.py#L124
    https://github.com/NVIDIA/Megatron-LM/blob/e8f8e63f13a074f7e35d72c8bfb3e1168cd84e8e/megatron/core/models/gpt/gpt_model.py#L213-L237
    https://github.com/NVIDIA/Megatron-LM/blob/e8f8e63f13a074f7e35d72c8bfb3e1168cd84e8e/megatron/core/tensor_parallel/layers.py#L649
    '''
    def forward(ctx, x):
        raise NotImplementedError
    def backward(ctx, grad_output):
        raise NotImplementedError

def parallelize_module(
    model: nn.Module, 
    world_size: int, 
    rank: int, 
):
    assert world_size > 1, "need at least two devices for TP"
    colwise_list = ['q_proj', 'k_proj', 'v_proj', 'ffn1']
    rowwise_list = ['o_proj', 'ffn2']

    for name, module in model.named_children():
        if isinstance(module, nn.Module):
            parallelize_module(module, world_size, rank)

        for _ in rowwise_list:
            if _ in name.lower():
                assert module.weight.size(1) % world_size == 0 
                chunk_size = module.weight.size(1)//world_size
                module.weight.data = module.weight.data[:, chunk_size*rank: chunk_size*(rank+1)].contiguous()
                module.forward = rowwise_forward.__get__(module)
        for _ in colwise_list:
            if _ in name.lower():
                assert module.weight.size(0) % world_size == 0
                chunk_size = module.weight.size(0)//world_size
                module.weight.data = module.weight.data[chunk_size*rank: chunk_size*(rank+1), :].contiguous()
                module.forward = colwise_forward.__get__(module)

def main(args):
    rank, world_size = init_dist()
    device = f"cuda:{rank}"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    vocab_size = len(tokenizer)
    block_size = tokenizer.model_max_length
    hidden, nhead, nlayer = args.hidden, 8, 2

    set_seed()
    model = Transformer(vocab_size, block_size, hidden, nhead, nlayer).to(device).train()
    if args.TP:
        assert model.nhead % world_size == 0, "nhead should be divisible by TP degree"
        parallelize_module(model, world_size, rank)
    lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)

    sent = "i love tensor parallelism."
    input_ids = tokenizer(sent, return_tensors='pt').to(device)

    if args.use_torch_profiler:
        num_wait_steps, num_warmup_steps, num_active_steps, num_repeat = 1, 2, 3, 1
        num_iter = int((num_wait_steps + num_warmup_steps + num_active_steps)*num_repeat)
        context = [
            get_torch_profiler(
                num_wait_steps=num_wait_steps,
                num_warmup_steps=num_warmup_steps,
                num_active_steps=num_active_steps,
                num_repeat=num_repeat,
                save_dir_name=f'TP_{args.TP}_world_size_{world_size}_hidden_{hidden}'
            )
        ]
    else:
        num_iter = 5
        context = []

    with ContextManagers(context) as p:
        for iter in range(num_iter):
            loss = model(input_ids['input_ids'])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            ## print outputs
            message = f'''
            iter: {iter+1}
            loss: {loss}
            '''
            print_message_with_master_process(rank, message)
            if args.use_torch_profiler:
                p.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', default=256, type=int)
    parser.add_argument('--TP', action='store_true')
    parser.add_argument('--use_torch_profiler', action='store_true')
    args, _ = parser.parse_known_args()
    main(args)