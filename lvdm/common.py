import os, math
import numpy as np
from inspect import isfunction

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def gather_data(data, return_np=True):
    ''' gather data from multiple processes to one list '''
    data_list = [torch.zeros_like(data) for _ in range(dist.get_world_size())]
    dist.all_gather(data_list, data)  # gather not supported with NCCL
    if return_np:
        data_list = [data.cpu().numpy() for data in data_list]
    return data_list

def autocast(f):
    def do_autocast(*args, **kwargs):
        with torch.cuda.amp.autocast(enabled=True,
                                     dtype=torch.get_autocast_gpu_dtype(),
                                     cache_enabled=torch.is_autocast_cache_enabled()):
            return f(*args, **kwargs)
    return do_autocast


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def exists(val):
    return val is not None

def identity(*args, **kwargs):
    return nn.Identity()

def uniq(arr):
    return{el: True for el in arr}.keys()

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)

def isimage(x):
    if not isinstance(x,torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def shape_to_str(x):
    shape_str = "x".join([str(x) for x in x.shape])
    return shape_str

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

#import deepspeed
#ckpt = deepspeed.checkpointing.checkpoint
ckpt = torch.utils.checkpoint.checkpoint
def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        try:
            return ckpt(func, *inputs)
        except:
            args = tuple(inputs) + tuple(params)
            return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    @torch.cuda.amp.custom_bwd # add this
    def backward(ctx, *output_grads):
        '''
        for x in ctx.input_tensors:
            if isinstance(x, int):
                print('-----------------', ctx.run_function)
        '''
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
