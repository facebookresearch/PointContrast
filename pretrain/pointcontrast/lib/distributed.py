# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

"""Distributed helpers."""
import pickle
import time

import functools
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd import Function


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def all_gather_differentiable(tensor):
    """
        Run differentiable gather function for SparseConv features with variable number of points.
        tensor: [num_points, feature_dim]
    """
    world_size = get_world_size()
    if world_size == 1:
        return [tensor]

    num_points, f_dim = tensor.size()
    local_np = torch.LongTensor([num_points]).to("cuda")
    np_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(np_list, local_np)
    np_list = [int(np.item()) for np in np_list]
    max_np = max(np_list)

    tensor_list = []
    for _ in np_list:
        tensor_list.append(torch.FloatTensor(size=(max_np, f_dim)).to("cuda"))
    if local_np != max_np:
        padding = torch.zeros(size=(max_np-local_np, f_dim)).to("cuda").float()
        tensor = torch.cat((tensor, padding), dim=0)
        assert tensor.size() == (max_np, f_dim)

    dist.all_gather(tensor_list, tensor)

    data_list = []
    for gather_np, gather_tensor in zip(np_list, tensor_list):
        gather_tensor = gather_tensor[:gather_np]
        assert gather_tensor.size() == (gather_np, f_dim)
        data_list.append(gather_tensor)
    return data_list


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def is_master_proc(num_gpus):
    """Determines if the current process is the master process.

    Master process is responsible for logging, writing and loading checkpoints.
    In the multi GPU setting, we assign the master role to the rank 0 process.
    When training using a single GPU, there is only one training processes
    which is considered the master processes.
    """
    return num_gpus == 1 or torch.distributed.get_rank() == 0


def init_process_group(proc_rank, world_size):
    """Initializes the default process group."""
    # Set the GPU to use
    torch.cuda.set_device(proc_rank)
    # Initialize the process group
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="tcp://{}:{}".format("localhost", "10001"),
        world_size=world_size,
        rank=proc_rank
    )

def destroy_process_group():
    """Destroys the default process group."""
    torch.distributed.destroy_process_group()

@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
        world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size,), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor

def all_gather_obj(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

def scaled_all_reduce_dict_obj(res_dict, num_gpus):
    """ Reduce a dictionary of arbitrary objects. """
    res_dict_list = all_gather_obj(res_dict)
    assert len(res_dict_list) == num_gpus
    res_keys = res_dict_list[0].keys()
    res_dict_reduced = {}
    for k in res_keys:
        res_dict_reduced[k] = 1.0 * sum([r[k] for r in res_dict_list]) / num_gpus
    return res_dict_reduced	

def scaled_all_reduce_dict(res_dict, num_gpus):
    """ Reduce a dictionary of tensors. """
    reductions = []
    for k in res_dict:
        reduction = torch.distributed.all_reduce(res_dict[k], async_op=True)
        reductions.append(reduction)
    for reduction in reductions:
        reduction.wait()
    for k in res_dict:
        res_dict[k] = res_dict[k].clone().mul_(1.0 / num_gpus)
    return res_dict

def scaled_all_reduce(tensors, num_gpus, is_scale=True):
    """Performs the scaled all_reduce operation on the provided tensors.

    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of
    the process group (equivalent to cfg.NUM_GPUS).
    """
    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = torch.distributed.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    if is_scale:
        for tensor in tensors:
            tensor.mul_(1.0 / num_gpus)
    return tensors

def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    # gathers = []
    tensor_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(
            # list(tensor_all.unbind(0)),
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)
        # gathers.append(gather)

    # Wait for gathers to finish
    # for gather in gathers:
    #     gather.wait()

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor

class AllGatherWithGradient(Function):
    """AllGatherWithGradient"""

    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, input):
        x_gather = all_gather_batch([input])[0]
        return x_gather

    def backward(self, grad_output):
        N = grad_output.size(0)
        mini_batchsize = N // self.args.num_gpus
        # Does not scale for gradient
        grad_output = scaled_all_reduce([grad_output], self.args.num_gpus, is_scale=False)[0]

        cur_gpu = get_rank()
        grad_output = \
            grad_output[cur_gpu * mini_batchsize: (cur_gpu + 1) * mini_batchsize]
        return grad_output

class AllGatherVariableSizeWithGradient(Function):
    """
        All Gather with Gradient for variable size inputs: [different num_points, ?].
        Return a list.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config


    def forward(self, input):
        x_gather_list = all_gather_differentiable(input)
        input_size_list =all_gather_obj(input.size(0))
        cur_gpu = get_rank()
        if (cur_gpu == 0):
            self.start_list = [sum(input_size_list[:t]) for t in range(len(input_size_list)+1)]

        dist.barrier()
        
        return torch.cat(x_gather_list, 0)

    def backward(self, grad_output):
        grad_output = scaled_all_reduce([grad_output], self.config.num_gpus, is_scale=False)[0]
        cur_gpu = get_rank()
        return grad_output[self.start[cur_gpu]:self.start[cur_gpu+1]]

        return grad_output

