import torch.distributed as dist
import torch


def broadcast_params(models, rank):
    for model in models:
        if isinstance(model, list):
            for subnet in model:
                for _, item in subnet.state_dict().items():
                    dist.broadcast(item, rank)
        else:
            for _, item in model.state_dict().items():
                dist.broadcast(item, rank)
        dist.barrier()



def all_gather(tensor):
    res = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(res, tensor)
    return torch.stack(res)


def all_reduce(tensor):
    dist.all_reduce(tensor)
    world_size = dist.get_world_size()
    tensor.data /= world_size

def reduce_value(value, average=True):
    world_size = dist.get_world_size()
    my_rank = dist.get_rank()
    value = value.cuda(my_rank)

    if world_size < 2:  # 单GPU的情况
        return value.cpu()

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size
        value = value.cpu()
        return value

def broadcast(tensor, rank=0):
    my_rank = dist.get_rank()
    tensor = tensor.cuda(my_rank)
    dist.broadcast(tensor, rank)
