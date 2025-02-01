import torch
import torch.distributed as dist
import os
import subprocess


def get_optimizer(parameters, config):
    if config.TYPE == "AdamW":
        return torch.optim.AdamW(parameters, **config.KWARGS)
    elif config.TYPE == "Adam":
        return torch.optim.Adam(parameters, **config.KWARGS)
    elif config.TYPE == "SGD":
        return torch.optim.SGD(parameters, **config.KWARGS)
    else:
        raise NotImplementedError


def get_scheduler(optimizer, config):
    if config.TYPE == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, **config.KWARGS)
    if config.TYPE == "CosineAnnealingWarmRestarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **config.KWARGS)
    else:
        raise NotImplementedError


def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    return rank, world_size
