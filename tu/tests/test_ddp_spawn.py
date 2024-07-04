from tu.train_setup import spawn_ddp
import json
import torch.nn as nn
import torch
import time
import argparse
import torch.distributed as dist
from tu.ddp import init_ddp
import sys


def worker(rank, world_size, args):
    print('worker from rank', rank, 'world size', world_size)
    args.rank = rank
    print(json.dumps(vars(args), indent=4, sort_keys=True))

    init_ddp(rank, world_size, port='12345')

    assert dist.is_nccl_available()
    assert dist.is_initialized()
    assert torch.cuda.device_count() == dist.get_world_size()

    print(torch.cuda.current_device())

    model = nn.Conv2d(1, 1, kernel_size=(3, 3))
    model.cuda()

    print(model.weight.device)

    time.sleep(20)

    assert args.rank == rank


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--foo', type=str, default='0')

    args = parser.parse_args()

    spawn_ddp(args, worker)
