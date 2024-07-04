import json
import os
import torch.nn as nn
import torch
import time
import argparse
import torch.distributed as dist
from tu.ddp import init_ddp_with_launch
import sys


def main():
    rank = int(os.environ['LOCAL_RANK'])
    init_ddp_with_launch(rank)

    print(json.dumps(vars(args), indent=4, sort_keys=True))

    assert dist.is_nccl_available()
    assert dist.is_initialized()
    assert torch.cuda.device_count() == dist.get_world_size()

    print('OMP_NUM_THREADS', os.getenv('OMP_NUM_THREADS'))
    print('current device', torch.cuda.current_device())

    model = nn.Conv2d(1, 1, kernel_size=(3, 3))
    model.cuda()

    print(model.weight.device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--foo', type=str, default='0')
    # parser.add_argument('--local_rank', type=int, required=True)

    args = parser.parse_args()

    main()
