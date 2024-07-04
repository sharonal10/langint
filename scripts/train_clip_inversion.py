# from langint.common import named_datasets
import torch.distributed as dist
import torch
from pathlib import Path
import json
from torch.utils.tensorboard import SummaryWriter
from tu.utils.config import build_from_config, overwrite_cfg_thorough_recursive
import os
import logging
from tu.train.setup import get_cfg, get_parser
from tu.train_setup import open_tensorboard, set_seed_benchmark, setup_ddp_safe, build_dataloaders, build_training_modules
from tu.utils.config import overwrite_cfg
from tu.trainers.simple_trainer import load_checkpoint


logger = logging.getLogger(__name__)


def main():
    rank = setup_ddp_safe()
    if os.getenv('DEBUG') == '1':
        torch.autograd.set_detect_anomaly(True)
    

    print('--------------------------------------------before parser-----------------------------------------------------')
    parser = get_parser()
    args = parser.parse_args()

    print('--------------------------------------------before logger -----------------------------------------------------')

    logger.info(json.dumps(vars(args)))

    print('--------------------------------------------before benchmark-----------------------------------------------------')

    set_seed_benchmark(args.seed)

    print('--------------------------------------------before configs-----------------------------------------------------')

    # args.dataset = named_datasets[args.dataset]
    cfg = get_cfg(args)

    print('--------------------------------------------before environment-----------------------------------------------------')

    if os.environ.get('SLURM_JOB_NAME') == 'bash' or os.getenv('DEBUG') == '1':
        overwrite_cfg(cfg['training']['train_loops_fn']['kwargs'], 'visualize_every', 200)
        overwrite_cfg(cfg['training']['train_loops_fn']['kwargs'], 'print_every', 200)
        # overwrite_cfg(cfg['training']['train_loops_fn']['kwargs'], 'checkpoint_every', 99999)
        # overwrite_cfg(cfg['training']['train_loops_fn']['kwargs'], 'eval_every', 100)

    print('--------------------------------------------before train loader-----------------------------------------------------')
    train_loader, val_loader = build_dataloaders(cfg, seed=args.seed, use_ddp=dist.is_initialized())
    overwrite_cfg_thorough_recursive(cfg, 'num_placeholder_words', train_loader.dataset.num_placeholder_words)
    print('config', cfg)
    modules, optimizers, schedulers = build_training_modules(cfg, use_ddp=dist.is_initialized())
    print('weight_decay:', list(optimizers.values())[0].param_groups[0]['weight_decay'])


    print('--------------------------------------------before logdir-----------------------------------------------------')
    log_dir = cfg['log_dir']
    if rank == 0:
        writer = SummaryWriter(log_dir)
        open_tensorboard(log_dir)
        logger.info(f'tensorboard --bind_all --logdir {Path(log_dir).absolute()}')
    else:
        writer = None

    trainer = build_from_config(cfg['trainer'], modules=modules, writer=writer, optimizers=optimizers, schedulers=schedulers)

    epoch = load_checkpoint(trainer, cfg, cfg['training']['checkpoint_dir'])

    build_from_config(cfg['training']['train_loops_fn'], cfg=cfg,
                      trainer=trainer, train_loader=train_loader, val_loader=val_loader, epoch=epoch)


if __name__ == "__main__":
    main()
