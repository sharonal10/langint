from typing import Dict, Union
from tu.configs import list_of_dicts__to__dict_of_lists
import time
from tu.utils.visualize import dump_helper
import json
import torch.distributed as dist
from tu.ddp import check_ddp_consistency
from tu.utils.config import check_cfg_consistency
import logging
import os
from tu.utils.training import (
    get_grad_norm, get_optimizer_lr, get_children_grad_norm, get_param_norm, get_grad_norm_safe, get_children_grad_norm_safe
)
from tu.loggers.utils import setup_vi
import torch
import torch.nn as nn
from tu.utils.training import process_batch
from tu.utils.config import build_from_config
import shutil
import numpy as np


logger = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(self, modules: Dict[str, nn.Module],
                 optimizers: Dict[str, Union[torch.optim.Optimizer]],
                 schedulers: Dict,
                 loss_modules: Dict[str, Dict],
                 loss_weights: Dict[str, float],
                 writer, title=None, it=-1, checkpoint_dir=None):
        self.modules = modules
        if len(modules) == 1:
            self.module_key, = modules.keys()
        else:
            self.module_key = None
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.it = it
        self.writer = writer

        self.loss_modules = {k: build_from_config(v) for k, v in loss_modules.items()}
        self.loss_weights = loss_weights

        if self.writer is not None:
            self.vi, self.vi_helper = setup_vi(writer.get_logdir())
        
        self.title = title
        self.checkpoint_dir = checkpoint_dir

    def compute_loss_terms(self, out, data, module_key=None, writer=None) -> Dict:
        if self.module_key is None:
            raise NotImplementedError()
        return {k: v(out=out, data=data, writer=writer) for k, v in self.loss_modules.items()}

    def compute_loss_total(self, loss_terms: Dict[str, torch.Tensor]):
        return sum(loss_terms[k] * self.loss_weights[k] for k in loss_terms.keys())

    def train_step(self, data, module_key=None, print_every=100, writer=None):
        self.it += 1
        print('tu.it:', self.it)

        if module_key is None:
            assert self.module_key is not None
            module_key = self.module_key
        module = self.modules[module_key]
        optimizer = self.optimizers[module_key]
        scheduler = self.schedulers[module_key]

        if self.it == 2 or (self.it > 1 and self.it % 1000 == 0):
            torch.save(module.state_dict(), os.path.join(self.writer.get_logdir(), f'tsne_params_{self.it}.pth'))
            print('saved tsne in:', os.path.join(self.writer.get_logdir(), f'tsne_params_{self.it}.pth'))

        module.train()

        ret = dict()
        optimizer.zero_grad()
        out = module(data)

        #relevant 7-6
        loss_terms = self.compute_loss_terms(out, data, module_key=module_key, writer=writer)
        loss = self.compute_loss_total(loss_terms)
        loss.backward()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if self.it % print_every == 0: #so we don't waste time computing if we're not going to save it
            for k, v in loss_terms.items():
                ret[f'train/loss/{module_key}/{k}'] = v
            ret[f'train/loss/{module_key}/total'] = loss

            grad_norm = get_grad_norm_safe(module, verbose=os.getenv('DEBUG') == '1')
            ret[f'stats/grad_norm/{module_key}'] = grad_norm

            param_norm = get_param_norm(module)
            ret[f'stats/param_norm/{module_key}'] = param_norm

            for k, v in get_optimizer_lr(optimizer).items():
                ret[f'stats/lr/{module_key}/{k}'] = v

            if len(list(module.children())) > 1:
                for k, v in get_children_grad_norm_safe(module, verbose=os.getenv('DEBUG') == '1').items():
                    if v is None:
                        v = -1
                    ret[f'grad_norm/{module_key}/{k}'] = v

            if isinstance(module.usage_counts[0], list):
                ret['train/usage_variance'] = np.var(module.usage_counts[0]) #change if we have 2 vaes
                ret['train/usage_max'] = max(module.usage_counts[0])
            else:
                ret['train/usage_variance'] = np.var(module.usage_counts)
                ret['train/usage_max'] = max(module.usage_counts)

            # for k, v in self.loss_modules.items():
            #     ret[f'train/fid/{module_key}/{k}'] = v.calculate_fid(gt_images=data['image'], generated_images=v.visualize({'embeddings': out['embeddings']}, data)['image'])

        return ret

    def visualize(self, data):
        # don't make a new page when we're doing inference training
        if self.writer is not None:
            self.vi.title = f'It {self.it}'
            self.vi.end_html()
            self.vi.begin_html()
        self._visualize_core(data, prefix='train')

    def _visualize_core(self, data, prefix=None):
        if self.module_key is None:
            raise NotImplementedError()
        model = self.modules[self.module_key]
        model.eval()
        out = model.visualize(data)
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                dump_helper(self, k, v, prefix=prefix)

    def validate(self, dataloader):
        module_key = self.module_key
        model = self.modules[module_key]
        model.eval()

        ret = dict()
        out_all = []
        for batch in dataloader:
            data = process_batch(batch)
            with torch.no_grad():
                out = model(data)
                loss_terms = self.compute_loss_terms(out, data)
                loss = self.compute_loss_total(loss_terms)
                out_all.append({**loss_terms, 'total': loss})

        out_all = list_of_dicts__to__dict_of_lists(out_all)
        for k, v in out_all.items():
            ret[f'val/loss/{module_key}/{k}'] = torch.stack(v).mean()

        if self.writer is not None:
            for k, v in ret.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                    v = v.float()
                self.writer.add_scalar(k, v, self.it)

    def save_checkpoint(self, filename=None, **kwargs):
        if dist.is_initialized() and dist.get_rank() > 0:
            return
        if filename is None:
            filename = f'it_{self.it:08d}.pt'
        path = os.path.join(self.writer.get_logdir(), 'checkpoints', filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'modules': {k: v.state_dict() for k, v in self.modules.items()},
            'optimizers': {k: v.state_dict() for k, v in self.optimizers.items()},
            'schedulers': {k: v.state_dict() if v is not None else None for k, v in self.schedulers.items()},
            'it': self.it,
            **kwargs,
        }, path)

    def load_checkpoint(self, path):
        state_dict = torch.load(path)
        # self.it = state_dict['it'] # we want iteration to start at 0 again actually
        logging.info(f'resume from it: {self.it}')
        old_lr = {}
        for k, v in state_dict['optimizers'].items():
            for param_group in self.optimizers[k].param_groups:
                assert k not in old_lr, old_lr # for now we should have only one param_group
                old_lr[k] = param_group['lr']

        for k, v in state_dict['modules'].items():
            self.modules[k].load_state_dict(v)
        for k, v in state_dict['optimizers'].items():
            self.optimizers[k].load_state_dict(v)
            for param_group in self.optimizers[k].param_groups:
                param_group['lr'] = old_lr[k]
        for k, v in state_dict['schedulers'].items():
            if v is not None:
                self.schedulers[k].load_state_dict(v)
        return state_dict


def load_checkpoint(trainer, cfg, path):
    if path is None:
        return -1
    with open(os.path.abspath(os.path.join(path, '../../cfg.json')), 'r') as f:
        checkpoint_cfg = json.load(f)
    _ = check_cfg_consistency(cfg, checkpoint_cfg, ignore_keys={'log_dir', 'runtime.*', 'training.*', 'trainer.*'})
    epoch = trainer.load_checkpoint(path)['epoch']
    return epoch


def train_loops(print_every, visualize_every, checkpoint_every, eval_every,
                cfg, trainer, train_loader, val_loader, max_it=None, max_epoch=None, epoch=-1):
    if os.getenv('DEBUG') == '1' and dist.is_initialized():
        for k in sorted(trainer.module_keys):  # must sort!! otherwise deadlock
            check_ddp_consistency(trainer.modules[k])

    trainer.visualize(process_batch(next(iter(train_loader))))
    # exit()
    assert max_it is not None or max_epoch is not None, (max_it, max_epoch)
    t00b = t0b = time.time()
    while True:
        epoch += 1
        print('epoch numbers:', epoch, max_epoch, trainer.it, max_it)
        if max_epoch is not None and epoch >= max_epoch:
            break
        if max_it is not None and trainer.it > max_it:
            break
        if dist.is_initialized():
            train_loader.sampler.set_epoch(epoch)
        for batch in train_loader:
            batch = process_batch(batch)
            loss = trainer.train_step(batch, print_every=print_every, writer=trainer.writer)
            if trainer.it % print_every == 0:
                if trainer.writer is not None:
                    for k, v in loss.items():
                        if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                            v = v.float()
                        trainer.writer.add_scalar(k, v, trainer.it)
                    trainer.writer.add_scalar('train/epoch', epoch, trainer.it)

                if trainer.it <= 100:
                    info_txt = '[Epoch %02d] it=%03d, time=%.3f' % (epoch, trainer.it, time.time() - t0b)
                    for (k, v) in loss.items():
                        info_txt += ', %s: %.4f' % (k, v)
                    logger.info(info_txt)
                t0b = time.time()

            if trainer.it % visualize_every == 0 or trainer.it == 100:
                batch["dataset"] = train_loader.dataset
                trainer.visualize(batch)

            if checkpoint_every > 0 and trainer.it % checkpoint_every == 0:
                trainer.save_checkpoint(epoch=epoch, loss=loss)

            if eval_every > 0 and (trainer.it % eval_every == 0  or trainer.it == 100):
                print('eval_every:', eval_every)
                trainer.validate(val_loader)

            if os.getenv('DEBUG') == '1' and dist.is_initialized() and trainer.it < 10:
                for k in sorted(trainer.module_keys):  # must sort!! otherwise deadlock
                    check_ddp_consistency(trainer.modules[k])

            if trainer.it % 1000 == 0:
                logdir = trainer.writer.get_logdir()
                it = trainer.it

                # Define the original and destination file paths
                original_path = os.path.join(logdir, 'index.html')
                destination_path = os.path.join(logdir, f'index{it}.html')

                # Assert that the original file exists
                assert os.path.isfile(original_path), f"No such file: '{original_path}'"

                # Copy the file
                shutil.copy2(original_path, destination_path)

    logger.info(f'Epoch done. time={time.time() - t00b:.3f}s')
