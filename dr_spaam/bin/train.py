import argparse
import os
from shutil import copyfile

import yaml

import torch
from torch import optim

from dr_spaam.utils.dataset import create_dataloader
from dr_spaam.utils.logger import create_logger, create_tb_logger
from dr_spaam.utils.train_utils import Trainer, LucasScheduler, load_checkpoint
from dr_spaam.utils.eval_utils import model_fn, eval_epoch_with_output, cfg_to_model

from eval import eval


torch.backends.cudnn.benchmark = True  # Run benchmark to select fastest implementation of ops.

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("--cfg", type=str, required=True, help="configuration of the experiment")
parser.add_argument("--ckpt", type=str, required=False, default=None)
args = parser.parse_args()

with open(args.cfg, 'r') as f:
    cfg = yaml.safe_load(f)
    cfg['name'] = os.path.basename(args.cfg).split(".")[0] + cfg['tag']


if __name__ == '__main__':
    root_result_dir = os.path.join('./', 'output', cfg['name'])
    os.makedirs(root_result_dir, exist_ok=True)
    copyfile(args.cfg, os.path.join(root_result_dir, os.path.basename(args.cfg)))

    ckpt_dir = os.path.join(root_result_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    logger, tb_logger = create_logger(root_result_dir), create_tb_logger(root_result_dir)
    logger.info('**********************Start logging**********************')

    # log to file
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    # create dataloader & network & optimizer
    train_loader, eval_loader = create_dataloader(data_path="./data/DROWv2-data",
                                                  num_scans=cfg['num_scans'],
                                                  batch_size=cfg['batch_size'],
                                                  num_workers=cfg['num_workers'],
                                                  network_type=cfg['network'],
                                                  train_with_val=cfg['train_with_val'],
                                                  use_data_augumentation=cfg['use_data_augumentation'],
                                                  cutout_kwargs=cfg['cutout_kwargs'],
                                                  polar_grid_kwargs=cfg['polar_grid_kwargs'],
                                                  pedestrian_only=cfg['pedestrian_only'])

    model = cfg_to_model(cfg)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), amsgrad=True)
    if 'lr_kwargs' in cfg:
        e0, e1 = cfg['lr_kwargs']['e0'], cfg['lr_kwargs']['e1']
    else:
        e0, e1 = 0, cfg['epochs']
    lr_scheduler = LucasScheduler(optimizer, 0, 1e-3, cfg['epochs'], 1e-6)

    if args.ckpt is not None:
        starting_iteration, starting_epoch = load_checkpoint(
            model=model, optimizer=optimizer, filename=args.ckpt, logger=logger)
    elif os.path.isfile(os.path.join(ckpt_dir, 'sigterm_ckpt.pth')):
        starting_iteration, starting_epoch = load_checkpoint(
            model=model, optimizer=optimizer,
            filename=os.path.join(ckpt_dir, 'sigterm_ckpt.pth'),
            logger=logger)
    else:
        starting_iteration, starting_epoch = 0, 0

    # start training
    logger.info('**********************Start training**********************')

    model_fn_eval = lambda m, d, e, i: eval_epoch_with_output(
            model=m, test_loader=d, epoch=e, it=i, root_result_dir=root_result_dir,
            tag=cfg['name'], split='val', writing=True, plotting=True, save_pkl=True,
            tb_log=tb_logger, vote_kwargs=cfg['vote_kwargs'], full_eval=False)

    trainer = Trainer(
        model,
        model_fn,
        optimizer,
        ckpt_dir=ckpt_dir,
        lr_scheduler=lr_scheduler,
        model_fn_eval=model_fn_eval,
        tb_log=tb_logger,
        grad_norm_clip=cfg['grad_norm_clip'],
        logger=logger)

    trainer.train(num_epochs=cfg['epochs'],
                  train_loader=train_loader,
                  eval_loader=eval_loader,
                  eval_frequency=max(int(cfg['epochs'] / 20), 1),
                  ckpt_save_interval=max(int(cfg['epochs'] / 10), 1),
                  lr_scheduler_each_iter=True,
                  starting_iteration=starting_iteration,
                  starting_epoch=starting_epoch)

    # testing
    logger.info('**********************Start testing (val)**********************')
    eval(model, cfg, epoch=trainer._epoch+1, split='val', it=trainer._it,
         writing=True, plotting=True, save_pkl=True, tb_log=tb_logger)

    logger.info('**********************Start testing (test)**********************')
    eval(model, cfg, epoch=trainer._epoch+1, split='test', it=trainer._it,
         writing=True, plotting=True, save_pkl=True, tb_log=tb_logger)

    tb_logger.close()
    logger.info('**********************End**********************')
