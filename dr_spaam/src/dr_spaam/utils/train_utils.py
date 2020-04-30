import os
import torch
from torch.nn.utils import clip_grad_norm_
import tqdm

def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state}


def save_checkpoint(state, filename='checkpoint', logger=None):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
    if logger is not None:
        logger.info('Checkpoint saved to %s' % filename)


def load_checkpoint(model=None, optimizer=None, filename='checkpoint', logger=None):
    if os.path.isfile(filename):
        if logger is not None:
            logger.info("Loading from checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch'] if 'epoch' in checkpoint.keys() else -1
        it = checkpoint.get('it', 0.0)
        if model is not None and checkpoint['model_state'] is not None:
#            # @TODO Dirty fix, to be removed
#            if 'gate.neighbor_masks' in checkpoint['model_state']:
#                del checkpoint['model_state']['gate.neighbor_masks']
            model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
    else:
        print('Could not find %s' % filename)
        raise FileNotFoundError

    return it, epoch


class LucasScheduler(object):
    """
    Return `v0` until `e` reaches `e0`, then exponentially decay
    to `v1` when `e` reaches `e1` and return `v1` thereafter, until
    reaching `eNone`, after which it returns `None`.

    Copyright (C) 2017 Lucas Beyer - http://lucasb.eyer.be =)
    """
    def __init__(self, optimizer, e0, v0, e1, v1, eNone=float('inf')):
        self.e0, self.v0 = e0, v0
        self.e1, self.v1 = e1, v1
        self.eNone = eNone
        self._optim = optimizer

    def step(self, epoch):
        if epoch < self.e0:
            lr = self.v0
        elif epoch < self.e1:
            lr = self.v0 * (self.v1/self.v0)**((epoch-self.e0)/(self.e1-self.e0))
        elif epoch < self.eNone:
            lr = self.v1

        for group in self._optim.param_groups:
            group['lr'] = lr

    def get_lr(self):
        return self._optim.param_groups[0]['lr']


class Trainer(object):
    def __init__(self, model, model_fn, optimizer, ckpt_dir, lr_scheduler,
                 model_fn_eval, logger, tb_log, grad_norm_clip):
        self.model, self.optimizer, self.lr_scheduler = model, optimizer, lr_scheduler
        self.model_fn, self.model_fn_eval = model_fn, model_fn_eval
        self.ckpt_dir, self.logger, self.tb_log = ckpt_dir, logger, tb_log
        self.grad_norm_clip = grad_norm_clip

        self._epoch, self._it = 0, 0

        import signal
        signal.signal(signal.SIGINT, self._sigterm_cb)
        signal.signal(signal.SIGTERM, self._sigterm_cb)

    def _sigterm_cb(self, signum, frame):
        self.logger.warning('Received signal %s at frame %s' % (signum, frame))
        ckpt_name = os.path.join(self.ckpt_dir, 'sigterm_ckpt')
        save_checkpoint(checkpoint_state(self.model, self.optimizer, self._epoch, self._it),
                        filename=ckpt_name, logger=self.logger)
        self.tb_log.flush()
        self.tb_log.close()
        import sys; sys.exit()

    def _train_it(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        loss, tb_dict, _ = self.model_fn(self.model, batch)

        loss.backward()
        if self.grad_norm_clip > 0:
            clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
        self.optimizer.step()

        return loss.item(), tb_dict

    def train(self, num_epochs, train_loader, eval_loader=None, eval_frequency=1,
              ckpt_save_interval=5, lr_scheduler_each_iter=True, starting_epoch=0,
              starting_iteration=0):
        self._it = starting_iteration
        with tqdm.trange(starting_epoch, num_epochs, desc='epochs') as tbar, \
                tqdm.tqdm(total=len(train_loader), leave=False, desc='train') as pbar:

            for self._epoch in tbar:
                if not lr_scheduler_each_iter:
                    self.lr_scheduler.step(self._epoch)

                # train one epoch
                for cur_it, batch in enumerate(train_loader):
                    if lr_scheduler_each_iter:
                        self.lr_scheduler.step(self._epoch + cur_it / len(train_loader))

                    cur_lr = self.lr_scheduler.get_lr()
                    self.tb_log.add_scalar('learning_rate', cur_lr, self._it)

                    loss, tb_dict = self._train_it(batch)

                    disp_dict = {'loss': loss, 'lr': cur_lr}

                    # log to console and tensorboard
                    pbar.update()
                    pbar.set_postfix(dict(total_it=self._it))
                    tbar.set_postfix(disp_dict)
                    tbar.refresh()

                    self.tb_log.add_scalar('train_loss', loss, self._it)
                    self.tb_log.add_scalar('learning_rate', cur_lr, self._it)
                    for key, val in tb_dict.items():
                        self.tb_log.add_scalar('train_' + key, val, self._it)

                    self._it += 1

                # save trained model
                trained_epoch = self._epoch + 1
                if trained_epoch % ckpt_save_interval == 0:
                    ckpt_name = os.path.join(self.ckpt_dir, 'ckpt_e%d' % trained_epoch)
                    save_checkpoint(
                        checkpoint_state(self.model, self.optimizer, trained_epoch, self._it),
                        filename=ckpt_name, logger=self.logger)

                # eval one epoch
                if eval_loader is not None and trained_epoch % eval_frequency == 0:
                    pbar.close()
                    with torch.set_grad_enabled(False):
                        self.model.eval()
                        self.model_fn_eval(self.model, eval_loader, self._epoch, self._it)

                self.tb_log.flush()

                pbar.close()
                pbar = tqdm.tqdm(total=len(train_loader), leave=False, desc='train')
                pbar.set_postfix(dict(total_it=self._it))