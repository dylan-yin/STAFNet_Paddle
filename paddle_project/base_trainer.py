import paddle
import os
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter
from utils.dist import get_world_size, all_gather, is_main_process, synchronize


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, args):
        self.config = config
        self.args = args
        self.logger = config.get_logger('trainer', config['trainer'][
            'verbosity'])
        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.start_save_epoch = cfg_trainer['start_save_epoch']
        self.monitor = cfg_trainer.get('monitor', 'off')
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf
        self.start_epoch = 1
        self.checkpoint_dir = config.save_dir
        self.writer = TensorboardWriter(config.log_dir, self.logger,
            cfg_trainer['tensorboard'])
        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            if is_main_process():
                log = {'epoch': epoch}
                log.update(result)
                for key, value in log.items():
                    self.logger.info('    {:s}: {}'.format(str(key), value))
                best = False
                if self.mnt_mode != 'off':
                    try:
                        improved = self.mnt_mode == 'min' and log[self.
                            mnt_metric
                            ] <= self.mnt_best or self.mnt_mode == 'max' and log[
                            self.mnt_metric] >= self.mnt_best
                    except KeyError:
                        self.logger.warning(
                            "Warning: Metric '{}' is not found. Model performance monitoring is disabled."
                            .format(self.mnt_metric))
                        self.mnt_mode = 'off'
                        improved = False
                    if improved:
                        self.mnt_best = log[self.mnt_metric]
                        not_improved_count = 0
                        best = True
                    else:
                        not_improved_count += 1
                if (epoch % self.save_period == 0 and epoch > self.
                    start_save_epoch):
                    self._save_checkpoint(epoch, save_best=best)
            synchronize()
            not_improved_count_list = all_gather(not_improved_count)
            if max(not_improved_count_list) > self.early_stop:
                if is_main_process():
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. Training stops."
                        .format(self.early_stop))
                break

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {'arch': arch, 'epoch': epoch, 'state_dict': {k.replace(
            'module.', ''): v for k, v in self.model.state_dict().items()},
            'optimizer': self.optimizer.state_dict(), 'monitor_best': self.
            mnt_best, 'config': self.config}
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.
            format(epoch))
        paddle.save(obj=state, path=filename)
        self.logger.info('Saving checkpoint: {} ...'.format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            paddle.save(obj=state, path=best_path)
            self.logger.info('Saving current best: model_best.pth ...')

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        if is_main_process():
            self.logger.info('Loading checkpoint: {} ...'.format(resume_path))
        checkpoint = paddle.load(path=resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning(
                'Warning: Architecture configuration given in config file is different from that of checkpoint. This may yield an exception while state_dict is being loaded.'
                )
        if get_world_size() > 1:
            self.model.set_state_dict(state_dict={('module.' + k): v for k,
                v in checkpoint['state_dict'].items()})
        else:
            self.model.set_state_dict(state_dict=checkpoint['state_dict'])
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer'
            ]['type']:
            self.logger.warning(
                'Warning: Optimizer type given in config file is different from that of checkpoint. Optimizer parameters not being resumed.'
                )
        else:
            self.optimizer.set_state_dict(state_dict=checkpoint['optimizer'])
        self.logger.info('Checkpoint loaded. Resume training from epoch {}'
            .format(self.start_epoch))

    def reduce_loss(self, loss):
        world_size = get_world_size()
        if world_size < 2:
            return loss
        with paddle.no_grad():
            all_loss = loss.clone()
            paddle.distributed.reduce(tensor=all_loss, dst=0)
            if paddle.distributed.get_rank() == 0:
                all_loss /= world_size
        return all_loss

    def _accumulate_predictions_from_multiple_gpus(self, predictions_per_gpu):
        all_predictions = all_gather(predictions_per_gpu)
        if not is_main_process():
            return None
        else:
            return all_predictions
