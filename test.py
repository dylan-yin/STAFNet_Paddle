import paddle
import paddle.nn as nn
import os
import numpy as np
import math
from math import sqrt
from Embed import DataEmbedding
import argparse
from dataloader import pygmmdataLoader
from tqdm import tqdm
from SATFNet import Gat_TimesNet_mm
# from pgl.nn import GATv2Conv


class Trainer:
    def __init__(self,model,data_loader,criterion, optimizer, device, 
                 num_epochs,metric_ftns,valid_data_loader=None):
        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.start_epoch = 1
        self.epochs = num_epochs
        self.save_period =1
        self.start_save_epoch=20
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)

        
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.device = device

    def train(self):
        loss_history = []
        epoch_loss = 0

        for epoch in range(self.start_epoch, self.epochs + 1):
            self._train_epoch(epoch)
            best = False
        # 保存模型
            if (epoch % self.save_period == 0 and epoch > self.start_save_epoch):
                self._save_checkpoint(epoch, save_best=best)
                filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
                paddle.save(model, path=filename)
                

        print("Training complete.")
        return loss_history
    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        # arch = type(self.model).__name__
        # state = {'arch': arch, 'epoch': epoch, 'state_dict': {k.replace(
        #     'module.', ''): v for k, v in self.model.state_dict().items()},
        #     'optimizer': self.optimizer.state_dict(), 'monitor_best': self.
        #     mnt_best, 'config': self.config}
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        paddle.save(obj=self.model, path=filename)
        # self.logger.info('Saving checkpoint: {} ...'.format(filename))
        # if save_best:
        #     best_path = str(self.checkpoint_dir / 'model_best.pth')
        #     paddle.save(obj=state, path=best_path)
        #     self.logger.info('Saving current best: model_best.pth ...')

    def _train_epoch(self, epoch): 
        self.model.train()
        pbar = tqdm(enumerate(self.data_loader), total=len(self.data_loader))
        for batch_idx, (data, target) in pbar:
            # for key, value in data.items():
            #     if paddle.is_tensor(x=value):
            #         data[key] = value.to(self.device)
            # target = target.to(self.device)

            self.optimizer.clear_grad()
            output, _ = self.model(data)
            loss = self.criterion(output[:, :], target[:, :, :1])
            loss.backward()
            self.optimizer.step()
            pbar.set_description('Train Epoch: {} {} '.format(epoch, self._progress(batch_idx + 1)))
            pbar.set_postfix(train_loss=loss.item())
            if batch_idx == self.len_epoch:
                break
            if self.do_validation:
                outputs, targets, attns = self._valid_epoch(epoch)
                outputs = [output.to(self.device) for output in outputs]
                targets = [target.to(self.device) for target in targets]
                # for met in self.metric_ftns:
                #     self.valid_metrics.update(met.__name__, met(paddle.
                #         concat(x=outputs, axis=0), paddle.concat(x=targets,
                #         axis=0)))
                val_loss = self.criterion(output[:, :], target[:, :, :1])
                pbar.set_description('Val Epoch: {} {} '.format(epoch, self._progress(batch_idx + 1)))
                pbar.set_postfix(train_loss=val_loss.item())
        return
        
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)        

args=  {
                "task_name": "forecast",
                "output_attention": True,
                "seq_len": 72,
                "label_len": 24,
                "pred_len": 48,

                "aq_gat_node_features" : 7,
                "aq_gat_node_num": 35,

                "mete_gat_node_features" : 7,
                "mete_gat_node_num": 18,

                "gat_hidden_dim": 32,
                "gat_edge_dim": 3,
                "gat_embed_dim": 32,

                "e_layers": 1,
                "enc_in": 32,
                "dec_in": 7,
                "c_out": 7,
                "d_model": 16 ,
                "embed": "fixed",
                "freq": "t",
                "dropout": 0.05,
                "factor": 3,
                "n_heads": 4,

                "d_ff": 32 ,
                "num_kernels": 6,
                "top_k": 4
            }

dataLoader_args = {
            "data_dir": "data/2020-2023_new/train_data.pkl",
            "batch_size": 32,
            "shuffle": True,
            "num_workers": 0,
            "training": True
        }
valid_loader_args=  {
            "data_dir": "data/2020-2023_new/val_data.pkl",
            "batch_size": 32,
            "shuffle": False,
            "num_workers": 0,
            "training": False
        }
lr = 0.0005
paddle.utils.run_check()
args = argparse.Namespace(**args)
device = paddle.set_device('gpu' if paddle.is_compiled_with_cuda() else 'cpu')
paddle.device.set_device('gpu')
model = Gat_TimesNet_mm(args)
dataloader = pygmmdataLoader(args,**dataLoader_args)
valid_loader = pygmmdataLoader(args,**valid_loader_args)
criterion = paddle.nn.MSELoss()
# criterion = paddle.nn.CrossEntropyLoss()
optimizer = paddle.optimizer.Adam(learning_rate=lr,
                                parameters=model.parameters())
# optimizer=  paddle.optimizer.Adam(**optimizer_args)
num_epochs = 40
metric_ftns = None
trainer = Trainer(model,dataloader,criterion, optimizer, device, num_epochs,metric_ftns,valid_loader)


trainer.train()