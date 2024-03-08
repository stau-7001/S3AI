import os
from itertools import chain
from typing import Dict, Callable
import torch

from trainers.trainer import Trainer


class IC50Trainer(Trainer):
    def __init__(self, model, args, metrics: Dict[str, Callable], main_metric: str,
                 device: torch.device, tensorboard_functions: Dict[str, Callable],
                 optim=None, main_metric_goal: str = 'min', loss_func=torch.nn.MSELoss,
                 scheduler_step_per_batch: bool = True, **kwargs):
        super(IC50Trainer, self).__init__(model, args, metrics, main_metric, device, tensorboard_functions,
                                                               optim, main_metric_goal, loss_func, scheduler_step_per_batch)
        
    def forward_pass(self, batch):
#         print(batch[0])
        pred = self.model(batch['H'],batch['L'],batch['spike'])  # forward the sequence to the model
        label = torch.FloatTensor(batch['IC50']).to(self.device)
#         print(pred.shape, label.shape, label.float().unsqueeze(0).shape)
        loss = self.loss_func(pred.float(), label.float().unsqueeze(0))
        return loss, pred, label.float().unsqueeze(0)

    def run_per_epoch_evaluations(self, data_loader):
        print('fitting linear probe')
        representations = []
        targets = []
        for batch in data_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            loss, pred, label = self.process_batch(batch, optim=None)
            representations.append(pred)
            targets.append(label)
            if len(representations) * len(pred) >= self.args.linear_probing_samples:
                break
        representations = torch.cat(representations, dim=0)
        targets = torch.cat(targets, dim=0)
        if len(representations) >= representations.shape[-1]:
            X, _ = torch.lstsq(targets, representations)
            X, _ = torch.lstsq(targets, representations)
            sol = X[:representations.shape[-1]]
            pred = representations @ sol
            mean_absolute_error = (pred - targets).abs().mean()
            self.writer.add_scalar('linear_probe_mae', mean_absolute_error.item(), self.optim_steps)
        else:
            raise ValueError(
                f'We have less linear_probing_samples {len(representations)} than the metric dimension {representations.shape[-1]}. Linear probing cannot be used.')

        print('finish fitting linear probe')


    def initialize_optimizer(self, optim):
        print(self.model.named_parameters())
        normal_params = [v for k, v in chain(self.model.named_parameters())]
#         batch_norm_params = [v for k, v in chain(self.model.named_parameters()) if
#                              'batch_norm' in k]
#         print(normal_params)
#         print(batch_norm_params)
        self.optim = optim([{'params': normal_params}],
                           **self.args.optimizer_params)

    def save_model_state(self, epoch: int, checkpoint_name: str):
        torch.save({
            'epoch': epoch,
            'best_val_score': self.best_val_score,
            'optim_steps': self.optim_steps,
            'model_state_dict': self.model.state_dict(),
            # 'model3d_state_dict': self.model3d.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'scheduler_state_dict': None if self.lr_scheduler == None else self.lr_scheduler.state_dict()
        }, os.path.join(self.writer.log_dir, checkpoint_name))
