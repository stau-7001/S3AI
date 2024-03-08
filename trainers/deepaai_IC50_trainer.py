import os
from itertools import chain
from typing import Dict, Callable
import torch
from commons.amino_seq_to_ft import amino_seq_to_one_hot
from trainers.trainer import Trainer
import numpy as np

def get_pos_in_raw_arr(sub_arr, raw_arr):
    '''
    :param sub_arr:  np array shape = [m]
    :param raw_arr:  np array shape = [n] 无重复数字
    :return: np array  shape = [m]
    '''
    raw_pos_dict = {}
    for i in range(raw_arr.shape[0]):
        raw_pos_dict[raw_arr[i]] = i
    trans_pos_arr = []
    for num in sub_arr:
        trans_pos_arr.append(raw_pos_dict[num])

    return np.array(trans_pos_arr, dtype=np.long)

class deepaaiIC50Trainer(Trainer):
    def __init__(self, model, args, metrics: Dict[str, Callable], main_metric: str,
                 device: torch.device, tensorboard_functions: Dict[str, Callable],
                 optim=None, main_metric_goal: str = 'min', loss_func=torch.nn.MSELoss,
                 scheduler_step_per_batch: bool = True, **kwargs):
        super(deepaaiIC50Trainer, self).__init__(model, args, metrics, main_metric, device, tensorboard_functions,
                                                               optim, main_metric_goal, loss_func, scheduler_step_per_batch)
        
    def forward_pass(self, batch):
        batch_antibody_idx = batch['batch_antibody_idx']        
        batch_virus_idx = batch['batch_virus_idx']
        antibody_graph_map_arr = get_map_index_for_sub_arr(
            antibody_graph_node_idx, np.arange(0, 254))
        virus_graph_map_arr = get_map_index_for_sub_arr(
            virus_graph_node_idx, np.arange(0, 940))
        batch_antibody_node_idx_in_graph = antibody_graph_map_arr[batch_antibody_idx]
        batch_virus_node_idx_in_graph = virus_graph_map_arr[batch_virus_idx]

#         print(batch[0])
        Ab_ft = torch.from_numpy(amino_seq_to_one_hot(batch['H'][0]+batch['L'][0],'antibody') )       
        Ag_ft = torch.from_numpy(amino_seq_to_one_hot(batch['spike'][0],'virus'))
#         print(Ab_ft.shape)
        ft_dict = {
            'antibody_graph_node_kmer_ft': antibody_graph_node_kmer_ft,
            'virus_graph_node_kmer_ft': virus_graph_node_kmer_ft,
            'antibody_graph_node_pssm_ft': antibody_graph_node_pssm_ft,
            'virus_graph_node_pssm_ft': virus_graph_node_pssm_ft,
            'antibody_amino_ft': batch_antibody_amino_ft,
            'virus_amino_ft': batch_virus_amino_ft,
            'antibody_idx': batch_antibody_node_idx_in_graph,
            'virus_idx': batch_virus_node_idx_in_graph
        }
        pred = self.model(Ab_ft.to(self.device),Ag_ft.to(self.device))  # forward the sequence to the model
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
