import argparse
import os
import re

from icecream import install

from commons.utils import seed_all, get_random_indices, TENSORBOARD_FUNCTIONS

import seaborn

from trainers.IC50_trainer import IC50Trainer
from trainers.onehot_IC50_trainer import onehotIC50Trainer
from trainers.Multitask_IC50_trainer import MultitaskIC50Trainer

from datasets.ic50_dataset import IC50Dataset
from commons.utils import (TENSORBOARD_FUNCTIONS, get_random_indices,
                           pad_collate, HIV_Cls_collate, HIV_Reg_collate,seed_all,move_to_device)
import yaml
from datasets.custom_collate import *  # do not remove
from models import *  # do not remove
from torch.nn import *  # do not remove
from torch.optim import *  # do not remove
from commons.losses import *  # do not remove
from torch.optim.lr_scheduler import *  # do not remove
from sklearn.svm import SVC, SVR

from torch.utils.data import DataLoader, Subset

from trainers.metrics import (ROCAUC, F1, MAE, RMSE, ClassificationAccuracy,
                              PearsonR, SpearmanR, MCC, PRAUC)
from trainers.trainer import Trainer

# turn on for debugging C code like Segmentation Faults
import faulthandler
faulthandler.enable()
install()
seaborn.set_theme()
from datasets.HIV_dataset import HIVClsDataset, HIVRegDataset
from main import get_trainer
def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='./configs/test.yml')
    p.add_argument('--experiment_name', type=str, help='name that will be added to the runs folder output')
    p.add_argument('--logdir', type=str, default='runs', help='tensorboard logdirectory')
    p.add_argument('--num_epochs', type=int, default=2500, help='number of times to iterate through all samples')
    p.add_argument('--batch_size', type=int, default=1024, help='samples that will be processed in parallel')
    p.add_argument('--patience', type=int, default=20, help='stop training after no improvement in this many epochs')
    p.add_argument('--minimum_epochs', type=int, default=0, help='minimum numer of epochs to run')
    p.add_argument('--dataset', type=str, default='IC50', help='[]')
    p.add_argument('--num_train', type=int, default=-1, help='n samples of the model samples to use for train')
    p.add_argument('--seed', type=int, default=123, help='seed for reproducibility')
    p.add_argument('--num_val', type=int, default=None, help='n samples of the model samples to use for validation')
    p.add_argument('--multithreaded_seeds', type=list, default=[],
                   help='if this is non empty, multiple threads will be started, training the same model but with the different seeds')
    p.add_argument('--seed_data', type=int, default=123, help='if you want to use a different seed for the datasplit')
    p.add_argument('--loss_func', type=str, default='MSELoss', help='Class name of torch.nn like [MSELoss, L1Loss]')
    p.add_argument('--loss_params', type=dict, default={}, help='parameters with keywords of the chosen loss function')
    p.add_argument('--optimizer', type=str, default='Adam', help='Class name of torch.optim like [Adam, SGD, AdamW]')
    p.add_argument('--optimizer_params', type=dict, help='parameters with keywords of the chosen optimizer like lr')
    p.add_argument('--lr_scheduler', type=str,
                   help='Class name of torch.optim.lr_scheduler like [CosineAnnealingLR, ExponentialLR, LambdaLR]')
    p.add_argument('--lr_scheduler_params', type=dict, help='parameters with keywords of the chosen lr_scheduler')
    p.add_argument('--scheduler_step_per_batch', default=True, type=bool,
                   help='step every batch if true step every epoch otherwise')
    p.add_argument('--log_iterations', type=int, default=-1,
                   help='log every log_iterations iterations (-1 for only logging after each epoch)')
    # p.add_argument('--expensive_log_iterations', type=int, default=100,
    #                help='frequency with which to do expensive logging operations')
    p.add_argument('--eval_per_epochs', type=int, default=0,
                   help='frequency with which to do run the function run_eval_per_epoch that can do some expensive calculations on the val set or sth like that. If this is zero, then the function will never be called')
    p.add_argument('--linear_probing_samples', type=int, default=500,
                   help='number of samples to use for linear probing in the run_eval_per_epoch function of the self supervised trainer')
    p.add_argument('--metrics', default=[], help='tensorboard metrics [mae, mae_denormalized, qm9_properties ...]')
    p.add_argument('--main_metric', default='mae_denormalized', help='for early stopping etc.')
    p.add_argument('--main_metric_goal', type=str, default='min', help='controls early stopping. [max, min]')
    p.add_argument('--val_per_batch', type=bool, default=False,
                   help='run evaluation every batch and then average over the eval results. When running the molhiv benchmark for example, this needs to be Fale because we need to evaluate on all val data at once since the metric is rocauc')
    p.add_argument('--tensorboard_functions', default=[], help='choices of the TENSORBOARD_FUNCTIONS in utils')
    p.add_argument('--checkpoint', type=str, help='path to directory that contains a checkpoint to continue training')
    p.add_argument('--pretrain_checkpoint_str', type=str, help='Specify path to finetune from a pretrained checkpoint')    
    p.add_argument('--pretrain_checkpoint_ab', type=str, help='Specify path to finetune from a pretrained checkpoint')    
    p.add_argument('--pretrain_checkpoint_ag', type=str, help='Specify path to finetune from a pretrained checkpoint')
    p.add_argument('--transfer_layers', default=[],
                   help='strings contained in the keys of the weights that are transferred')
    p.add_argument('--frozen_layers', default=[],
                   help='strings contained in the keys of the weights that are transferred')
    p.add_argument('--transferred_lr', type=float, default=None, help='set to use a different LR for transfer layers')
    p.add_argument('--exclude_from_transfer', default=[],
                   help='parameters that usually should not be transferred like batchnorm params')
    p.add_argument('--num_epochs_local_only', type=int, default=1,
                   help='when training with OptimalTransportTrainer, this specifies for how many epochs only the local predictions will get a loss')

    p.add_argument('--collate_function', default='graph_collate', help='the collate function to use for DataLoader')
    p.add_argument('--collate_params', type=dict, default={},
                   help='parameters with keywords of the chosen collate function')
    p.add_argument('--device', type=str, default='cuda', help='What device to train on: cuda or cpu')

    p.add_argument('--models_to_save', type=list, default=[],
                   help='specify after which epochs to remember the best model')

    p.add_argument('--model_type', type=str, default='MultiTask_S3AI', help='Classname of one of the models in the models dir')
    p.add_argument('--model_parameters', type=dict, help='dictionary of model parameters')
    p.add_argument('--trainer', type=str, default='contrastive', help='[contrastive, byol, alternating, philosophy]')
    p.add_argument('--train_sampler', type=str, default=None, help='any of pytorchs samplers or a custom sampler')

    p.add_argument('--eval_on_test', type=bool, default=False, help='runs evaluation on test set if true')
    return p.parse_args()

def load_infer_model(args, device):
    if args.model_type == "SVM":
        model = SVR(C=32, gamma= 0.03125)
    else:
        model = globals()[args.model_type](
            device=device,
            **args.model_parameters
        )
    pretrained_path = args.checkpoint
    print('load checkpoint from '+ pretrained_path)
     # Load pre-trained parameters
    pretrained_params = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(pretrained_params['model_state_dict'], strict=False)

    return model

import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.patches as patches

def get_cam(args, Ab_seq_H, Ab_seq_L, Ag_seq):
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    model = load_infer_model(args, device=device)
    model.eval()
    with torch.no_grad():
        reg_pred, cls_pred = model(Ab_seq_H, Ab_seq_L, Ag_seq)  # forward the sequence to the model
        # 获取模型的激活图
        activations = model.conv_module.activations

    fig, axes = plt.subplots(nrows=1, ncols=len(activations), figsize=(len(activations) * 5, 5))
    # widths = [(25,32),(51,57),(98,113),(147,158),(173,179),(211,220)]
    widths = []
    for i, (name, activation) in enumerate(activations.items()):
        # 选取每个激活的第一个特征图进行可视化
        ax = axes[i]
        # 这里假设激活的形状为 (N, C, H, W), 我们选取第一个样本的第一个通道
        # heatmap = activation[0, 1].cpu().numpy()
        heatmap = torch.mean(activation[0], dim=0).cpu().numpy()

        print(heatmap.shape)
        ax.matshow(heatmap, cmap='viridis')
        ax.axis('on')
        for start, end in widths:
            rect = patches.Rectangle((start, 0), end - start, heatmap.shape[0], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        ax.set_title(name)
    plt.savefig('activations.png')
    plt.show()
    return reg_pred, cls_pred


def inference(args,Ab_seq_H, Ab_seq_L,Ag_seq):
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    model = load_infer_model(args, device=device)
    
    model.eval()
    print('model trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    with torch.no_grad():
        reg_pred, cls_pred = model(Ab_seq_H, Ab_seq_L, Ag_seq)  # forward the sequence to the model
    # TODO: add trans
    print(reg_pred, cls_pred)
    return reg_pred, cls_pred

def test_HIV(args, device, metrics_dict):
    model = load_infer_model(args, device=device)
    model.eval()
    all_data = HIVClsDataset()
    
    all_idx = np.load('./data/hiv_idx/cls/train_index_shuffled.npy')
    test_idx = all_idx[int(0.8 * len(all_idx)):int(0.9 * len(all_idx))]
    unseen_test_idx = np.load('./data/hiv_idx/cls/test_unseen_index.npy')

    print('model trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    print(f'Testing on {len(test_idx)} samples')
    print(f'(Unseen) Testing on {len(unseen_test_idx)} samples')

    unseen_test_loader = DataLoader(Subset(all_data, unseen_test_idx), batch_size=args.batch_size,collate_fn=HIV_Cls_collate)
    metrics = {metric: metrics_dict[metric] for metric in args.metrics}
    test_loader = DataLoader(Subset(all_data, test_idx), batch_size=args.batch_size,collate_fn=HIV_Cls_collate)

    trainer = get_trainer(args=args, model=model, data=all_data, device=device, metrics=metrics)
    test_metrics = trainer.evaluation(test_loader, data_split='test')
    unseen_test_metrics = trainer.evaluation(unseen_test_loader, data_split='unseen_test')
    
def test(args):
    seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    metrics_dict = {'mae': MAE(),
                    'custom': CustomL1Loss(),
                    'pearsonr': PearsonR(),
                    'rmse': RMSE(),
                    'spearman': SpearmanR(),
                    'acc': ClassificationAccuracy(),
                    'f1': F1(),
                    'rocauc': ROCAUC(),
                    'prauc': PRAUC(),
                    'mcc': MCC()
                    }
    print('using device: ', device)
    print(args.dataset)
    if args.dataset == 'IC50':
        return tsne(args, device, metrics_dict)
    
def test_ic50(args, device, metrics_dict):
    print('test_ic50')
    all_data = IC50Dataset(ic50_data = '/code/lyt/S3AI/data/test.csv')
    all_idx = get_random_indices(len(all_data), args.seed_data)
    test_idx = all_idx

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    model = load_infer_model(args, device=device)
    
    model.eval()
    print('model trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    print(f'Testing on {len(test_idx)} samples')
    test_loader = DataLoader(Subset(all_data, test_idx), batch_size=1,collate_fn=pad_collate)

    metrics = {metric: metrics_dict[metric] for metric in args.metrics}
    trainer = get_trainer(args=args, model=model, data=all_data, device=device, metrics=metrics)
    metrics = trainer.predict(test_loader, 1)
    test_score = metrics[trainer.main_metric]
    test_loss = metrics[type(trainer.loss_func).__name__]
    
    metrics_str = ', '.join([f'{key}: {value:.7f}' for key, value in metrics.items()])
    print('test loss:  %s: %.6f, loss: %.6f' % (trainer.main_metric, test_score, test_loss))
    print(f'test: {metrics_str}')
    return metrics

def tsne(args, device, metrics_dict):
    print('train_ic50')
    all_data = IC50Dataset()

    all_idx = get_random_indices(len(all_data), args.seed_data)
    model_idx = all_idx[:int(0.8 * len(all_data))]
    test_idx = all_idx[int(0.8 * len(all_data)):int(0.9 * len(all_data))]
    val_idx = all_idx[int(0.9 * len(all_data)):]
    train_idx = model_idx

    if args.num_val != None:
        train_idx = all_idx[:args.num_train]
        val_idx = all_idx[len(train_idx): len(train_idx) + args.num_val]
        test_idx = all_idx[len(train_idx) + args.num_val: len(train_idx) + 2*args.num_val]
        
    test_loader = DataLoader(Subset(all_data, test_idx), batch_size=args.batch_size,collate_fn=pad_collate)

    model = load_infer_model(args, device=device)
    print('model trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    print(f'Testing on {len(test_idx)} samples')

    metrics = {metric: metrics_dict[metric] for metric in args.metrics}
    trainer = get_trainer(args=args, model=model, data=all_data, device=device, metrics=metrics)
    trainer.model.eval()
    combined_representations = []
    class_labels = []
    for i, batch in enumerate(test_loader):
        batch = move_to_device(batch,device)
        loss, reg_pred, cls_pred, reg_label, cls_label,combined_representation = trainer.forward_pass(batch)
        combined_representations.append(combined_representation.detach())  # Detach from the current graph to avoid memory issues
        class_labels.append(cls_label.detach()) 
    # metrics = trainer.predict(test_loader, 1)
    # test_score = metrics[trainer.main_metric]
    # test_loss = metrics[type(trainer.loss_func).__name__]
    
    # metrics_str = ', '.join([f'{key}: {value:.7f}' for key, value in metrics.items()])
    # print('test loss:  %s: %.6f, loss: %.6f' % (trainer.main_metric, test_score, test_loss))
    # print(f'test: {metrics_str}')
    combined_representations_np = np.concatenate([c.detach().cpu().numpy() for c in combined_representations], axis=0)
    class_labels_np = np.concatenate([l.detach().cpu().numpy() for l in class_labels], axis=0)

    # Use numpy.savez to save both arrays in a single .npz file
    np.savez('test_data_no_inter_str.npz', combined_representations=combined_representations_np, class_labels=class_labels_np)

    print(f'Saved combined representations and class labels for {len(test_idx)} samples in test_data.npz.')
    return metrics

def get_arguments():
    args = parse_arguments()

    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    else:
        config_dict = {}
        
    return args


if __name__ == '__main__':
    args = get_arguments()
    # H_chain = 'QVQLVETGGGLIQPGGSLRLSCAASGFTVSSNYMSWVRQAPGKGLEWVSVIYSGGSTFYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDLERAGGMDVWGQGTMVTVSS'
    # L_chain = 'EIVMTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGSLYTFGQGTKVDIK'
    # A_chain = 'NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKST'
    test(args)
    

# EVQLVESGGGLVQPGGSLRLSCAASGFIVSSNYMSWVRQAPGKGLEWVSVIYSGGSTYYADSVKGRFTISRHNSKNTLYLQMNSLRAEDTAVYYCAREAYGMDVWGQGTTVTVSSDIVMTQSPSFLSASVGDRVTITCRASQGISSYLAWYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTEFTLTISSLQPEDFATYYCQQLNSYPPYTFGQGTKLEIK
# NITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLAPATVCGPKKST