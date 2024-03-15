import argparse
# turn on for debugging C code like Segmentation Faults
import faulthandler
import os

import yaml
from sklearn.svm import SVC, SVR
from torch.nn import *  # do not remove
from torch.optim import *  # do not remove
from torch.optim.lr_scheduler import *  # do not remove
from torch.utils.data import DataLoader, Subset

from commons.losses import *  
from commons.utils import (TENSORBOARD_FUNCTIONS, get_random_indices,
                           pad_collate, HIV_Cls_collate, HIV_Reg_collate,seed_all)
from datasets.ic50_dataset import IC50Dataset
from datasets.HIV_dataset import HIVClsDataset, HIVRegDataset
from models import * 
from trainers.IC50_trainer import IC50Trainer
from trainers.metrics import (ROCAUC, F1, MAE, RMSE, ClassificationAccuracy,
                              PearsonR, SpearmanR, MCC, PRAUC)
from trainers.Multitask_IC50_trainer import MultitaskIC50Trainer
from trainers.onehot_IC50_trainer import onehotIC50Trainer
from trainers.Cls_trainer import ClsTrainer
from trainers.Reg_trainer import RegTrainer
from trainers.onehot_Reg_trainer import OnehotRegTrainer
from trainers.parapred_trainer import ParapredTrainer
from trainers.trainer import Trainer
from trainers.onehot_parapred_reg_trainer import OnehotRegParapredTrainer

# import seaborn

faulthandler.enable()
import numpy as np


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

    p.add_argument('--collate_function', default='None', help='the collate function to use for DataLoader')
    p.add_argument('--collate_params', type=dict, default={},
                   help='parameters with keywords of the chosen collate function')
    p.add_argument('--device', type=str, default='cuda', help='What device to train on: cuda or cpu')

    p.add_argument('--models_to_save', type=list, default=[],
                   help='specify after which epochs to remember the best model')

    p.add_argument('--model_type', type=str, default='MultiTask_S3AI', help='Classname of one of the models in the models dir')
    p.add_argument('--model_parameters', type=dict, help='dictionary of model parameters')
    p.add_argument('--trainer', type=str, default='contrastive', help='Classname of one of the trainers in the trainers dir')
    p.add_argument('--train_sampler', type=str, default=None, help='any of pytorchs samplers or a custom sampler')

    p.add_argument('--eval_on_test', type=bool, default=False, help='runs evaluation on test set if true')
    p.add_argument('--OOD_test', type=bool, default=True, help='runs ODD evaluation on test set if true')
    return p.parse_args()


def get_trainer(args, model, data, device, metrics):
    tensorboard_functions = {function: TENSORBOARD_FUNCTIONS[function] for function in args.tensorboard_functions}
    trainer = globals().get(args.trainer, Trainer)
    return trainer(model=model, args=args, metrics=metrics, main_metric=args.main_metric,
                    main_metric_goal=args.main_metric_goal, optim=globals()[args.optimizer],
                    loss_func=globals()[args.loss_func](**args.loss_params), device=device,
                    tensorboard_functions=tensorboard_functions,
                    scheduler_step_per_batch=args.scheduler_step_per_batch)

def adjust_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = f"{key.replace('esm.encoder.', '')}"
        if 'layer.' in new_key:
            new_key = f"{new_key.replace('layer.','layers.')}"
        if 'attention.self.key' in new_key:
            new_key = f"{new_key.replace('attention.self.key','self_attn.k_proj')}"
        if 'attention.self.value' in new_key:
            new_key = f"{new_key.replace('attention.self.value','self_attn.v_proj')}"
        if 'attention.self.query' in new_key:
            new_key = f"{new_key.replace('attention.self.query','self_attn.q_proj')}"
        if 'attention.self.rotary_embeddings' in new_key:
            new_key = f"{new_key.replace('attention.self.rotary_embeddings','self_attn.rot_emb')}" 
        if 'attention.LayerNorm' in new_key:
            new_key = f"{new_key.replace('attention.LayerNorm','self_attn_layer_norm')}"
        if 'LayerNorm' in new_key:
            new_key = f"{new_key.replace('LayerNorm','final_layer_norm')}"
        if 'intermediate.dense' in new_key:
            new_key = f"{new_key.replace('intermediate.dense','fc1')}"
        if 'attention.output.dense' in new_key:
            new_key = f"{new_key.replace('attention.output.dense','self_attn.out_proj')}" # 
        if 'output.dense' in new_key:
            new_key = f"{new_key.replace('output.dense','fc2')}"
        new_state_dict[new_key] = value# 
    return new_state_dict

def load_model(args, data, device):
    """
    Load a model based on provided arguments and pre-trained parameters.

    Args:
        args (argparse.Namespace): Command line arguments specifying model type and parameters.
        data (object): Data object, possibly containing average degree information.
        device (torch.device): The device to which the model should be loaded.

    Returns:
        object: The loaded machine learning model.

    The fully configured and possibly pre-trained model is returned.
    """
    if "S3AI" not in args.model_type:
        model = globals()[args.model_type](
            **args.model_parameters
        )
    else:
        model = globals()[args.model_type](
            device=device,
            **args.model_parameters
        )
    if args.checkpoint:
        return model
    if args.pretrain_checkpoint_ab:
    # Define the paths for pre-trained parameters
        ab_pretrained_path = os.path.join(args.pretrain_checkpoint_ab, "Ab_pretrain.bin")
        print('load checkpoint from '+ ab_pretrained_path)
        ab_pretrained_params = torch.load(ab_pretrained_path, map_location=device)
        ab_pretrained_params = adjust_keys(ab_pretrained_params)
        model.Ab_encoder.model.load_state_dict(ab_pretrained_params, strict=False)
    if args.pretrain_checkpoint_ag:
        ag_pretrained_path = os.path.join(args.pretrain_checkpoint_ag, "Ag_pretrain.bin")
        print('load checkpoint from '+ ag_pretrained_path)
        ag_pretrained_params = torch.load(ag_pretrained_path, map_location=device)
        ag_pretrained_params = adjust_keys(ag_pretrained_params)
        model.Ag_encoder.model.load_state_dict(ag_pretrained_params, strict=False)
    if args.pretrain_checkpoint_str:
        str_pretrained_path = os.path.join(args.pretrain_checkpoint_str, "new_str.pt")
        print('load checkpoint from '+ str_pretrained_path)
        str_pretrained_params = torch.load(str_pretrained_path, map_location=device)
        model.MLP.load_state_dict(str_pretrained_params['model_state_dict'], strict=False)

    return model



def train(args):
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
    # print(args.dataset)
    if args.dataset == 'IC50':
        return train_ic50(args, device, metrics_dict)
    elif args.dataset == 'HIVcls':
        return train_HIV_cls(args, device, metrics_dict)
    elif args.dataset == 'HIVreg':
        return train_HIV_reg(args, device, metrics_dict)
    
    
def train_ic50(args, device, metrics_dict):
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

    model = load_model(args, data=all_data, device=device)
    print('model trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    print(f'Training on {len(train_idx)} samples from the model sequences')
    print(f'Validating on {len(val_idx)} samples')
    print(f'Testing on {len(test_idx)} samples')
    
    if args.train_sampler:
        sampler = globals()[args.train_sampler](data_source=all_data, batch_size=args.batch_size, indices=train_idx)
        train_loader = DataLoader(Subset(all_data, train_idx), batch_sampler=sampler,collate_fn=pad_collate)
    else:
        train_loader = DataLoader(Subset(all_data, train_idx), batch_size=args.batch_size, shuffle=True,collate_fn=pad_collate)
    val_loader = DataLoader(Subset(all_data, val_idx), batch_size=args.batch_size,collate_fn=pad_collate)
    test_loader = DataLoader(Subset(all_data, test_idx), batch_size=args.batch_size,collate_fn=pad_collate)

    metrics = {metric: metrics_dict[metric] for metric in args.metrics}
    
    trainer = get_trainer(args=args, model=model, data=all_data, device=device, metrics=metrics)
    
    if args.eval_on_test:
        print('running evaluation on test set, this might take a while...')
        test_metrics = trainer.evaluation(test_loader, data_split='test')
        return test_metrics, trainer.writer.log_dir
    val_metrics = trainer.train(train_loader, val_loader)

    return val_metrics

def train_HIV_cls(args, device, metrics_dict):
    print('train_HIV_cls')
    all_data = HIVClsDataset()
    if  args.OOD_test:
        all_idx = get_random_indices(np.load('./data/hiv_idx/cls/train_index.npy'))
        train_idx = all_idx[:int(0.8 * len(all_idx))]
        test_idx = all_idx[int(0.8 * len(all_idx)):int(0.9 * len(all_idx))]
        val_idx = all_idx[int(0.9 * len(all_idx)):]
        np.save('./data/hiv_idx/cls/train_index_shuffled.npy', all_idx)
        unseen_test_idx = np.load('./data/hiv_idx/cls/test_unseen_index.npy')
    else:
        all_idx = get_random_indices(len(all_data), args.seed_data)
        train_idx = all_idx[:int(0.8 * len(all_idx))]
        test_idx = all_idx[int(0.8 * len(all_idx)):int(0.9 * len(all_idx))]
        val_idx = all_idx[int(0.9 * len(all_idx)):]
        
    if args.num_val != None:
        train_idx = all_idx[:args.num_train]
        val_idx = all_idx[len(train_idx): len(train_idx) + args.num_val]
        test_idx = all_idx[len(train_idx) + args.num_val: len(train_idx) + 2*args.num_val]

    model = load_model(args, data=all_data, device=device)
    print('model trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    print(f'Training on {len(train_idx)} samples from the model sequences')
    print(f'Validating on {len(val_idx)} samples')
    print(f'Testing on {len(test_idx)} samples')
    if  args.OOD_test:
        print(f'(Unseen) Testing on {len(unseen_test_idx)} samples')
    
    if args.train_sampler:
        sampler = globals()[args.train_sampler](data_source=all_data, batch_size=args.batch_size, indices=train_idx)
        train_loader = DataLoader(Subset(all_data, train_idx), batch_sampler=sampler,collate_fn=HIV_Cls_collate)
    else:
        train_loader = DataLoader(Subset(all_data, train_idx), batch_size=args.batch_size, shuffle=True,collate_fn=HIV_Cls_collate)
    val_loader = DataLoader(Subset(all_data, val_idx), batch_size=args.batch_size,collate_fn=HIV_Cls_collate)
    test_loader = DataLoader(Subset(all_data, test_idx), batch_size=args.batch_size,collate_fn=HIV_Cls_collate)

    metrics = {metric: metrics_dict[metric] for metric in args.metrics}

    trainer = get_trainer(args=args, model=model, data=all_data, device=device, metrics=metrics)
    if args.eval_on_test:
        print('running evaluation on test set, this might take a while...')
        # test_metrics = trainer.evaluation(test_loader, data_split='test')
        if args.OOD_test:
            unseen_test_loader = DataLoader(Subset(all_data, unseen_test_idx), batch_size=args.batch_size,collate_fn=HIV_Cls_collate)
            unseen_test_metrics = trainer.evaluation(unseen_test_loader, data_split='unseen_test')
            return unseen_test_metrics
    
    
    val_metrics = trainer.train(train_loader, val_loader)
    test_metrics = trainer.evaluation(test_loader, data_split='test')
    if args.OOD_test:
        unseen_test_loader = DataLoader(Subset(all_data, unseen_test_idx), batch_size=args.batch_size,collate_fn=HIV_Cls_collate)
        unseen_test_metrics = trainer.evaluation(unseen_test_loader, data_split='unseen_test')
    return val_metrics


def train_HIV_reg(args, device, metrics_dict):
    print('train_HIV_reg')
    all_data = HIVRegDataset()

    if  args.OOD_test:
        all_idx = get_random_indices(np.load('./data/hiv_idx/reg/train_index.npy'))
        train_idx = all_idx[:int(0.8 * len(all_idx))]
        test_idx = all_idx[int(0.8 * len(all_idx)):int(0.9 * len(all_idx))]
        val_idx = all_idx[int(0.9 * len(all_idx)):]
        np.save('./data/hiv_idx/cls/train_index_shuffled.npy', all_idx)
        unseen_test_idx = np.load('./data/hiv_idx/reg/test_unseen_index.npy')
    else:
        all_idx = get_random_indices(len(all_data), args.seed_data)
        train_idx = all_idx[:int(0.8 * len(all_idx))]
        test_idx = all_idx[int(0.8 * len(all_idx)):int(0.9 * len(all_idx))]
        val_idx = all_idx[int(0.9 * len(all_idx)):]

    if args.num_val != None:
        train_idx = all_idx[:args.num_train]
        val_idx = all_idx[len(train_idx): len(train_idx) + args.num_val]
        test_idx = all_idx[len(train_idx) + args.num_val: len(train_idx) + 2*args.num_val]

    model = load_model(args, data=all_data, device=device)
    print('model trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    print(f'Training on {len(train_idx)} samples from the model sequences')
    print(f'Validating on {len(val_idx)} samples')
    print(f'Testing on {len(test_idx)} samples')
    if  args.OOD_test:
        print(f'(Unseen) Testing on {len(unseen_test_idx)} samples')
    
    if args.train_sampler:
        sampler = globals()[args.train_sampler](data_source=all_data, batch_size=args.batch_size, indices=train_idx)
        train_loader = DataLoader(Subset(all_data, train_idx), batch_sampler=sampler,collate_fn=HIV_Reg_collate)
    else:
        train_loader = DataLoader(Subset(all_data, train_idx), batch_size=args.batch_size, shuffle=True,collate_fn=HIV_Reg_collate)
    val_loader = DataLoader(Subset(all_data, val_idx), batch_size=args.batch_size,collate_fn=HIV_Reg_collate)
    test_loader = DataLoader(Subset(all_data, test_idx), batch_size=args.batch_size,collate_fn=HIV_Reg_collate)
    # unseen_test_loader = DataLoader(Subset(all_data, unseen_test_idx), batch_size=args.batch_size,collate_fn=HIV_Reg_collate)

    metrics = {metric: metrics_dict[metric] for metric in args.metrics}

    trainer = get_trainer(args=args, model=model, data=all_data, device=device, metrics=metrics)
    
    if args.eval_on_test: 
        print('running evaluation on test set, this might take a while...')
        # test_metrics = trainer.evaluation(test_loader, data_split='test')
        if args.OOD_test:
            unseen_test_loader = DataLoader(Subset(all_data, unseen_test_idx), batch_size=args.batch_size,collate_fn=HIV_Reg_collate)
            unseen_test_metrics = trainer.evaluation(unseen_test_loader, data_split='unseen_test')
            return unseen_test_metrics
    
    val_metrics = trainer.train(train_loader, val_loader)
    
    test_metrics = trainer.evaluation(test_loader, data_split='test')
    if args.OOD_test:
        unseen_test_loader = DataLoader(Subset(all_data, unseen_test_idx), batch_size=args.batch_size,collate_fn=HIV_Reg_collate)
        unseen_test_metrics = trainer.evaluation(unseen_test_loader, data_split='unseen_test')
    
    return val_metrics

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

    if args.checkpoint:  # overwrite args with args from checkpoint except for the args that were contained in the config file
        arg_dict = args.__dict__
        with open(os.path.join(os.path.dirname(args.checkpoint), 'train_arguments.yaml'), 'r') as arg_file:
            checkpoint_dict = yaml.load(arg_file, Loader=yaml.FullLoader)
        for key, value in checkpoint_dict.items():
            if key not in config_dict.keys():
                if isinstance(value, list):
                    for v in value:
                        arg_dict[key].append(v)
                else:
                    arg_dict[key] = value

    return args

if __name__ == '__main__':
    args = get_arguments()
    train(args)