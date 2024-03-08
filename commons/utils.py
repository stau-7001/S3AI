import os
import random
from argparse import Namespace
from collections import MutableMapping
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
# import sklearn
import math
import torch
from scipy.stats import norm
from torch.utils.tensorboard import SummaryWriter

HydrogenBondAcceptor_Donor = {
    'A':[5.0,3.0],
    'V':[5.0,3.0],
    'L':[5.0,3.0],
    'I':[5.0,3.0],
    'P':[5.0,2.0],
    'S':[7.0,4.0],
    'T':[7.0,4.0],
    'M':[5.0,3.0],
    'E':[9.0,4.0],
    'Q':[8.0,5.0],
    'H':[7.0,4.0],
    'K':[6.0,5.0],
    'R':[8.0,7.0],
    'F':[5.0,3.0],
    'Y':[7.0,4.0],
    'W':[6.0,4.0],
    'D':[9.0,4.0],
    'N':[8.0,5.0],
    'C':[5.0,3.0],
    'G':[5.0,3.0],
}

Charged_Side_Chains = {
    'A':[0.],
    'V':[0.],
    'L':[0.],
    'I':[0.],
    'P':[0.],
    'S':[0.],
    'T':[0.],
    'M':[0.],
    'E':[1.0],
    'Q':[0.],
    'H':[-1.0],
    'K':[-1.0],
    'R':[-1.0],
    'F':[0.],
    'Y':[0.],
    'W':[0.],
    'D':[1.0],
    'N':[0.],
    'C':[0.],
    'G':[0.],
}

Hydropathy_Index = {
    'A':[0.62],
    'V':[1.1],
    'L':[1.1],
    'I':[1.4],
    'P':[0.12],
    'S':[-0.18],
    'T':[-0.05],
    'M':[0.64],
    'E':[-0.74],
    'Q':[-0.85],
    'H':[-0.40],
    'K':[-1.5],
    'R':[-2.5],
    'F':[1.2],
    'Y':[0.26],
    'W':[0.81],
    'D':[-0.9],
    'N':[-0.78],
    'C':[0.29],
    'G':[0.48],
}

Volume = {
    'A':[88.6],
    'V':[140.0],
    'L':[166.7],
    'I':[166.7],
    'P':[112.7],
    'S':[89.0],
    'T':[116.1],
    'M':[162.9],
    'E':[138.4],
    'Q':[143.8],
    'H':[153.2],
    'K':[168.6],
    'R':[173.4],
    'F':[189.9],
    'Y':[193.6],
    'W':[227.8],
    'D':[111.1],
    'N':[114.1],
    'C':[108.5],
    'G':[60.1],
}
# Zamyatnin, A.A., Protein volume in solution, Prog. Biophys. Mol. Biol., 24:107-123 (1972), PMID: 4566650.
def update_matrix(value):
    if isinstance(value, torch.Tensor):
        return value.item()  
    else:
        return value  

def generate_Chem_tensor(Abseq, Agseq):
    Abseq_len = len(Abseq)
    Agseq_len = len(Agseq)
    # print(Abseq,Agseq)
    tensor = np.zeros((Agseq_len, Abseq_len, 5))
    for i in range(Agseq_len):
        for j in range(Abseq_len):
            ab_aa = Abseq[j]
            ag_aa = Agseq[i]

            ab_prop = HydrogenBondAcceptor_Donor[ab_aa]
            ag_prop = HydrogenBondAcceptor_Donor[ag_aa]
            tensor[i, j, 0] = min(ab_prop[0], ag_prop[1])
            tensor[i, j, 1] = min(ab_prop[1], ag_prop[0])
            
            ab_prop = Charged_Side_Chains[ab_aa]
            ag_prop = Charged_Side_Chains[ag_aa]
            tensor[i, j, 2] = 0.5 * abs(ab_prop[0]- ag_prop[0])
            
            ab_prop = Hydropathy_Index[ab_aa]
            ag_prop = Hydropathy_Index[ag_aa]
            tensor[i, j, 3] = 1 - 0.25*abs(ab_prop[0]- ag_prop[0])
            
            ab_prop = Volume[ab_aa]
            ag_prop = Volume[ag_aa]
            tensor[i, j, 4] = math.exp(-((ab_prop[0] + ag_prop[0]) - 282.52)**2 / (2 * (55.54**2)))
    return torch.from_numpy(tensor).float()


def generate_Hbond_tensor(Abseq, Agseq):
    Abseq_len = len(Abseq)
    Agseq_len = len(Agseq)
    # print(Abseq,Agseq)

    tensor = np.zeros((Agseq_len, Abseq_len, 2))

    for i in range(Agseq_len):
        for j in range(Abseq_len):
            ab_aa = Abseq[j]
            ag_aa = Agseq[i]

            ab_prop = HydrogenBondAcceptor_Donor[ab_aa]
            ag_prop = HydrogenBondAcceptor_Donor[ag_aa]
            tensor[i, j, 0] = min(ab_prop[0], ag_prop[1])
            tensor[i, j, 1] = min(ab_prop[1], ag_prop[0])
            
    return torch.from_numpy(tensor).float()


def z_score_normalization(data_list, mean = 6.788316352970909,std = 4.333276584928288):
    data = np.array(data_list)
    normalized_data = (data - mean) / (std + 1e-8)
    return normalized_data.tolist()

def pad_collate(batch):
    H_seq = [sample['H'] for sample in batch]
    L_seq = [sample['L'] for sample in batch]
    ic50_values = [sample['IC50'] for sample in batch]
    # ic50 = z_score_normalization(ic50_values)
    ic50 = [np.log(sample['IC50']) for sample in batch]
    # ic50 = z_score_normalization(ic50_values)
    spike = [sample['spike'] for sample in batch]
    weight = [sample['weight'] for sample in batch]
    
    # 根据 IC50 值计算 cls_label
    # cls_label = [[0, 1] if ic50_value < 10 else [1, 0] for ic50_value in ic50_values]
    cls_label = [0. if ic50_value < 10 else 1. for ic50_value in ic50_values]
    # cls_label = [0 if ic50_value < 10 else 1 for ic50_value in ic50_values]

    return {'H': H_seq, 'L': L_seq, 'IC50': ic50, 'spike': spike, 'cls_label': cls_label, 'weight':weight }

def HIV_Cls_collate(batch):
    H_seq = [sample['H'] for sample in batch]
    L_seq = [sample['L'] for sample in batch]
    ic50_values = [sample['IC50'] for sample in batch]
    # ic50 = z_score_normalization(ic50_values)
    spike = [sample['spike'] for sample in batch]
    
    # 根据 IC50 值计算 cls_label
    # cls_label = [[0, 1] if ic50_value <= 0.5 else [1, 0] for ic50_value in ic50_values]
    cls_label = [0. if ic50_value < 0.5 else 1. for ic50_value in ic50_values]# 1 or 0

    return {'H': H_seq, 'L': L_seq, 'spike': spike, 'cls_label': cls_label}

def HIV_Reg_collate(batch):
    H_seq = [sample['H'] for sample in batch]
    L_seq = [sample['L'] for sample in batch]
    ic50_values = [sample['IC50'] for sample in batch]
    ic50 = [np.log(sample['IC50']) for sample in batch]
    spike = [sample['spike'] for sample in batch]
    
    # 根据 IC50 值计算 cls_label
    cls_label = [[0, 1] if ic50_value < 10 else [1, 0] for ic50_value in ic50_values]
    # cls_label = [0 if ic50_value < 10 else 1 for ic50_value in ic50_values]
    return {'H': H_seq, 'L': L_seq, 'IC50': ic50, 'spike': spike, 'cls_label': cls_label}


def seed_all(seed):
    if not seed:
        seed = 0

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def get_random_indices(length, seed=123):
    st0 = np.random.get_state()
    np.random.seed(seed)
    random_indices = np.random.permutation(length)
    np.random.set_state(st0)
    return random_indices

def flatten_dict(params: Dict[Any, Any], delimiter: str = '/') -> Dict[str, Any]:
    """
    Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a/b': 'c'}``.

    Args:
        params: Dictionary containing the hyperparameters
        delimiter: Delimiter to express the hierarchy. Defaults to ``'/'``.

    Returns:
        Flattened dict.
    Examples:
        flatten_dict({'a': {'b': 'c'}})
        {'a/b': 'c'}
        flatten_dict({'a': {'b': 123}})
        {'a/b': 123}
        flatten_dict({5: {'a': 123}})
        {'5/a': 123}
    """

    def _dict_generator(input_dict, prefixes=None):
        prefixes = prefixes[:] if prefixes else []
        if isinstance(input_dict, MutableMapping):
            for key, value in input_dict.items():
                key = str(key)
                if isinstance(value, (MutableMapping, Namespace)):
                    value = vars(value) if isinstance(value, Namespace) else value
                    for d in _dict_generator(value, prefixes + [key]):
                        yield d
                else:
                    yield prefixes + [key, value if value is not None else str(None)]
        else:
            yield prefixes + [input_dict if input_dict is None else str(input_dict)]

    dictionary = {delimiter.join(keys): val for *keys, val in _dict_generator(params)}
    for k in dictionary.keys():
        # convert relevant np scalars to python types first (instead of str)
        if isinstance(dictionary[k], (np.bool_, np.integer, np.floating)):
            dictionary[k] = dictionary[k].item()
        elif type(dictionary[k]) not in [bool, int, float, str, torch.Tensor]:
            dictionary[k] = str(dictionary[k])
    return dictionary

def tensorboard_singular_value_plot(predictions, targets, writer: SummaryWriter, step, data_split):
    u, s, v = torch.pca_lowrank(predictions.detach().cpu(), q=min(predictions.shape))
    fig, ax = plt.subplots()
    s = 100 * s / s.sum()
    ax.plot(s.numpy())
    writer.add_figure(f'singular_values/{data_split}', figure=fig, global_step=step)
    fig, ax = plt.subplots()
    ax.plot(np.cumsum(s.numpy()))
    writer.add_figure(f'singular_values_cumsum/{data_split}', figure=fig, global_step=step)


def tensorboard_gradient_magnitude(optimizer: torch.optim.Optimizer, writer: SummaryWriter, step, param_groups=[0]):
    for i, param_group in enumerate(optimizer.param_groups):
        if i in param_groups:
            all_params = []
            for params in param_group['params']:
                if params.grad != None:
                    all_params.append(params.grad.view(-1))
            writer.add_scalar(f'gradient_magnitude_param_group_{i}', torch.cat(all_params).abs().mean(),
                              global_step=step)


TENSORBOARD_FUNCTIONS = {
    'singular_values': tensorboard_singular_value_plot
}

def move_to_device(element, device):
    if isinstance(element, list):
        return [move_to_device(x, device) for x in element]
    else:
        return element.to(device) if isinstance(element, torch.Tensor) else element

