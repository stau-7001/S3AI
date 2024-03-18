import argparse
import os
import re

from commons.utils import seed_all, get_random_indices, TENSORBOARD_FUNCTIONS, pad_collate

from trainers.IC50_trainer import IC50Trainer
from trainers.onehot_IC50_trainer import onehotIC50Trainer
from trainers.Multitask_IC50_trainer import MultitaskIC50Trainer

from datasets.ic50_dataset import IC50Dataset
from commons.utils import flatten_dict, tensorboard_gradient_magnitude, move_to_device
import yaml
from tqdm import tqdm

from models import *  # do not remove
from torch.nn import *  # do not remove
from torch.optim import *  # do not remove
from commons.losses import *  # do not remove
from torch.optim.lr_scheduler import *  # do not remove
from sklearn.svm import SVC, SVR

import random
from torch.utils.data import DataLoader, Subset

from trainers.metrics import  PearsonR,RMSE,SpearmanR,MAE,ClassificationAccuracy
from trainers.trainer import Trainer

from datasets.HIV_dataset import HIVClsDataset, HIVRegDataset
from main import get_trainer
def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs_clean/fingerprint_inference.yml')
    p.add_argument('--experiment_name', type=str, help='name that will be added to the runs folder output')
    p.add_argument('--logdir', type=str, default='runs', help='tensorboard logdirectory')
    p.add_argument('--indexdir', type=str, default='./shapley_data', help='Save directory for train index')
    p.add_argument('--seed', type=int, default=123, help='seed for reproducibility')
    p.add_argument('--seed_data', type=int, default=123, help='if you want to use a different seed for the datasplit')
    p.add_argument('--checkpoint', type=str, help='Specify path to finetune from a pretrained checkpoint')
    p.add_argument('--device', type=str, default='cuda', help='What device to train on: cuda or cpu')

    p.add_argument('--model_type', type=str, default='MPNN', help='Classname of one of the models in the models dir')

    # ---------------- Shapley_value ----------------
    p.add_argument('--target_layer', type=str, default='conv1', help='The layer of CNN module to calculate Shapley value')
    p.add_argument('--output_type', type=str, default='classification', help='The output type of model to calculate Shapley value')
    p.add_argument('--sample_st', type=int, default=0, help='The start node of samples to calculate Shapley value')
    p.add_argument('--sample_num', type=int, default=3000, help='The num of samples to calculate Shapley value')
    p.add_argument('--shapleydir', type=str, default='./shapley_saved', help='Save directory for Shapley value')
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

def generate_all_masks(length: int) -> list:
    masks = list(range(2**length))
    masks = [np.binary_repr(mask, width=length) for mask in masks]
    masks = [[bool(int(item)) for item in mask] for mask in masks]
    return masks

def get_mask_output(args, model, activation, Ab_seq_H, Ab_seq_L, CDR_H, CDR_L):
    # generate player
    device = activation.device
    activation = activation.cpu()
    combine_CDR =[(l,r) for l,r in CDR_H]
    Ab_seq_H_len = len(Ab_seq_H[0])
    for l,r in CDR_L:
        combine_CDR.append((l+Ab_seq_H_len,r+Ab_seq_H_len))
    print('CDR: ', combine_CDR)
    st = 0
    ed = activation.shape[3] - 1 
    player = []
    random.seed(args.seed)
    for l,r in combine_CDR:
        if st <= l-1:
            if l-st>=20:
                li = random.randint(10,15)
                player.append((st,st+li-1))
                player.append((st+li,l-1))
            else:
                player.append((st,l-1))
        st = r+1
    if st <= ed:
        player.append((st,ed))
    player.append((-1,-1))
    print('player:',player)
    # get mask output v of all the masks activation
    components_num = len(player)
    baseline = 0
    masks = torch.BoolTensor(generate_all_masks(components_num))
    v_S = []
    with torch.no_grad():
        m = masks[0]
        x_act = activation.clone().to(device)
        for i in range(components_num):
            if m[i] == False:
                if player[i][0]!=-1:
                    x_act[:,:,:,player[i][0]:player[i][1]+1] = 0
                else:
                    for cdr in combine_CDR:
                        x_act[:,:,:,cdr[0]:cdr[1]+1] = 0
        v_empty = model.feature2out(Ab_seq_H, Ab_seq_L, x_act, args.target_layer, args.output_type)
        for m in tqdm(masks):
            x_act = activation.clone().to(device)
            for i in range(components_num):
                if m[i] == False:
                    if player[i][0]!=-1:
                        x_act[:,:,:,player[i][0]:player[i][1]+1] = 0
                    else:
                        for cdr in combine_CDR:
                            x_act[:,:,:,cdr[0]:cdr[1]+1] = 0
            v_act = model.feature2out(Ab_seq_H, Ab_seq_L, x_act, args.target_layer, args.output_type)
            v_S.append((v_act-v_empty).cpu())
        
    # calculate shapley value
    phi = []
    for i in range(components_num):
        phi_i = 0.0
        for j, m in enumerate(masks):
            if m[i] == True: continue
            C_S = torch.sum(m)
            w_i = 1.0
            for k in range(1,components_num-C_S):
                w_i = w_i*k/(C_S+k)
            w_i = w_i/components_num
            phi_i += w_i*(v_S[j+(1<<(components_num-i-1))]-v_S[j])
        phi.append(phi_i.item())
    print('phi:',phi)
    return v_empty, phi, player


def Shapley(args, model, device, Ab_seq_H, Ab_seq_L, Ag_seq, CDR_H, CDR_L):
    with torch.no_grad():
        reg_pred, cls_pred, _ = model(Ab_seq_H, Ab_seq_L, Ag_seq)  # forward the sequence to the model
        activations = model.conv_module.activations
    activation = activations[args.target_layer]
    v_empty, phi, player = get_mask_output(args, model, activation, Ab_seq_H, Ab_seq_L, CDR_H, CDR_L)
    with torch.no_grad():
        cls_pred = model.feature2out(Ab_seq_H, Ab_seq_L, activation, args.target_layer, args.output_type)
    return v_empty, phi, player, cls_pred

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
    seed_all(args.seed)
    test_command = './AbRSA -i ab.fasta -k'
    all_data = IC50Dataset()

    all_idx = get_random_indices(len(all_data), args.seed_data)
    model_idx = all_idx[args.sample_st:args.sample_st+args.sample_num]
    train_idx = model_idx
    
    if not os.path.exists(args.indexdir):
        os.makedirs(args.indexdir)
    if not os.path.exists(args.shapleydir):
        os.makedirs(args.shapleydir)
    index_path = os.path.join(args.indexdir,'train_index_shuffled_full.npy')
    np.save(index_path, all_idx)
    
    train_loader = DataLoader(Subset(all_data, train_idx), batch_size=1, shuffle=False,collate_fn=pad_collate)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    model = load_infer_model(args, device=device)

    model.eval()

    v_empty_list, v_N_list, phi_list, player_list = [], [], [], []
    CDR_H_list, CDR_L_list = [], []
    cls_label_list, ic50_list = [], []
    Ab_seq_H_list, Ab_seq_L_list = [], [] 
    batch_idx = 0
    for i, batch in enumerate(tqdm(train_loader)):
        batch_idx = batch_idx + 1

        CDR_H = []
        CDR_L = []
        Ab_seq_H = batch['H']
        Ab_seq_L = batch['L']
        cls_label = batch['cls_label']
        ic50 = batch['IC50']
        Ag_seq = batch['spike']
        with open('ab.fasta','w') as f:
            f.write('>1-H\n')
            f.write(Ab_seq_H[0]+'\n')
            f.write('>1-L\n')
            f.write(Ab_seq_L[0]+'\n')
        with os.popen(test_command) as f:
            res = f.readlines()
        cdr_idx = 0
        for x in res: 
            if '_CDR' in x:
                cdr_idx +=1
                ri = x[8:-1]
                if cdr_idx <= 3:
                    cdr_l = Ab_seq_H[0].find(ri)+1
                    cdr_r = cdr_l+len(ri)-1
                    CDR_H.append((cdr_l,cdr_r))
                else:
                    cdr_l = Ab_seq_L[0].find(ri)+1
                    cdr_r = cdr_l+len(ri)-1
                    CDR_L.append((cdr_l,cdr_r))
        v_empty, phi, player, v_N = Shapley(args, model, device, Ab_seq_H, Ab_seq_L, Ag_seq, CDR_H, CDR_L)
        v_empty, v_N = v_empty.cpu().numpy(), v_N.cpu().numpy()

        v_empty_list.append(v_empty)
        v_N_list.append(v_N)
        phi_list.append(phi)
        player_list.append(player)
        CDR_H_list.append(CDR_H)
        CDR_L_list.append(CDR_L)
        ic50_list.append(ic50)
        cls_label_list.append(cls_label)
        Ab_seq_H_list.append(Ab_seq_H)
        Ab_seq_L_list.append(Ab_seq_L)
        if batch_idx%500 == 0:
            shapley_save_file = os.path.join(args.shapleydir,f'shapley_phi_st{args.sample_st}_ed{args.sample_st+batch_idx}.npz')
            np.savez(shapley_save_file,v_empty=v_empty_list, v_N=v_N_list, phi=phi_list, player=player_list, CDR_H=CDR_H_list,CDR_L=CDR_L_list, ic50=ic50_list, cls_label=cls_label_list, Ab_seq_H=Ab_seq_H_list, Ab_seq_L=Ab_seq_L_list)
    total_save_file = os.path.join(args.shapleydir,f'shapley_phi_st{args.sample_st}_samplenum{args.sample_num}.npz')
    np.savez(total_save_file,v_empty=v_empty_list, v_N=v_N_list, phi=phi_list, player=player_list, CDR_H=CDR_H_list, CDR_L=CDR_L_list, ic50=ic50_list, cls_label=cls_label_list, Ab_seq_H=Ab_seq_H_list, Ab_seq_L=Ab_seq_L_list)
