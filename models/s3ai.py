from functools import partial
from typing import Callable, Dict, List, Union

import esm
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from commons.utils import generate_Chem_tensor, generate_Hbond_tensor
from models.AbAgAttention import BidirectionalResidueAttention
from models.base_layers import MLP



EPS = 1e-5
def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def attention_weighted_feature(feature, weights, alpha=0.8):
    # Normalize the weights using Softmax
    # Normalize the weights using Softmax along the last dimension (axis=-1)
    normalized_weights = F.softmax(*weights, dim=-1)

    # Element-wise multiplication to obtain the weighted feature
    expanded_weights = normalized_weights.unsqueeze(0).unsqueeze(-1)
    device = feature.device
    weighted_feature = feature * expanded_weights.to(device)

    # Linearly combine the original feature and the weighted feature based on alpha
    final_feature = alpha * feature + (1 - alpha) * weighted_feature

    return final_feature

def fixed_attention_weighted_feature(feature, weights, alpha=0.8):
    # Normalize the weights using Softmax
    # Normalize the weights using Softmax along the last dimension (axis=-1)
    normalized_weights = F.softmax(alpha * torch.ones_like(weights) + (1 - alpha) * weights)

    # Element-wise multiplication to obtain the weighted feature
    expanded_weights = normalized_weights.unsqueeze(0).unsqueeze(-1)
    device = feature.device
    weighted_feature = feature * expanded_weights.to(device)

    return weighted_feature


class ESM_encoder(nn.Module):
    def __init__(self, device, **kwargs):
        super(ESM_encoder, self).__init__()
        self.device = device
        self.model, self.alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model = self.model.to(self.device)

    def forward(self, data_H, data_L):
        tmp = [('', data_H[i]+data_L[i]) for i in range(len(data_H))]
        data = tmp
        _, _, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[30], return_contacts=True)
        return results["representations"][30], batch_lens  # shape: (batch_size, max_seq_len, esm_hidden_dim=640), (batch_size,)

class MultiTask_S3AI(nn.Module):
    '''
        :param str_hidden_dim
        :param alpha
    '''
    def __init__(self, device, esm_hidden_dim=640, str_hidden_dim = 64, alpha = 0.8, **kwargs):
        super(MultiTask_S3AI, self).__init__()
        self.device = device
        self.Ab_encoder = ESM_encoder(device)
        self.Ag_encoder = ESM_encoder(device)
#         self.MLP = nn.Linear(esm_hidden_dim, str_hidden_dim)
        self.MLP = MLP(in_dim=640, hidden_size=640,
                          mid_batch_norm=False, out_dim=str_hidden_dim,
                          layers=4, batch_norm_momentum=0.1,device = device).to(device)
        self.Bi_Att = BidirectionalResidueAttention(esm_hidden_dim).to(device)
        self.cls_dim = esm_hidden_dim+str_hidden_dim
#         self.regression_head = RegressionHead(esm_hidden_dim+str_hidden_dim, 1)
#         self.classification_head = ClassificationHead(esm_hidden_dim+str_hidden_dim, 2)
        self.classification_head = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim, self.cls_dim // 4),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 4),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 64),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),
            
            nn.Linear(self.cls_dim // 64, 2),
            
            nn.Softmax(dim=1)).to(device)

        self.regression_head = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim, self.cls_dim // 4),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 4),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 64),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),
            
            nn.Linear(self.cls_dim // 64, 1)).to(device)
        self.alpha = alpha


    def forward(self, Abseq_H, Abseq_L, Agseq):
        Zab, Ab_batch_lens = self.Ab_encoder(Abseq_H, Abseq_L)  # shape: (batch_size, Abseq_len, esm_hidden_dim=640), (batch_size,)
        Zag, Ag_batch_lens = self.Ag_encoder(Agseq, ['']*len(Agseq))  # shape: (batch_size, Agseq_len, esm_hidden_dim=640), (batch_size,)
#         Zag = fixed_attention_weighted_feature(Zag, *weight,self.alpha)
        # print(Zag.device)
        att_data, Att_ab_ag = self.Bi_Att(Zab, Zag,True)  # shape: (batch_size, Abseq_len, esm_hidden_dim=640)
        
        Str_Zab = self.MLP(Zab)  # shape: (batch_size, seq_len, str_hidden_dim)

        sequence_representations_int = []
        for i, tokens_len in enumerate(Ab_batch_lens):
            sequence_representations_int.append(Att_ab_ag[i, 1:tokens_len - 1].mean(0))  # [esm_hidden_dim]

        x_int = torch.stack(sequence_representations_int, dim=0)  # shape: (batch_size, esm_hidden_dim)

        sequence_representations_str = []
        for i, tokens_len in enumerate(Ab_batch_lens):
            sequence_representations_str.append(Str_Zab[i, 1:tokens_len - 1].mean(0))  # [str_hidden_dim]

        x_str = torch.stack(sequence_representations_str, dim=0)  # shape: (batch_size, str_hidden_dim)

        x_reg_final = self.regression_head(torch.cat((x_str, x_int), dim=1))  # shape: (batch_size, 1)
        x_cls_final = self.classification_head(torch.cat((x_str, x_int), dim=1))  # shape: (batch_size, 1)

        return x_reg_final, x_cls_final


class CNNModule(nn.Module):
    def __init__(self, max_Agseq_len_rbd, max_Abseq_len, esm_hidden_dim):
        super(CNNModule, self).__init__()
        
        # Define your CNN layers here
        self.conv1 = nn.Conv2d(in_channels=esm_hidden_dim, out_channels=esm_hidden_dim*2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=esm_hidden_dim*2, out_channels=esm_hidden_dim*2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=esm_hidden_dim*2, out_channels=esm_hidden_dim, kernel_size=3, padding=1)
        
        # Define batch normalization layers
        self.bn0 = nn.LayerNorm([max_Agseq_len_rbd, max_Abseq_len])
        self.bn1 = nn.LayerNorm([max_Agseq_len_rbd, max_Abseq_len])
        self.bn2 = nn.LayerNorm([max_Agseq_len_rbd, max_Abseq_len])
        self.bn3 = nn.LayerNorm([max_Agseq_len_rbd, max_Abseq_len])
        
        # Define ELU activation function with adaptive scaling
        self.elu = nn.ELU(True)
    
    def feature2out(self, x, target_layer:str):
        if target_layer == 'input':
            x = self.bn0(x)
            x = self.bn1(self.elu(self.conv1(x)))
            x = self.bn2(self.elu(self.conv2(x)))
            x = self.bn3(self.elu(self.conv3(x)))
        elif target_layer == 'input_bn':
            x = self.bn1(self.elu(self.conv1(x)))
            x = self.bn2(self.elu(self.conv2(x)))
            x = self.bn3(self.elu(self.conv3(x)))
        elif target_layer == 'conv1':
            x = self.bn2(self.elu(self.conv2(x)))
            x = self.bn3(self.elu(self.conv3(x)))
        elif target_layer == 'conv2':
            x = self.bn3(self.elu(self.conv3(x)))
        x = torch.mean(x, dim=[2, 3])
        return x


    def forward(self, x):
    #     x = x.permute(0, 3, 1, 2)
        
    #     # Apply convolutional layers with ELU activation and batch normalization
    #     x = self.bn0(x)
    #     x = self.bn1(self.elu(self.conv1(x)))
    #     x = self.bn2(self.elu(self.conv2(x)))
    #     x = self.bn3(self.elu(self.conv3(x)))
        
    #     # Global average pooling
    #     x = torch.mean(x, dim=[2, 3])
        self.activations = {}
        
        x = x.permute(0, 3, 1, 2)
        self.activations['x'] = x
        self.activations['input'] = x
        x = self.bn0(x)
        self.activations['input_bn'] = x
        self.activations['conv1'] = self.bn1(self.elu(self.conv1(x)))
        x = self.activations['conv1']
        self.activations['conv2'] = self.bn2(self.elu(self.conv2(x)))
        x = self.activations['conv2']
        self.activations['conv3'] = self.bn3(self.elu(self.conv3(x)))
        x = self.activations['conv3']
        
        # Global average pooling
        x = torch.mean(x, dim=[2, 3])
        
        return x


class CNNModule_light(nn.Module):
    def __init__(self, max_Agseq_len_rbd, max_Abseq_len, input_hidden_dim = 2):
        super(CNNModule_light, self).__init__()
        
        # Define your CNN layers here
        self.conv1 = nn.Conv2d(in_channels=input_hidden_dim, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        
        # Define batch normalization layers
        self.bn0 = nn.LayerNorm([max_Agseq_len_rbd, max_Abseq_len])
        self.bn1 = nn.LayerNorm([max_Agseq_len_rbd, max_Abseq_len])
        self.bn2 = nn.LayerNorm([max_Agseq_len_rbd, max_Abseq_len])
        self.bn3 = nn.LayerNorm([max_Agseq_len_rbd, max_Abseq_len])
        
        # Define ELU activation function with adaptive scaling
        self.elu = nn.ELU(True)
        
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        
        # Apply convolutional layers with ELU activation and batch normalization
        x = self.bn0(x)
        x = self.bn1(self.elu(self.conv1(x)))
        x = self.bn2(self.elu(self.conv2(x)))
        x = self.bn3(self.elu(self.conv3(x)))
        
        # Global average pooling
        x = torch.mean(x, dim=[2, 3])
        
        return x
    
class MultiTask_S3AI_INTER(nn.Module):
    def __init__(self, device, esm_hidden_dim=640, str_hidden_dim = 64, alpha = 0.8, max_Ab_len = 230, max_Ag_len = 204, **kwargs):
        super(MultiTask_S3AI_INTER, self).__init__()
        self.device = device
        self.max_Ab_len = max_Ab_len
        self.max_Ag_len = max_Ag_len
        self.Ab_encoder = ESM_encoder(device)
        self.Ag_encoder = ESM_encoder(device)
#         self.MLP = nn.Linear(esm_hidden_dim, str_hidden_dim)
        self.MLP = MLP(in_dim=640, hidden_size=640,
                          mid_batch_norm=False, out_dim=str_hidden_dim,
                          layers=4, batch_norm_momentum=0.1,device = device).to(device)
        self.conv_module = CNNModule(self.max_Ag_len, self.max_Ab_len, esm_hidden_dim).to(device)

        
        self.cls_dim = esm_hidden_dim+str_hidden_dim
        self.classification_head = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim, self.cls_dim // 4),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), ß#nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 4),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 64),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),
            
            nn.Linear(self.cls_dim // 64, 2),
            
            nn.Softmax(dim=1)).to(device)

        self.regression_head = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim, self.cls_dim // 4),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 4),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 64),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),
            
            nn.Linear(self.cls_dim // 64, 1)).to(device)
        self.alpha = alpha


    def forward(self, Abseq_H, Abseq_L, Agseq):
        Zab, Ab_batch_lens = self.Ab_encoder(Abseq_H, Abseq_L)  # shape: (batch_size, Abseq_len, esm_hidden_dim=640), (batch_size,)
        Zag, Ag_batch_lens = self.Ag_encoder(Agseq, ['']*len(Agseq))  # shape: (batch_size, Agseq_len, esm_hidden_dim=640), (batch_size,)

        Str_Zab = self.MLP(Zab)  # shape: (batch_size, seq_len, str_hidden_dim)
        RBD_start = Agseq[0].find('NITN')
        RBD_end = Agseq[0].find('KKST')
        inter_batch = []
        for i, tokens_len in enumerate(Ab_batch_lens):
            Zag_inter = Zag[i, RBD_start :RBD_end + 1].unsqueeze(1)  # [batch_size, Agseq_len_rbd, 1 esm_hidden_dim]
            Zab_inter = Zab[i, 1:tokens_len - 1].unsqueeze(0)  # [batch_size, 1, Abseq_len, esm_hidden_dim]
            # print(Zab_inter.shape,Zag_inter.shape)
            inter_matrix = Zag_inter*Zab_inter
            l1_padding = self.max_Ag_len - inter_matrix.shape[0]
            l2_padding = self.max_Ab_len - inter_matrix.shape[1]
            inter_batch.append(F.pad(inter_matrix.unsqueeze(0), (0,0,1,l2_padding-1,1,l1_padding-1), "constant", 0).squeeze(0))
        x_inter = torch.stack(inter_batch, dim=0) # [batch_size, max_Agseq_len_rbd, max_Abseq_len, esm_hidden_dim]
        # print(x_inter.shape)
        x_inter = self.conv_module(x_inter)
        


        sequence_representations_str = []
        for i, tokens_len in enumerate(Ab_batch_lens):
            sequence_representations_str.append(Str_Zab[i, 1:tokens_len - 1].mean(0))  # [str_hidden_dim]

        x_str = torch.stack(sequence_representations_str, dim=0)  # shape: (batch_size, str_hidden_dim)

        x_reg_final = self.regression_head(torch.cat((x_str, x_inter), dim=1))  # shape: (batch_size, 1)
        x_cls_final = self.classification_head(torch.cat((x_str, x_inter), dim=1))  # shape: (batch_size, 1)

        return x_reg_final, x_cls_final
    

class MultiTask_S3AI_INTER_Hbond(nn.Module):
    def __init__(self, device, esm_hidden_dim=640, str_hidden_dim = 64, alpha = 0.8, max_Ab_len = 230, max_Ag_len = 204, virus_name = 'HIV', **kwargs):
        super(MultiTask_S3AI_INTER_Hbond, self).__init__()
        self.device = device
        self.max_Ab_len = max_Ab_len
        self.max_Ag_len = max_Ag_len
        self.virus_name = virus_name 
        self.Ab_encoder = ESM_encoder(device)
        self.Ag_encoder = ESM_encoder(device)
#         self.MLP = nn.Linear(esm_hidden_dim, str_hidden_dim)
        self.MLP = MLP(in_dim=640, hidden_size=640,
                          mid_batch_norm=False, out_dim=str_hidden_dim,
                          layers=4, batch_norm_momentum=0.1,device = device).to(device)
        self.conv_module = CNNModule(self.max_Ag_len, self.max_Ab_len, esm_hidden_dim+2).to(device)

        
        self.cls_dim = esm_hidden_dim + str_hidden_dim + 2
        self.classification_head = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim, self.cls_dim // 4),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), ß#nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 4),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 64),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),
            
            nn.Linear(self.cls_dim // 64, 2),
            
            nn.Softmax(dim=1)).to(device)

        self.regression_head = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim, self.cls_dim // 4),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 4),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 64),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),
            
            nn.Linear(self.cls_dim // 64, 1)).to(device)
        self.alpha = alpha


    def forward(self, Abseq_H, Abseq_L, Agseq):
        Zab, Ab_batch_lens = self.Ab_encoder(Abseq_H, Abseq_L)  # shape: (batch_size, Abseq_len, esm_hidden_dim=640), (batch_size,)
        Zag, Ag_batch_lens = self.Ag_encoder(Agseq, ['']*len(Agseq))  # shape: (batch_size, Agseq_len, esm_hidden_dim=640), (batch_size,)

        Str_Zab = self.MLP(Zab)  # shape: (batch_size, seq_len, str_hidden_dim)

        inter_batch = []
        for i, tokens_len in enumerate(Ab_batch_lens):
            RBD_start = Agseq[i].find('NITN')
            RBD_end = Agseq[i].find('KKST')
            Zag_inter = Zag[i, RBD_start :RBD_end + 1].unsqueeze(1)  # [Agseq_len_rbd, 1 esm_hidden_dim]
            Zab_inter = Zab[i, 1:tokens_len - 1].unsqueeze(0)  # [ 1, Abseq_len, esm_hidden_dim]
            # print(Zab_inter.shape,Zag_inter.shape)
            inter_matrix = Zag_inter*Zab_inter
            # print(inter_matrix.shape)
            Hbond_matrix = generate_Hbond_tensor(Abseq_H[i]+Abseq_L[i],Agseq[i][RBD_start :RBD_end + 1]).to(self.device)
            # print(Hbond_matrix.shape)
            inter_matrix = torch.cat([inter_matrix,Hbond_matrix], dim=2)
            # print(inter_matrix.shape)
            
            l1_padding = self.max_Ag_len - inter_matrix.shape[0]
            l2_padding = self.max_Ab_len - inter_matrix.shape[1]
            inter_batch.append(F.pad(inter_matrix.unsqueeze(0), (0,0,1,l2_padding-1,1,l1_padding-1), "constant", 0).squeeze(0))
        x_inter = torch.stack(inter_batch, dim=0) # [batch_size, max_Agseq_len_rbd, max_Abseq_len, esm_hidden_dim]
        # print(x_inter.shape)
        x_inter = self.conv_module(x_inter)
        


        sequence_representations_str = []
        for i, tokens_len in enumerate(Ab_batch_lens):
            sequence_representations_str.append(Str_Zab[i, 1:tokens_len - 1].mean(0))  # [str_hidden_dim]

        x_str = torch.stack(sequence_representations_str, dim=0)  # shape: (batch_size, str_hidden_dim)

        x_reg_final = self.regression_head(torch.cat((x_str, x_inter), dim=1))  # shape: (batch_size, 1)
        x_cls_final = self.classification_head(torch.cat((x_str, x_inter), dim=1))  # shape: (batch_size, 1)

        return x_reg_final, x_cls_final


class MultiTask_S3AI_Chem(nn.Module):
    def __init__(self, device, esm_hidden_dim=640, str_hidden_dim = 64, alpha = 0.8, max_Ab_len = 250, max_Ag_len = 204,virus_name = 'SARSCOV2' ,**kwargs):
        super(MultiTask_S3AI_Chem, self).__init__()
        self.device = device
        self.max_Ab_len = max_Ab_len
        self.max_Ag_len = max_Ag_len
        self.virus_name = virus_name
        self.Ab_encoder = ESM_encoder(device)
        self.Ag_encoder = ESM_encoder(device)
#         self.MLP = nn.Linear(esm_hidden_dim, str_hidden_dim)
        self.MLP = MLP(in_dim=640, hidden_size=640,
                          mid_batch_norm=False, out_dim=str_hidden_dim,
                          layers=4, batch_norm_momentum=0.1,device = device).to(device)
        self.conv_module = CNNModule(self.max_Ag_len, self.max_Ab_len, esm_hidden_dim+5).to(device)

        
        self.cls_dim = esm_hidden_dim + str_hidden_dim + 5
        self.classification_head = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim, self.cls_dim // 4),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), ß#nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 4),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 64),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),
            
            nn.Linear(self.cls_dim // 64, 1),
            
            nn.Sigmoid()).to(device)

        self.regression_head = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim, self.cls_dim // 4),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 4),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 64),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),
            
            nn.Linear(self.cls_dim // 64, 1)).to(device)
        self.alpha = alpha

    def feature2out(self, Abseq_H, Abseq_L, activation, target_layer, output_type):
        Zab, Ab_batch_lens = self.Ab_encoder(Abseq_H, Abseq_L)  # shape: (batch_size, Abseq_len, esm_hidden_dim=640), (batch_size,)
        Str_Zab = self.MLP(Zab)  # shape: (batch_size, seq_len, str_hidden_dim)

        x_inter = self.conv_module.feature2out(activation, target_layer)
        sequence_representations_str = []
        for i, tokens_len in enumerate(Ab_batch_lens):
            sequence_representations_str.append(Str_Zab[i, 1:tokens_len - 1].mean(0))  # [str_hidden_dim]

        x_str = torch.stack(sequence_representations_str, dim=0)  # shape: (batch_size, str_hidden_dim)

        x = torch.cat((x_str, x_inter), dim=1)
        if output_type == 'classification':
            x_cls_final = self.classification_head(x)  # shape: (batch_size, 1)
            re = x_cls_final[0,0]
        elif output_type == 'regression':
            x_reg_final = self.regression_head(x)
            re = x_reg_final
        return re

    def forward(self, Abseq_H, Abseq_L, Agseq):
        Zab, Ab_batch_lens = self.Ab_encoder(Abseq_H, Abseq_L)  # shape: (batch_size, Abseq_len, esm_hidden_dim=640), (batch_size,)
        Zag, Ag_batch_lens = self.Ag_encoder(Agseq, ['']*len(Agseq))  # shape: (batch_size, Agseq_len, esm_hidden_dim=640), (batch_size,)

        Str_Zab = self.MLP(Zab)  # shape: (batch_size, seq_len, str_hidden_dim)

        inter_batch = []
        for i, tokens_len in enumerate(Ab_batch_lens):
            if self.virus_name == 'HIV':
                RBD_start = Agseq[i].find('GVP') - 8
                RBD_end = max(Agseq[i].find('APT')+15, Agseq[i].find('YKVV')+25)
            else:
                RBD_start = Agseq[i].find('NITN')
                RBD_end = Agseq[i].find('KKST') + 4
            Zag_inter = Zag[i, RBD_start :RBD_end + 1].unsqueeze(1)  # [Agseq_len_rbd, 1 esm_hidden_dim]
            Zab_inter = Zab[i, 1:tokens_len - 1].unsqueeze(0)  # [ 1, Abseq_len, esm_hidden_dim]
            # print(Zab_inter.shape,Zag_inter.shape)
            inter_matrix = Zag_inter*Zab_inter
            # print(inter_matrix.shape)
            Chem_matrix = generate_Chem_tensor(Abseq_H[i]+Abseq_L[i],Agseq[i][RBD_start :RBD_end + 1]).to(self.device)
            # hydropathy_matrix = generate_hydropathy_tensor(Abseq_H[i]+Abseq_L[i],Agseq[i][RBD_start :RBD_end + 1]).to(self.device)
            # print(Hbond_matrix.shape)
            # inter_matrix = torch.cat([inter_matrix,Hbond_matrix,hydropathy_matrix], dim=2)
            inter_matrix = torch.cat([inter_matrix,Chem_matrix], dim=2)
            # print(inter_matrix.shape)
            
            l1_padding = self.max_Ag_len - inter_matrix.shape[0]
            l2_padding = self.max_Ab_len - inter_matrix.shape[1]
            inter_batch.append(F.pad(inter_matrix.unsqueeze(0), (0,0,1,l2_padding-1,1,l1_padding-1), "constant", 0).squeeze(0))
        x_inter = torch.stack(inter_batch, dim=0) # [batch_size, max_Agseq_len_rbd, max_Abseq_len, esm_hidden_dim]
        # print(x_inter.shape)
        x_inter = self.conv_module(x_inter)
        pooled_Str_Zab = [Str_Zab[i, 1:tokens_len - 1].mean(0) for i, tokens_len in enumerate(Ab_batch_lens)]
        x_str = torch.stack(pooled_Str_Zab, dim=0)  # shape: (batch_size, str_hidden_dim)

        combined_representation = torch.cat((x_str, x_inter), dim=1)

        x_reg_final = self.regression_head(combined_representation)  # shape: (batch_size, 1)
        x_cls_final = self.classification_head(combined_representation)  # shape: (batch_size, 1)

        return x_reg_final, x_cls_final, combined_representation

class MultiTask_S3AI_Chem_HIV(nn.Module):
    def __init__(self, device, esm_hidden_dim=640, str_hidden_dim = 64, alpha = 0.8, max_Ab_len = 230, max_Ag_len = 700, virus_name = 'HIV', **kwargs):
        super(MultiTask_S3AI_Chem_HIV, self).__init__()
        self.device = device
        self.max_Ab_len = max_Ab_len
        self.max_Ag_len = max_Ag_len
        self.virus_name = virus_name
        self.Ab_encoder = ESM_encoder(device)
        self.Ag_encoder = ESM_encoder(device)
#         self.MLP = nn.Linear(esm_hidden_dim, str_hidden_dim)
        self.MLP = MLP(in_dim=640, hidden_size=640,
                          mid_batch_norm=False, out_dim=str_hidden_dim,
                          layers=4, batch_norm_momentum=0.1,device = device).to(device)
        self.conv_module = CNNModule(self.max_Ag_len, self.max_Ab_len, esm_hidden_dim+5).to(device)

        
        self.cls_dim = esm_hidden_dim + str_hidden_dim + 5
        self.classification_head = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim, self.cls_dim // 4),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), ß#nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 4),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 64),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),
            
            nn.Linear(self.cls_dim // 64, 1),
            
            nn.Sigmoid()).to(device)
        
        self.regression_head = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim, self.cls_dim // 4),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 4),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 64),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),
            
            nn.Linear(self.cls_dim // 64, 1)).to(device)
        self.alpha = alpha


    def forward(self, Abseq_H, Abseq_L, Agseq):
        Zab, Ab_batch_lens = self.Ab_encoder(Abseq_H, Abseq_L)  # shape: (batch_size, Abseq_len, esm_hidden_dim=640), (batch_size,)
        Zag, Ag_batch_lens = self.Ag_encoder(Agseq, ['']*len(Agseq))  # shape: (batch_size, Agseq_len, esm_hidden_dim=640), (batch_size,)

        Str_Zab = self.MLP(Zab)  # shape: (batch_size, seq_len, str_hidden_dim)

        inter_batch = []
        for i, tokens_len in enumerate(Ab_batch_lens):
            if self.virus_name == 'HIV':
                RBD_start = Agseq[i].find('GVP') - 8
                RBD_end = max(Agseq[i].find('APT')+15, Agseq[i].find('YKVV')+25)
            else:
                RBD_start = Agseq[i].find('NITN')
                RBD_end = Agseq[i].find('KKST') + 4
            Zag_inter = Zag[i, RBD_start :RBD_end + 1].unsqueeze(1)  # [Agseq_len_rbd, 1 esm_hidden_dim]
            Zab_inter = Zab[i, 1:tokens_len - 1].unsqueeze(0)  # [ 1, Abseq_len, esm_hidden_dim]
            inter_matrix = Zag_inter*Zab_inter
            Chem_matrix = generate_Chem_tensor(Abseq_H[i].replace("X", "G")+Abseq_L[i].replace("X", "G"),Agseq[i][RBD_start :RBD_end + 1].replace("X", "G")).to(self.device)
            inter_matrix = torch.cat([inter_matrix,Chem_matrix], dim=2)
            
            l1_padding = self.max_Ag_len - inter_matrix.shape[0]
            l2_padding = self.max_Ab_len - inter_matrix.shape[1]
            inter_batch.append(F.pad(inter_matrix.unsqueeze(0), (0,0,1,l2_padding-1,1,l1_padding-1), "constant", 0).squeeze(0))
        x_inter = torch.stack(inter_batch, dim=0) # [batch_size, max_Agseq_len_rbd, max_Abseq_len, esm_hidden_dim]
        # print(x_inter.shape)
        x_inter = self.conv_module(x_inter)
        pooled_Str_Zab = [Str_Zab[i, 1:tokens_len - 1].mean(0) for i, tokens_len in enumerate(Ab_batch_lens)]
        x_str = torch.stack(pooled_Str_Zab, dim=0)  # shape: (batch_size, str_hidden_dim)


        # sequence_representations_str = []
        # for i, tokens_len in enumerate(Ab_batch_lens):
        #     sequence_representations_str.append(Str_Zab[i, 1:tokens_len - 1].mean(0))  # [str_hidden_dim]

        # x_str = torch.stack(sequence_representations_str, dim=0)  # shape: (batch_size, str_hidden_dim)
        combined_representation = torch.cat((x_str, x_inter), dim=1)

        x_reg_final = self.regression_head(combined_representation)  # shape: (batch_size, 1)
        x_cls_final = self.classification_head(combined_representation)  # shape: (batch_size, 1)

        return x_reg_final, x_cls_final, combined_representation


class Cls_S3AI_Chem(nn.Module):
    def __init__(self, device, esm_hidden_dim=640, str_hidden_dim = 64, alpha = 0.8, max_Ab_len = 230, max_Ag_len = 700, virus_name = 'HIV',**kwargs):
        super(Cls_S3AI_Chem, self).__init__()
        self.device = device
        self.max_Ab_len = max_Ab_len
        self.max_Ag_len = max_Ag_len
        self.virus_name = virus_name
        self.Ab_encoder = ESM_encoder(device)
        self.Ag_encoder = ESM_encoder(device)
#         self.MLP = nn.Linear(esm_hidden_dim, str_hidden_dim)
        self.MLP = MLP(in_dim=640, hidden_size=640,
                          mid_batch_norm=False, out_dim=str_hidden_dim,
                          layers=4, batch_norm_momentum=0.1,device = device).to(device)
        self.conv_module = CNNModule(self.max_Ag_len, self.max_Ab_len, esm_hidden_dim+5).to(device)

        
        self.cls_dim = esm_hidden_dim + str_hidden_dim + 5
        self.classification_head = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim, self.cls_dim // 4),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), ß#nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 4),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 64),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),
            
            nn.Linear(self.cls_dim // 64, 1),
            
            nn.Sigmoid()).to(device)
        self.alpha = alpha


    def forward(self, Abseq_H, Abseq_L, Agseq):
        Zab, Ab_batch_lens = self.Ab_encoder(Abseq_H, Abseq_L)  # shape: (batch_size, Abseq_len, esm_hidden_dim=640), (batch_size,)
        Zag, Ag_batch_lens = self.Ag_encoder(Agseq, ['']*len(Agseq))  # shape: (batch_size, Agseq_len, esm_hidden_dim=640), (batch_size,)

        Str_Zab = self.MLP(Zab)  # shape: (batch_size, seq_len, str_hidden_dim)

        inter_batch = []
        for i, tokens_len in enumerate(Ab_batch_lens):
            if self.virus_name == 'HIV':
                RBD_start = Agseq[i].find('GVP') - 8
                RBD_end = max(Agseq[i].find('APT')+15, Agseq[i].find('YKVV')+25)
            else:
                RBD_start = Agseq[i].find('NITN')
                RBD_end = Agseq[i].find('KKST') + 4
            Zag_inter = Zag[i, RBD_start :RBD_end + 1].unsqueeze(1)  # [Agseq_len_rbd, 1 esm_hidden_dim]
            Zab_inter = Zab[i, 1:tokens_len - 1].unsqueeze(0)  # [ 1, Abseq_len, esm_hidden_dim]
            # print(Zab_inter.shape,Zag_inter.shape)
            inter_matrix = Zag_inter*Zab_inter
            # print(inter_matrix.shape)
            Chem_matrix = generate_Chem_tensor(Abseq_H[i].replace("X", "G")+Abseq_L[i].replace("X", "G"),Agseq[i][RBD_start :RBD_end + 1].replace("X", "G")).to(self.device)
            # hydropathy_matrix = generate_hydropathy_tensor(Abseq_H[i]+Abseq_L[i],Agseq[i][RBD_start :RBD_end + 1]).to(self.device)
            # print(Hbond_matrix.shape)
            # inter_matrix = torch.cat([inter_matrix,Hbond_matrix,hydropathy_matrix], dim=2)
            inter_matrix = torch.cat([inter_matrix,Chem_matrix], dim=2)
            # print(inter_matrix.shape)
            
            l1_padding = self.max_Ag_len - inter_matrix.shape[0]
            l2_padding = self.max_Ab_len - inter_matrix.shape[1]
            inter_batch.append(F.pad(inter_matrix.unsqueeze(0), (0,0,1,l2_padding-1,1,l1_padding-1), "constant", 0).squeeze(0))
        x_inter = torch.stack(inter_batch, dim=0) # [batch_size, max_Agseq_len_rbd, max_Abseq_len, esm_hidden_dim]
        # print(x_inter.shape)
        x_inter = self.conv_module(x_inter)
        


        sequence_representations_str = []
        for i, tokens_len in enumerate(Ab_batch_lens):
            sequence_representations_str.append(Str_Zab[i, 1:tokens_len - 1].mean(0))  # [str_hidden_dim]

        x_str = torch.stack(sequence_representations_str, dim=0)  # shape: (batch_size, str_hidden_dim)

        x_cls_final = self.classification_head(torch.cat((x_str, x_inter), dim=1))  # shape: (batch_size, 1)

        return x_cls_final

class Reg_S3AI_Chem(nn.Module):
    def __init__(self, device, esm_hidden_dim=640, str_hidden_dim = 64, alpha = 0.8, max_Ab_len = 230, max_Ag_len = 700,virus_name = 'HIV', **kwargs):
        super(Reg_S3AI_Chem, self).__init__()
        self.device = device
        self.max_Ab_len = max_Ab_len
        self.max_Ag_len = max_Ag_len
        self.virus_name = virus_name
        self.Ab_encoder = ESM_encoder(device)
        self.Ag_encoder = ESM_encoder(device)
#         self.MLP = nn.Linear(esm_hidden_dim, str_hidden_dim)
        self.MLP = MLP(in_dim=640, hidden_size=640,
                          mid_batch_norm=False, out_dim=str_hidden_dim,
                          layers=4, batch_norm_momentum=0.1,device = device).to(device)
        self.conv_module = CNNModule(self.max_Ag_len, self.max_Ab_len, esm_hidden_dim+5).to(device)

        
        self.cls_dim = esm_hidden_dim + str_hidden_dim + 5
        self.regression_head = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim, self.cls_dim // 4),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 4),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 64),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),
            
            nn.Linear(self.cls_dim // 64, 1)).to(device)
        self.alpha = alpha


    def forward(self, Abseq_H, Abseq_L, Agseq):
        Zab, Ab_batch_lens = self.Ab_encoder(Abseq_H, Abseq_L)  # shape: (batch_size, Abseq_len, esm_hidden_dim=640), (batch_size,)
        Zag, Ag_batch_lens = self.Ag_encoder(Agseq, ['']*len(Agseq))  # shape: (batch_size, Agseq_len, esm_hidden_dim=640), (batch_size,)

        Str_Zab = self.MLP(Zab)  # shape: (batch_size, seq_len, str_hidden_dim)
        # print(Zab, Zag)
        inter_batch = []
        for i, tokens_len in enumerate(Ab_batch_lens):
            if self.virus_name == 'HIV':
                RBD_start = Agseq[i].find('GVP') - 8
                RBD_end = max(Agseq[i].find('APT')+15, Agseq[i].find('YKVV')+25)
            else:
                RBD_start = Agseq[i].find('NITN')
                RBD_end = Agseq[i].find('KKST') + 4
            Zag_inter = Zag[i, RBD_start :RBD_end + 1].unsqueeze(1)  # [Agseq_len_rbd, 1 esm_hidden_dim]
            Zab_inter = Zab[i, 1:tokens_len - 1].unsqueeze(0)  # [ 1, Abseq_len, esm_hidden_dim]
            # print(Zab_inter.shape,Zag_inter.shape)
            inter_matrix = Zag_inter*Zab_inter
            # print(inter_matrix.shape)
            Chem_matrix = generate_Chem_tensor(Abseq_H[i].replace("X", "G")+Abseq_L[i].replace("X", "G"),Agseq[i][RBD_start :RBD_end + 1].replace("X", "G")).to(self.device)
            # hydropathy_matrix = generate_hydropathy_tensor(Abseq_H[i]+Abseq_L[i],Agseq[i][RBD_start :RBD_end + 1]).to(self.device)
            # print(Hbond_matrix.shape)
            # inter_matrix = torch.cat([inter_matrix,Hbond_matrix,hydropathy_matrix], dim=2)
            inter_matrix = torch.cat([inter_matrix,Chem_matrix], dim=2)
            # print(inter_matrix.shape)
            # print(inter_matrix)
            l1_padding = self.max_Ag_len - inter_matrix.shape[0]
            l2_padding = self.max_Ab_len - inter_matrix.shape[1]
            inter_batch.append(F.pad(inter_matrix.unsqueeze(0), (0,0,1,l2_padding-1,1,l1_padding-1), "constant", 0).squeeze(0))
        x_inter = torch.stack(inter_batch, dim=0) # [batch_size, max_Agseq_len_rbd, max_Abseq_len, esm_hidden_dim]
        # print(x_inter.shape)
        x_inter = self.conv_module(x_inter)
        


        sequence_representations_str = []
        for i, tokens_len in enumerate(Ab_batch_lens):
            sequence_representations_str.append(Str_Zab[i, 1:tokens_len - 1].mean(0))  # [str_hidden_dim]

        x_str = torch.stack(sequence_representations_str, dim=0)  # shape: (batch_size, str_hidden_dim)
        # print(x_str, x_inter)

        x_reg_final = self.regression_head(torch.cat((x_str, x_inter), dim=1))  # shape: (batch_size, 1)

        return x_reg_final


class MultiTask_S3AI_non_inter(nn.Module):
    def __init__(self, device, esm_hidden_dim=640, str_hidden_dim = 64, alpha = 0.8, max_Ab_len = 250, max_Ag_len = 204,virus_name = 'SARSCOV2' ,**kwargs):
        super(MultiTask_S3AI_non_inter, self).__init__()
        self.device = device
        self.max_Ab_len = max_Ab_len
        self.max_Ag_len = max_Ag_len
        self.virus_name = virus_name
        self.Ab_encoder = ESM_encoder(device)
        self.Ag_encoder = ESM_encoder(device)
#         self.MLP = nn.Linear(esm_hidden_dim, str_hidden_dim)
        self.MLP = MLP(in_dim=640, hidden_size=640,
                          mid_batch_norm=False, out_dim=str_hidden_dim,
                          layers=4, batch_norm_momentum=0.1,device = device).to(device)
        self.conv_module = CNNModule(self.max_Ag_len, self.max_Ab_len, esm_hidden_dim+5).to(device)

        
        self.cls_dim = esm_hidden_dim * 2 + str_hidden_dim
        self.classification_head = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim, self.cls_dim // 4),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), ß#nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 4),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 64),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),
            
            nn.Linear(self.cls_dim // 64, 1),
            
            nn.Sigmoid()).to(device)

        self.regression_head = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim, self.cls_dim // 4),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 4),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim//4, self.cls_dim // 64),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),
            
            nn.Linear(self.cls_dim // 64, 1)).to(device)
        self.alpha = alpha


    def forward(self, Abseq_H, Abseq_L, Agseq):
        Zab, Ab_batch_lens = self.Ab_encoder(Abseq_H, Abseq_L)  # shape: (batch_size, Abseq_len, esm_hidden_dim=640), (batch_size,)
        Zag, Ag_batch_lens = self.Ag_encoder(Agseq, ['']*len(Agseq))  # shape: (batch_size, Agseq_len, esm_hidden_dim=640), (batch_size,)

        Str_Zab = self.MLP(Zab)  # shape: (batch_size, seq_len, str_hidden_dim)

        pooled_Zab = [Zab[i, :Ab_batch_lens[i]].mean(0) for i in range(len(Ab_batch_lens))]
        pooled_Zag = [Zag[i, :Ag_batch_lens[i]].mean(0) for i in range(len(Ag_batch_lens))]
        # print(x_inter.shape)
        pooled_Str_Zab = [Str_Zab[i, 1:tokens_len - 1].mean(0) for i, tokens_len in enumerate(Ab_batch_lens)]

        x_str = torch.stack(pooled_Zab, dim=0)  # shape: (batch_size, esm_hidden_dim)
        x_ag = torch.stack(pooled_Zag, dim=0)  # shape: (batch_size, esm_hidden_dim)
        x_str_ab = torch.stack(pooled_Str_Zab, dim=0)  # shape: (batch_size, str_hidden_dim)
        # Combining pooled representations
        combined_representation = torch.cat((x_str, x_ag, x_str_ab), dim=1)  # shape: (batch_size, esm_hidden_dim * 2 + str_hidden_dim)

        # Passing combined representation through classification and regression heads
        x_reg_final = self.regression_head(combined_representation)  # shape: (batch_size, 1)
        x_cls_final = self.classification_head(combined_representation)  # shape: (batch_size, 1)
        return x_reg_final, x_cls_final, combined_representation
    
class RegressionHead(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(RegressionHead, self).__init__()
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.linear(x)

class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(ClassificationHead, self).__init__()
        self.linear = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        return self.sigmoid(x)
