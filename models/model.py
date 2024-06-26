from typing import Dict, List, Union, Callable

import torch
import numpy as np
from functools import partial

from torch import nn
import torch.nn.functional as F
import esm

from models.AbAgAttention import BidirectionalResidueAttention
from models.base_layers import MLP
EPS = 1e-5


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


class AbAgPRED(nn.Module):
    def __init__(self, device, esm_hidden_dim=640, str_hidden_dim = 64, **kwargs):
        super(AbAgPRED, self).__init__()
        self.Ab_encoder = ESM_encoder(device)
        self.Ag_encoder = ESM_encoder(device)
        self.MLP = MLP(in_dim=640, hidden_size=640,
                          mid_batch_norm=True, out_dim=64,
                          layers=4, batch_norm_momentum=0.1,device = device).to(device)
        self.Bi_Att = BidirectionalResidueAttention(esm_hidden_dim)
        self.output_layer = nn.Linear(esm_hidden_dim+str_hidden_dim, 1)

    def forward(self, Abseq_H, Abseq_L, Agseq):
        Zab, Ab_batch_lens = self.Ab_encoder(Abseq_H, Abseq_L)  # shape: (batch_size, Abseq_len, esm_hidden_dim=640), (batch_size,)
        Zag, Ag_batch_lens = self.Ag_encoder(Agseq, ['']*len(Agseq))  # shape: (batch_size, Agseq_len, esm_hidden_dim=640), (batch_size,)
        Att_ab_ag = self.Bi_Att(Zab, Zag)  # shape: (batch_size, Abseq_len, esm_hidden_dim=640)
        Str_Zab = self.MLP(Zab)  # shape: (batch_size, seq_len, str_hidden_dim)

        sequence_representations_int = []
        for i, tokens_len in enumerate(Ab_batch_lens):
            sequence_representations_int.append(Att_ab_ag[i, 1:tokens_len - 1].mean(0))  # [esm_hidden_dim]

        x_int = torch.stack(sequence_representations_int, dim=0)  # shape: (batch_size, esm_hidden_dim)

        sequence_representations_str = []
        for i, tokens_len in enumerate(Ab_batch_lens):
            sequence_representations_str.append(Str_Zab[i, 1:tokens_len - 1].mean(0))  # [str_hidden_dim]

        x_str = torch.stack(sequence_representations_str, dim=0)  # shape: (batch_size, str_hidden_dim)

        x_final = self.output_layer(torch.cat((x_str, x_int), dim=1))  # shape: (batch_size, 1)

        return x_final

class MultiTask_AbAgPRED(nn.Module):
    def __init__(self, device, esm_hidden_dim=640, str_hidden_dim = 64, **kwargs):
        super(MultiTask_AbAgPRED, self).__init__()
        self.Ab_encoder = ESM_encoder(device)
        self.Ag_encoder = ESM_encoder(device)
#         self.MLP = nn.Linear(esm_hidden_dim, str_hidden_dim)
        self.MLP = MLP(in_dim=640, hidden_size=640,
                          mid_batch_norm=False, out_dim=str_hidden_dim,
                          layers=4, batch_norm_momentum=0.1,device = device).to(device)
        self.Bi_Att = BidirectionalResidueAttention(esm_hidden_dim)
        self.regression_head = RegressionHead(esm_hidden_dim+str_hidden_dim, 1)
        self.classification_head = ClassificationHead(esm_hidden_dim+str_hidden_dim, 2)

    def forward(self, Abseq_H, Abseq_L, Agseq):
        Zab, Ab_batch_lens = self.Ab_encoder(Abseq_H, Abseq_L)  # shape: (batch_size, Abseq_len, esm_hidden_dim=640), (batch_size,)
        Zag, Ag_batch_lens = self.Ag_encoder(Agseq, ['']*len(Agseq))  # shape: (batch_size, Agseq_len, esm_hidden_dim=640), (batch_size,)
        Att_ab_ag = self.Bi_Att(Zab, Zag)  # shape: (batch_size, Abseq_len, esm_hidden_dim=640)
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
    
class MultiTask_AbAgPRED_nba(nn.Module):
    def __init__(self, device, esm_hidden_dim=640, str_hidden_dim = 64, **kwargs):
        super(MultiTask_AbAgPRED, self).__init__()
        self.Ab_encoder = ESM_encoder(device)
        self.Ag_encoder = ESM_encoder(device)
#         self.MLP = nn.Linear(esm_hidden_dim, str_hidden_dim)
        self.MLP = MLP(in_dim=640, hidden_size=640,
                          mid_batch_norm=False, out_dim=str_hidden_dim,
                          layers=4, batch_norm_momentum=0.1,device = device).to(device)
#         self.Bi_Att = BidirectionalResidueAttention(esm_hidden_dim)
        self.regression_head = RegressionHead(esm_hidden_dim+str_hidden_dim, 1)
        self.classification_head = ClassificationHead(esm_hidden_dim+str_hidden_dim, 2)

    def forward(self, Abseq_H, Abseq_L, Agseq, weight):
        Zab, Ab_batch_lens = self.Ab_encoder(Abseq_H, Abseq_L)  # shape: (batch_size, Abseq_len, esm_hidden_dim=640), (batch_size,)
        Zag, Ag_batch_lens = self.Ag_encoder(Agseq, ['']*len(Agseq))  # shape: (batch_size, Agseq_len, esm_hidden_dim=640), (batch_size,)
#         Att_ab_ag = self.Bi_Att(Zab, Zag)  # shape: (batch_size, Abseq_len, esm_hidden_dim=640)
        Str_Zab = self.MLP(Zab)  # shape: (batch_size, seq_len, str_hidden_dim)

        all_Z = torch.cat((Zab, Zag), dim=1)
        sequence_representations_int = []
        for i, tokens_len in enumerate(Ab_batch_lens):
            all_len = tokens_len + Ag_batch_lens[i]
            sequence_representations_int.append(all_Z[i, 1:all_len - 2].mean(0))  # [esm_hidden_dim]

        x_int = torch.stack(sequence_representations_int, dim=0)  # shape: (batch_size, esm_hidden_dim)

        sequence_representations_str = []
        for i, tokens_len in enumerate(Ab_batch_lens):
            sequence_representations_str.append(Str_Zab[i, 1:tokens_len - 1].mean(0))  # [str_hidden_dim]

        x_str = torch.stack(sequence_representations_str, dim=0)  # shape: (batch_size, str_hidden_dim)

        x_reg_final = self.regression_head(torch.cat((x_str, x_int), dim=1))  # shape: (batch_size, 1)
        x_cls_final = self.classification_head(torch.cat((x_str, x_int), dim=1))  # shape: (batch_size, 1)

        return x_reg_final, x_cls_final
    
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