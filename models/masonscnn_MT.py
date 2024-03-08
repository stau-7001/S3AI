#!/usr/bin/env
# coding:utf-8

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn

class CNNmodule(nn.Module):
    def __init__(self, in_channel, kernel_width, l=0):
        super(CNNmodule, self).__init__()
        self.kernel_width = kernel_width
        self.conv = nn.Conv1d(in_channels=in_channel, out_channels=64, kernel_size=2, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.out_linear = nn.Linear(l*64, 32)
        self.dropout = nn.Dropout(0.5)


    def forward(self, protein_ft):
        '''
        :param protein_ft: batch*len*amino_dim
        :return:
        '''
        batch_size = protein_ft.shape[0]
#         print(batch_size)
#         print(type(protein_ft))        
        protein_ft = protein_ft.transpose(1, 2)
        conv_ft = self.conv(protein_ft)
        conv_ft = self.dropout(conv_ft)
        conv_ft = self.pool(conv_ft).view(batch_size, -1)
        conv_ft = self.out_linear(conv_ft)
        return conv_ft


class MasonsCNN(nn.Module):
    def __init__(self, amino_ft_dim = 21, max_antibody_len = 230, max_virus_len = 1275,
                 h_dim=512,
                 dropout=0.1,reg = True, cls = True,**kwargs
                 ):
        super(MasonsCNN, self).__init__()
        self.amino_ft_dim = amino_ft_dim
        self.h_dim = h_dim
        self.dropout = dropout
        self.inception_out_channel = 3
        self.max_antibody_len = max_antibody_len
        self.max_virus_len = max_virus_len
        self.reg = reg
        self.cls = cls

        self.cnnmodule = CNNmodule(in_channel=22, kernel_width=amino_ft_dim, l=self.max_antibody_len)
        self.cnnmodule2 = CNNmodule(in_channel=22, kernel_width=amino_ft_dim, l=self.max_virus_len)
        self.cls_dim = 64
        # self.out_linear1 = nn.Linear(64, 32)
        # self.out_linear2 = nn.Linear(32, 1)
        #         self.classification_head = ClassificationHead(esm_hidden_dim+str_hidden_dim, 2)
        self.classification_head = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim, self.cls_dim // 2),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),
            
            nn.Linear(self.cls_dim // 2, 1),
            
            nn.Sigmoid())

        self.regression_head = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim, self.cls_dim // 2),
            nn.ELU(True),#nn.Tanh(),#nn.LeakyReLU(True), #nn.ReLU(True),

            nn.Linear(self.cls_dim // 2, 1))

        self.activation = nn.ELU()


    def forward(self, batch_antibody_ft, batch_virus_ft):
        '''
        :param batch_antibody_ft:   tensor    batch, len, amino_ft_dim
        :param batch_virus_ft:     tensor    batch, amino_ft_dim
        :return:
        '''

        batch_size = batch_antibody_ft.shape[0]
        antibody_ft = self.cnnmodule(batch_antibody_ft).view(batch_size, -1)
        virus_ft = self.cnnmodule2(batch_virus_ft).view(batch_size, -1)

        pair_ft = torch.cat([virus_ft, antibody_ft], dim=-1).view(batch_size, -1)
        # pair_ft = self.activation(pair_ft)
        # pair_ft = self.out_linear1(pair_ft)
        # pair_ft = self.activation(pair_ft)
        # pred = self.out_linear2(pair_ft)
        if self.reg and self.cls:
            reg_pred = self.regression_head(pair_ft)
            cls_pred = self.classification_head(pair_ft)
            return reg_pred, cls_pred
        if self.cls:
            cls_pred = self.classification_head(pair_ft)
            return cls_pred
        if self.reg:
            reg_pred = self.regression_head(pair_ft)
            return reg_pred