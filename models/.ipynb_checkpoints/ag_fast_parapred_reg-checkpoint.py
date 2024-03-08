#!/usr/bin/env
# coding:utf-8

from torch.nn.utils import weight_norm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy


class AttentionLayer(nn.Module):
    def __init__(self, antibody_ft_dim, virus_ft_dim, heads=1, dropout=0.15):
        super(AttentionLayer, self).__init__()
        self.heads = heads
        self.dropout = dropout
        self.antibody_ft_dim = antibody_ft_dim
        self.virus_ft_dim = virus_ft_dim

        self.antibody_trans_list = nn.ModuleList()
        self.virus_trans_list = nn.ModuleList()
        for idx in range(self.heads):
            self.antibody_trans_list.append(nn.Conv1d(self.antibody_ft_dim, 1, 1))
            self.virus_trans_list.append(nn.Conv1d(self.virus_ft_dim, 1, 1))

    def forward(self, antibody_ft_mat, virus_ft_mat, bias_mat):
        res_mat = []
        for idx in range(self.heads):
            w_antibody = self.antibody_trans_list[idx](antibody_ft_mat)
            w_virus = self.virus_trans_list[idx](virus_ft_mat)
            w = F.leaky_relu(w_virus + torch.transpose(w_antibody, 1, 2), negative_slope=0.2)  # batch, virus_len, antibody_len
            w = F.softmax(w + bias_mat, dim=-1)
            w = F.dropout(w, p=self.dropout, training=self.training)

            trans_ft = torch.bmm(w, torch.transpose(virus_ft_mat, 1, 2))
            res_mat.append(trans_ft)
        res_mat = torch.cat(res_mat, dim=2)
        return res_mat



class AgFastParapredReg(nn.Module):
    def __init__(self, ft_dim, max_antibody_len, max_virus_len,
                 h_dim=256,
                 position_coding=True
                 ):
        super(AgFastParapredReg, self).__init__()
        self.ft_dim = ft_dim
        self.max_virus_len = max_virus_len
        self.max_antibody_len = max_antibody_len

        self.h_dim = h_dim
        self.position_coding = position_coding

        # channel -->  amino ft dim
        self.antibody_conv1 = nn.Conv1d(self.ft_dim, 64, 3, padding=1)  # antibody first a trous convolutional layer
        self.antibody_conv2 = nn.Conv1d(64, 128, 3, padding=2, dilation=2)  #antibody second a trous convolutional layer
        self.antibody_conv3 = nn.Conv1d(128, 256, 3, padding=4, dilation=4)  #antibody third a trous convolutional layer

        self.virus_conv1 = nn.Conv1d(self.ft_dim, 64, 3, padding=1)   # antigen first a trous convolutional layer
        self.virus_conv2 = nn.Conv1d(64, 128, 3, padding=2, dilation=2)  #antigen second a trous convolutional layer
        self.virus_conv3 = nn.Conv1d(128, 256, 3, padding=4, dilation=4)

        self.antibody_bn1 = nn.BatchNorm1d(64)  # batch normalisation after the first convolutional layer for antibody
        self.antibody_bn2 = nn.BatchNorm1d(128)
        self.antibody_bn3 = nn.BatchNorm1d(256)

        self.bn4 = nn.BatchNorm1d(512)  # after cat (raw, att)

        self.virus_bn1 = nn.BatchNorm1d(64)  # batch normalisation after the first convolutional layer for antigen
        self.virus_bn2 = nn.BatchNorm1d(128)
        self.virus_bn3 = nn.BatchNorm1d(256)

        self.dropout1 = nn.Dropout(0.15)
        self.dropout2 = nn.Dropout(0.5)

        self.attention_layer = AttentionLayer(antibody_ft_dim=256, virus_ft_dim=256, heads=1, dropout=0.15)
        self.final_linear1 = nn.Linear(512, self.h_dim)
        self.final_linear2 = nn.Linear(self.h_dim, 1)

        self.activation = nn.ELU()
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0.0)
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0.0)

    def generate_bias_mat(self, batch_antibody_len, batch_virus_len, device='cpu'):
        batch_size = batch_antibody_len.shape[0]
        bias_mat = []
        for idx in range(batch_size):
            antibody_len = batch_antibody_len[idx]
            virus_len = batch_virus_len[idx]
            mat = torch.zeros(1, self.max_antibody_len, self.max_virus_len).to(device)
            mat[:, antibody_len:, :] = -1e9
            mat[:, :, virus_len:] = -1e9
            bias_mat.append(mat)
        bias_mat = torch.cat(bias_mat, dim=0)
        return bias_mat

    def forward(self, batch_antibody_ft, batch_virus_ft, batch_antibody_len, batch_virus_len):
        '''
        :param batch_antibody_ft:   tensor    batch, max_antibody_len, amino_ft_dim
        :param batch_virus_ft:     tensor    batch, max_virus_len, amino_ft_dim
        :param batch_antibody_len:   np array      batch
        :param batch_virus_len:     np array      batch
        :return:
        '''
        assert batch_antibody_ft.size()[0] == batch_virus_ft.size()[0]
        batch_size = batch_antibody_ft.size()[0]

        # virus_ft_mat shape ->  batch, amino_ft_dim, amino_max_len
        virus_ft_mat = batch_virus_ft.transpose(1, 2)
        antibody_ft_mat = batch_antibody_ft.transpose(1, 2)

        # generate mask mat
        antibody_mask_mat = torch.zeros(batch_size, 1, self.max_antibody_len).to(batch_antibody_ft.device)
        virus_mask_mat = torch.zeros(batch_size, 1, self.max_virus_len).to(batch_virus_ft.device)
        for idx in range(batch_size):
            antibody_mask_mat[idx, :batch_antibody_len[idx]] = 1
            virus_mask_mat[idx, :batch_virus_len[idx]] = 1

        # antibody ft extraction
        antibody_ft_mat = torch.mul(antibody_ft_mat, antibody_mask_mat)

        antibody_ft_mat = self.antibody_conv1(antibody_ft_mat)
        antibody_ft_mat = torch.mul(antibody_ft_mat, antibody_mask_mat)
        antibody_ft_mat = self.antibody_bn1(antibody_ft_mat)
        antibody_ft_mat = self.activation(antibody_ft_mat)
        antibody_ft_mat = torch.mul(antibody_ft_mat, antibody_mask_mat)
        antibody_ft_mat = self.dropout1(antibody_ft_mat)

        antibody_ft_mat = self.antibody_conv2(antibody_ft_mat)
        antibody_ft_mat = torch.mul(antibody_ft_mat, antibody_mask_mat)
        antibody_ft_mat = self.antibody_bn2(antibody_ft_mat)
        antibody_ft_mat = self.activation(antibody_ft_mat)
        antibody_ft_mat = torch.mul(antibody_ft_mat, antibody_mask_mat)
        antibody_ft_mat = self.dropout1(antibody_ft_mat)

        antibody_ft_mat = self.antibody_conv3(antibody_ft_mat)
        antibody_ft_mat = torch.mul(antibody_ft_mat, antibody_mask_mat)
        antibody_ft_mat = self.antibody_bn3(antibody_ft_mat)
        antibody_ft_mat = self.activation(antibody_ft_mat)
        antibody_ft_mat = torch.mul(antibody_ft_mat, antibody_mask_mat)
        antibody_ft_mat = self.dropout1(antibody_ft_mat)

        # virus ft extraction
        virus_ft_mat = torch.mul(virus_ft_mat, virus_mask_mat)

        virus_ft_mat = self.virus_conv1(virus_ft_mat)
        virus_ft_mat = torch.mul(virus_ft_mat, virus_mask_mat)
        virus_ft_mat = self.virus_bn1(virus_ft_mat)
        virus_ft_mat = self.activation(virus_ft_mat)
        virus_ft_mat = torch.mul(virus_ft_mat, virus_mask_mat)
        virus_ft_mat = self.dropout1(virus_ft_mat)

        virus_ft_mat = self.virus_conv2(virus_ft_mat)
        virus_ft_mat = torch.mul(virus_ft_mat, virus_mask_mat)
        virus_ft_mat = self.virus_bn2(virus_ft_mat)
        virus_ft_mat = self.activation(virus_ft_mat)
        virus_ft_mat = torch.mul(virus_ft_mat, virus_mask_mat)
        virus_ft_mat = self.dropout1(virus_ft_mat)

        virus_ft_mat = self.virus_conv3(virus_ft_mat)
        virus_ft_mat = torch.mul(virus_ft_mat, virus_mask_mat)
        virus_ft_mat = self.virus_bn3(virus_ft_mat)
        virus_ft_mat = self.activation(virus_ft_mat)
        virus_ft_mat = torch.mul(virus_ft_mat, virus_mask_mat)
        virus_ft_mat = self.dropout1(virus_ft_mat)

        # multi heads attention    344 * 912
        # update antibody ft
        # head = 1
        # bias_mat:  344 * 912  0:save   -1e9:
        bias_mat = self.generate_bias_mat(batch_antibody_len, batch_virus_len, device=batch_virus_ft.device)
        trans_ft = self.attention_layer(antibody_ft_mat, virus_ft_mat, bias_mat)

        trans_ft = trans_ft.transpose(1, 2)
        antibody_ft_mat = torch.cat([antibody_ft_mat, trans_ft], dim=1)
        antibody_ft_mat = torch.mul(antibody_ft_mat, antibody_mask_mat)

        antibody_ft_mat = self.bn4(antibody_ft_mat)
        antibody_ft_mat = self.activation(antibody_ft_mat)
        antibody_ft_mat = torch.mul(antibody_ft_mat, antibody_mask_mat)
        antibody_ft_mat, _ = torch.max(antibody_ft_mat, dim=-1)

        pair_ft = self.final_linear1(antibody_ft_mat)
        pair_ft = self.activation(pair_ft)
        pred = self.final_linear2(pair_ft)
        return torch.sigmoid(pred)


