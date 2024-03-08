#!/usr/bin/env
# coding:utf-8

import torch
import torch.nn as nn
# from models.layers.attention_layer import Att2One

class Att2One(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(Att2One, self).__init__()

        self.linear_trans = nn.Linear(input_dim, hidden_dim)

        self.linear_q = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, ft_mat):
        '''
        :param ft_mat:  batch, n_channel, ft
        :return:
        '''
        w_mat = torch.tanh(self.linear_trans(ft_mat))
        w_mat = self.linear_q(w_mat)
        # print(w_mat[0])
        w_mat = F.softmax(w_mat, dim=1)  # batch n_channel 1
        # print(w_mat.shape, w_mat[0])
        ft_mat = torch.sum(ft_mat * w_mat, dim=1) # batch 1 ft
        ft_mat = ft_mat.squeeze()
        # print(ft_mat.size())

        return ft_mat

class BiLSTMAttentionReg(nn.Module):
    def __init__(self, h_dim, amino_ft_dim, dropout_num, amino_embedding_dim, bilstm_num_layers):
        super(BiLSTMAttentionReg, self).__init__()
        self.h_dim = h_dim
        self.amino_ft_dim = amino_ft_dim
        self.dropout_num = dropout_num
        self.amino_embedding_dim = amino_embedding_dim

        self.amino_embedding_layer = nn.Embedding(self.amino_ft_dim, self.amino_embedding_dim)
        self.share_bi_lstm_layer = nn.LSTM(
            input_size=self.amino_embedding_dim,
            hidden_size=self.h_dim // 2,
            num_layers=bilstm_num_layers,
            dropout=self.dropout_num,
            batch_first=True,
            bidirectional=True
        )


        self.share_att_layer = Att2One(self.h_dim, self.h_dim)
        self.dropout_layer = nn.Dropout(self.dropout_num)
        # self.linear_merge = nn.Linear(self.h_dim * 2, self.h_dim)
        # self.linear_pred = nn.Linear(self.h_dim, 1)
        # self.activation = nn.ELU()
        self.cls_dim = self.h_dim * 2
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
        # self.activation = nn.ELU()
        self.activation = nn.ELU()

    def forward(self,
                batch_antibody_amino_ft,
                batch_virus_amino_ft
                ):
        self.share_bi_lstm_layer.flatten_parameters()
        batch_size = batch_antibody_amino_ft.size()[0]
        deivce = batch_antibody_amino_ft.device

        virus_ft = self.amino_embedding_layer(batch_virus_amino_ft)
        antibody_ft = self.amino_embedding_layer(batch_antibody_amino_ft)

        h_0 = torch.randn(2 * self.share_bi_lstm_layer.num_layers, batch_size, self.share_bi_lstm_layer.hidden_size).to(deivce)
        c_0 = torch.randn(2 * self.share_bi_lstm_layer.num_layers, batch_size, self.share_bi_lstm_layer.hidden_size).to(deivce)
        virus_ft, (hn, cn) = self.share_bi_lstm_layer(virus_ft, (h_0, c_0))

        h_0 = torch.randn(2 * self.share_bi_lstm_layer.num_layers, batch_size, self.share_bi_lstm_layer.hidden_size).to(deivce)
        c_0 = torch.randn(2 * self.share_bi_lstm_layer.num_layers, batch_size, self.share_bi_lstm_layer.hidden_size).to(deivce)
        antibody_ft, (hn, cn) = self.share_bi_lstm_layer(antibody_ft, (h_0, c_0))

        antibody_ft = self.activation(antibody_ft)
        virus_ft = self.activation(virus_ft)
        antibody_ft = self.dropout_layer(antibody_ft)
        virus_ft = self.dropout_layer(virus_ft)

        antibody_ft = self.share_att_layer(antibody_ft)
        virus_ft = self.share_att_layer(virus_ft)

        pair_ft = torch.cat([virus_ft, antibody_ft], dim=-1)
        pair_ft = self.activation(pair_ft)
        pair_ft = self.dropout_layer(pair_ft)

        # pair_ft = self.linear_merge(pair_ft)
        # pair_ft = self.activation(pair_ft)
        # pair_ft = self.dropout_layer(pair_ft)
        # pred = self.linear_pred(pair_ft)
        cls_pred = self.classification_head(pair_ft)
        reg_pred = self.regression_head(pair_ft)
        return reg_pred, cls_pred

        # return pred
