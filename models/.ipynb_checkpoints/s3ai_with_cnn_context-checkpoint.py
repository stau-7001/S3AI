from typing import Dict, List, Union, Callable

import torch
import numpy as np
from functools import partial

from torch import nn
import torch.nn.functional as F
import esm
import random

from models.AbAgAttention import BidirectionalResidueAttention
from models.base_layers import MLP
EPS = 1e-5
device=torch.device('cuda')

def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

class LayerNorm(nn.Module):
    def __init__(self, eps=1e-5):
        """
        LayerNorm constructor.

        Args:
            eps (float, optional): A value added to the denominator for numerical stability.
        """
        super(LayerNorm, self).__init__()
        self.eps = eps

    def forward(self, input_tensor):
        """
        LayerNorm forward pass.

        Args:
            input_tensor (torch.Tensor): Input tensor to be normalized.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        mean = input_tensor.mean(dim=(1, 2), keepdim=True)
        var = input_tensor.var(dim=(1, 2), unbiased=False, keepdim=True)
        div = torch.sqrt(var + self.eps)
        output_tensor = (input_tensor - mean) / div
        return output_tensor

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

class ContextPooling(nn.Module):
    def __init__(self,seq_len = 1,in_dim=1024):
        super(ContextPooling,self).__init__()
        self.seq_len=seq_len
        self.conv=nn.Sequential(
            nn.Conv1d(in_dim,in_dim*2,3,stride=1,padding=1),
            LayerNorm(),
            nn.LeakyReLU(True),

            nn.Conv1d(in_dim*2,in_dim*2,3,stride=1,padding=1),
            LayerNorm(),
            nn.LeakyReLU(True),

            nn.Conv1d(in_dim*2,2,3,stride=1,padding=1),
            LayerNorm(),
            nn.LeakyReLU(True),
        )

    def _local_normal(self,s,center,seq_len,r=0.1):
        PI=3.1415926
        std_=(r*seq_len*s[:,center]).unsqueeze(1) #[B,1]
        mean_=center
        place=torch.arange(seq_len).float().repeat(std_.shape[0],1).to(device) # [B,L]

        #print(std_)

        ret=pow(2*PI,-0.5)*torch.pow(std_,-1)*torch.exp(-torch.pow(place-mean_,2)/(1e-5+2*torch.pow(std_,2)))

        #ret-=torch.max(ret,dim=1)[0].unsqueeze(1)
        #ret=torch.softmax(ret,dim=1)

        ret/=torch.max(ret,dim=1)[0].unsqueeze(1)
#         print(ret.shape)


        return ret

    def forward(self,feats): # feats: [B,L,1024]
        feats_=feats.permute(0,2,1)
        feats_=self.conv(feats_) #output: [B,2,L]
        s,w=feats_[:,0,:].squeeze(1),feats_[:,1,:].squeeze(1) #[B,L]
        s=torch.softmax(s,1)
        w=torch.softmax(w,1)
#         print(w.shape)

        out=[]

        for i in range(feats.shape[-2]):
            w_=self._local_normal(s,i,feats.shape[-2])*w
            w_=w_.unsqueeze(2) # [B,L,1]
            out.append((w_*feats).sum(1,keepdim=True)) # w_ [B,L,1], feats [B,L,1024]

        out=torch.cat(out,dim=1) # [B,L,1024]
        return out

class SoluModel(nn.Module):
    def __init__(self, seq_len = 1 ,in_dim=640, sa_out=640, conv_out=640):
        super(SoluModel, self).__init__()
        
        #self.self_attention=SelfAttention(in_dim,4,sa_out,0.6) # input: [B,L,1024] output: [B,L,1024]
        self.contextpooling=ContextPooling(seq_len,in_dim)

        self.conv=nn.Sequential( #input: [B,1024,L] output: [B,1024,L]
            nn.Conv1d(in_dim,in_dim*2,3,stride=1,padding=1),# [B,in_dim*2,L] 
            LayerNorm(),
            nn.LeakyReLU(True),

            nn.Conv1d(in_dim*2,in_dim*2,3,stride=1,padding=1),
            LayerNorm(),
            nn.LeakyReLU(True),

            nn.Conv1d(in_dim*2,conv_out,3,stride=1,padding=1),
            LayerNorm(),
            nn.LeakyReLU(True),
        )

        self.cls_dim=sa_out+conv_out
        
        self.mlp = nn.Sequential(
            nn.Dropout(p=0.6),
            nn.Linear(self.cls_dim, self.cls_dim // 2),
            nn.LeakyReLU(True))
        
        self._initialize_weights()

    def forward(self, feats):
        out_sa=self.contextpooling(feats)+feats

        out_conv=self.conv(feats.permute(0,2,1))
        out_conv=out_conv.permute(0,2,1)+feats

        out=torch.cat([out_sa,out_conv],dim=2)
#         out=torch.max(out,dim=1)[0].squeeze()
        
        out = self.mlp(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class MultiTask_S3AI_with_cnn_context(nn.Module):
    def __init__(self, device, esm_hidden_dim=640, str_hidden_dim = 64, alpha = 0.8, **kwargs):
        super(MultiTask_S3AI_with_cnn_context, self).__init__()
        self.device = device
        self.Ab_encoder = ESM_encoder(device)
        self.Ag_encoder = ESM_encoder(device)
#         self.MLP = nn.Linear(esm_hidden_dim, str_hidden_dim)
        self.MLP = MLP(in_dim=640, hidden_size=640,
                          mid_batch_norm=False, out_dim=str_hidden_dim,
                          layers=4, batch_norm_momentum=0.1,device = device).to(device)
        self.solu = SoluModel(in_dim=640, sa_out=640, conv_out=640)
        self.Bi_Att = BidirectionalResidueAttention(esm_hidden_dim)
        self.regression_head = RegressionHead(esm_hidden_dim+str_hidden_dim, 1)
        self.classification_head = ClassificationHead(esm_hidden_dim+str_hidden_dim, 2)
        self.alpha = alpha


    def forward(self, Abseq_H, Abseq_L, Agseq, weight):
        Zab, Ab_batch_lens = self.Ab_encoder(Abseq_H, Abseq_L)  # shape: (batch_size, Abseq_len, esm_hidden_dim=640), (batch_size,)
        Zag, Ag_batch_lens = self.Ag_encoder(Agseq, ['']*len(Agseq))  # shape: (batch_size, Agseq_len, esm_hidden_dim=640), (batch_size,)
#         Zag = fixed_attention_weighted_feature(Zag, weight,self.alpha)
        Zag = self.solu(Zag)# (batch_size, Agseq_len, esm_hidden_dim=640)
    
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
    
class MultiTask_S3AI_nba(nn.Module):
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
        Zag = attention_weighted_feature(Zag,weight)
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