import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AntibodyAntigenAttention(nn.Module):
    def __init__(self, hidden_d):
        super(AntibodyAntigenAttention, self).__init__()
        self.Wq = nn.Linear(hidden_d, hidden_d)
        self.Wk = nn.Linear(hidden_d, hidden_d)
        # self.Wv = nn.Linear(hidden_d, hidden_d)

    def forward(self, antibody, antigen, return_attention_weights=True):
        '''
        antibody:(batch_size, n, hidden_d)
        antigen:(batch_size, m, hidden_d)
        '''
        # query，key, value
        queries = self.Wq(antibody)
        keys = self.Wk(antigen)
        # values = self.Wv(antigen)#(batch_size, m, hidden_d)

        # (batch_size, n, m)
        attention_scores = torch.bmm(queries, keys.transpose(1, 2))

        # 
        d_k = keys.size(-1)
        attention_scores = attention_scores / math.sqrt(d_k)

        # (batch_size, n, m)
        attention_weights = F.softmax(attention_scores, dim=-1)

        if return_attention_weights:
            return attention_weights # (batch_size, n, m)

        # (batch_size, n, hidden_d)
        # output = torch.bmm(attention_weights, values

class BidirectionalResidueAttention(nn.Module):
    """
    Computes bidirectional attention weights between antibody and antigen.
    
    Args:
        antibody (torch.Tensor): Tensor of shape (batch_size, n, hidden_d)
        antigen (torch.Tensor): Tensor of shape (batch_size, m, hidden_d)
    
    Returns:
        bidirectional_attention_weights (torch.Tensor): Tensor of shape (batch_size, n)
    """
    def __init__(self, hidden_d,device = torch.device("cuda:0")):
        super(BidirectionalResidueAttention, self).__init__()
        self.attention_antibody = AntibodyAntigenAttention(hidden_d).to(device)
        self.attention_antigen = AntibodyAntigenAttention(hidden_d).to(device)
        self.Wv = nn.Linear(hidden_d, hidden_d).to(device)

    def forward(self, antibody, antigen, return_att = False):
        values = self.Wv(antigen) #(batch_size, m, hidden_d)
        attention_weights_ab2ag = self.attention_antibody(antibody, antigen, return_attention_weights=True) # (batch_size, n, m)
        
        attention_weights_ag2ab = self.attention_antigen(antigen, antibody, return_attention_weights=True) # (batch_size, m, n)
        attention_weights_ag2ab_transposed = attention_weights_ag2ab.transpose(1, 2) # (batch_size, n, m)
        import matplotlib.pyplot as plt
        for batch_idx in range(attention_weights_ab2ag.shape[0]):
            plt.figure(figsize=(10, 6))
            plt.imshow(attention_weights_ab2ag[batch_idx].cpu(), cmap='viridis', aspect='auto', vmin=0, vmax=1)
            plt.colorbar()
#             plt.xticks(np.arange(att_data.shape[2]), Agseq, rotation=90)
#             plt.yticks(np.arange(att_data.shape[1]), Abseq_H + Abseq_L)
#             plt.xlabel('Agseq')
#             plt.ylabel('Abseq_H+Abseq_L')
            plt.title(f'Attention Heatmap_attention_weights_ag2ab - Sample {batch_idx}')
            plt.tight_layout()
            plt.savefig(f'attention_heatmap_sample_attention_weights_ag2ab_{batch_idx}.png')
            plt.close()
        output_weights = (attention_weights_ab2ag + attention_weights_ag2ab_transposed) # (batch_size, n, m)

        bidirectional_attention_weights = F.softmax(output_weights, dim=-1) # (batch_size, n, m)
        
        output = torch.bmm(bidirectional_attention_weights, values) # (batch_size, n, hidden_d)
        
        if return_att:
            return output_weights,output
      
        return output