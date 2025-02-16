import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

import math, numpy as np


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_V, num_heads,act,ln=False):
        super(MAB, self).__init__()
        self.dim_V     = dim_V
        self.num_heads = num_heads
        self.act = act
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_Q, dim_V)
        self.fc_v = nn.Linear(dim_Q, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, inst_mode=False):
        Q    = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + self.act(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        
        if inst_mode: #[N,K,D] --> [N,K,D]
            return O
        else:# bag mode [N,1,D] --> [N,D]
            return O.squeeze(1)


class FR_MIL(nn.Module):
    def __init__(self, num_classes,num_heads,in_dim,k,act,hidden_dim):
        super(FR_MIL,self).__init__()
        self.num_outputs = num_classes
        num_heads        = num_heads
        self.act        = act
        self.k           = k
        dim_hidden       = hidden_dim
        self.feature = nn.Linear(in_dim, dim_hidden)
        self.enc = nn.Sequential(
            nn.Linear(dim_hidden, 1),
            nn.Sigmoid()
        )
        
        self.cls_token = nn.Parameter(torch.Tensor(1, 1, dim_hidden))
        nn.init.xavier_uniform_(self.cls_token)
        
        self.conv_head = torch.nn.Conv2d(dim_hidden, dim_hidden, 3, 1, 3//2,  groups=dim_hidden)  
        torch.nn.init.xavier_uniform_(self.conv_head.weight)
        
        self.selt_att  = MAB(dim_hidden, dim_hidden, num_heads,act)
        self.fc   = nn.Sequential(
            nn.Linear(dim_hidden, self.num_outputs),
        )
        
        self.mode = 0
        
    def recalib(self, inputs, option='max'):
        A1  = []
        Q   = []
        bs  = inputs.shape[0]
        if option == 'mean':
            Q  = torch.mean(inputs, dim=1, keepdim=True)
            A1 = self.enc(Q.squeeze(1))
            return A1, Q
        else:
            for i in range(bs):
                a1  = self.enc(inputs[i].unsqueeze(0)).squeeze(0)
                #print(a1.shape)
                _, m_indices = torch.sort(a1, 0, descending=True)
                #print(m_indices.shape)
                feat_q = []
                len_i = m_indices.shape[0] - 1
                for i_q in range(self.k):
                    if option == 'max':
                        feats = torch.index_select(inputs[i], dim=0, index=m_indices[i_q, :])
                    else:
                        feats = torch.index_select(inputs[i], dim=0, index=m_indices[len_i - i_q, :])
                    feat_q.append(feats)
                    
                feats = torch.stack(feat_q)
        
                A1.append(a1.squeeze(1))
                Q.append(feats.mean(0))
                
            A1 = torch.stack(A1)
            Q  = torch.stack(Q)
            return A1, Q
            
    def forward(self, inputs, return_WSI_attn = False, return_WSI_feature = False):
        forward_return = {}
        inputs = self.feature(inputs)
        
        A1, Q = self.recalib(inputs, 'max')

        inputs  = self.act(inputs - Q)
        i_shift = inputs
        #---->pad inputs 
        H          = inputs.shape[1] # Number of Instances in Bag    
        _H, _W     = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        inputs     = torch.cat([inputs, inputs[:,:add_length,:]],dim = 1) #[B, N+29, D//2 ]
        
    
        #---->cls_token
        B          = inputs.shape[0] # Batch Size
        cls_tokens = self.cls_token.expand(B, -1, -1)
        inputs     = torch.cat((cls_tokens, inputs), dim=1)
        
        # CNN Position Learning
        B, _, C  = inputs.shape
        cls_token, feat_token = inputs[:, 0], inputs[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, _H, _W)
        cnn_feat = self.conv_head(cnn_feat) + cnn_feat
        x        = cnn_feat.flatten(2).transpose(1, 2)
        x        = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        # Bag pooling with critical feature
        bag = self.selt_att(Q, x) 
        logits = self.fc(bag)
        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = bag
        if return_WSI_attn:
            WSI_attn = torch.matmul(feat_token, cls_token.transpose(0, 1)).squeeze(0)[:H,:]
            forward_return['WSI_attn'] = WSI_attn
        return forward_return
    
