import torch
import torch.nn as nn
import numpy as np
from .AMD_Layer import AMD_Layer

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class AMDLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, agent_num=512, tem=0, pool=False, thresh=None, thresh_tem='classical', kaiming_init=False):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = AMD_Layer(
            dim = dim,
            agent_num=agent_num,
            heads = 8,          
        )

    def forward(self, x, return_WSI_attn = False):
        forward_return = self.attn(self.norm(x), return_WSI_attn)
        x = x + forward_return['amd_out']
        new_forward_return = {}
        new_forward_return['amd_out'] = x
        if return_WSI_attn:
            new_forward_return['WSI_attn'] = forward_return['WSI_attn']

        return new_forward_return


'''
@article{shao2021transmil,
  title={Transmil: Transformer based correlated multiple instance learning for whole slide image classification},
  author={Shao, Zhuchen and Bian, Hao and Chen, Yang and Wang, Yifeng and Zhang, Jian and Ji, Xiangyang and others},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={2136--2147},
  year={2021}
}
'''
class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class AMD_MIL(nn.Module):
    def __init__(self, num_classes,in_dim,embed_dim,dropout,act,agent_num=256):
        super(AMD_MIL, self).__init__()
        self.pos_layer = PPEG(dim=embed_dim) # PPEG from TransMIL

        self._fc1 = [nn.Linear(in_dim, embed_dim)]
        self._fc1 += [act]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]
        self._fc1 = nn.Sequential(*self._fc1)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        nn.init.normal_(self.cls_token, std=1e-6)
        self.num_classes = num_classes
        self.amdlayer1 = AMDLayer(dim=embed_dim, agent_num = agent_num)
        self.amdlayer2 = AMDLayer(dim=embed_dim, agent_num = agent_num)
        self.norm = nn.LayerNorm(embed_dim)         
        self._fc2 = nn.Linear(embed_dim, self.num_classes)
        self.apply(initialize_weights)

    def forward(self, x, return_WSI_attn = False, return_WSI_feature = False):
        forward_return = {}
        B = x.shape[0]
        N = x.shape[1]
        h = self._fc1(x) 
        # fit for PPEG
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) 
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)
        h = self.amdlayer1(h)['amd_out']
        h = self.pos_layer(h, _H, _W) 
        amd_return = self.amdlayer2(h, return_WSI_attn)
        h = amd_return['amd_out']
        h = self.norm(h)[:,:1] 
        h = h.squeeze(1)
        logits = self._fc2(h)
        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = h
        if return_WSI_attn:
            forward_return['WSI_attn'] = amd_return['WSI_attn'][:,:,0,1:N+1].mean(1).transpose(0,1)
        return forward_return
