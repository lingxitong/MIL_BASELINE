import torch
import torch.nn as nn
import numpy as np
from .nystrom_attention import NystromAttention
from utils.process_utils import get_act
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
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

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x

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

class TRANS_MIL(nn.Module):
    def __init__(self,num_classes,dropout,act,in_dim):
        super(TRANS_MIL, self).__init__()
        self.in_dim = in_dim
        self.act = act
        self.dropout = dropout
        self.num_classes = num_classes
        self.pos_layer = PPEG(dim=512)
        #self.pos_layer = nn.Identity()
        # self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),nn.Dropout(0.25))
        self._fc1 = [nn.Linear(self.in_dim, 512)]

        self._fc1 += [act]

        if dropout:
            self._fc1 += [nn.Dropout(self.dropout)]

        #self._fc1 += [SwinEncoder(attn='swin',pool='none',n_heads=2,trans_conv=False)]
        
        self._fc1 = nn.Sequential(*self._fc1)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.num_classes)

        self.apply(initialize_weights)

    def forward(self, x, return_WSI_attn = False, return_WSI_feature = False):
        forward_return = {}
        n = x.shape[1]
        h = x.float() #[B, n, 1024]
        
        h = self._fc1(h) #[B, n, 512]
        if len(h.size()) == 2:
            h = h.unsqueeze(0)
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]


        h = self.norm(h)
        #---->cls_token
        cls_token = h[:, 0]
        #---->patch_tokens
        patch_tokens = h[:, 1:n+1]
        # print(h.shape)
        #---->predict
        logits = self._fc2(cls_token) #[B, n_classes]
        # Y_hat = torch.argmax(logits, dim=1)
        # Y_prob = F.softmax(logits, dim = 1)
        # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = cls_token # cls_token can be seen as the global WSI feature
        if return_WSI_attn:
            WSI_attn = torch.matmul(patch_tokens, cls_token.transpose(0, 1)).squeeze(0)
            forward_return['WSI_attn'] = WSI_attn
        return forward_return

