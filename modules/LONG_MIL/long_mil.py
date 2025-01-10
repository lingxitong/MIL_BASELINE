from __future__ import annotations
import math
import copy
import pdb
import numpy as np
from .rotary import apply_rotary_position_embeddings, Rotary2D
from torch import autograd
# try:
from xformers.ops import memory_efficient_attention
# except:
#     print('please install xformer')
try:
    from fla.layers import GatedLinearAttention
except:
    print('please install fla from https://github.com/sustcsonglin/flash-linear-attention')



import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def exists(val):
    return val is not None

def precompute_freqs_cis(dim: int, end: int, pos_idx, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    # t = pos_idx.cpu()
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    # pdb.set_trace()
    return freqs_cis[pos_idx.long()]

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,freqs_cis: torch.Tensor,):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.05, proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        # self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, freqs_cis=None, alibi=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4) #
        q, k, v = qkv[0], qkv[1], qkv[2]
        q,k = q.view(1,N,1,-1), k.view(1,N,1,-1)
        if exists(freqs_cis):
            q,k = apply_rotary_position_embeddings(freqs_cis, q, k)
        q, k = q.view(1, N, self.num_heads, -1), k.view(1, N, self.num_heads, -1)
        if exists(alibi):
            try:
                x = memory_efficient_attention(q, k, v, alibi,p=self.attn_drop).reshape(B, N, C)
            except:
                # print('xformer error')
                pass
        else:
            x = memory_efficient_attention(q,k,v,p=self.attn_drop).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, None

class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.05, proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.gla = GatedLinearAttention(d_model=dim, num_heads=num_heads,causal=False)#.to(device).to(dtype)

    def forward(self, x, freqs_cis=None, alibi=None):
        # pdb.set_trace()
        return self.gla(x) #, None

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512,num_heads=4,attn_drop=0.05,mlp_ratio=4.):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = Attention(dim=dim, num_heads=num_heads,attn_drop=attn_drop)
        # self.attn = LinearAttention(dim=dim,num_heads=num_heads)

    def forward(self, x, rope=None, alibi=None):
        temp = self.attn(self.norm(x), rope, alibi) #[0], None
        return x+temp[0], temp[1]


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class TransBlock(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512,num_heads=4,attn_drop=0.05,mlp_ratio=4.,drop_path=0.):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = Attention(dim=dim, num_heads=num_heads,attn_drop=attn_drop)
        # self.attn = LinearAttention(dim=dim,num_heads=num_heads)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x, rope=None, alibi=None):
        temp = self.attn(self.norm(x), rope, alibi)   # attn for spatial mixture
        # return x+temp[0], temp[1]
        x = x + self.drop_path(temp[0])
        x = x + self.drop_path(self.mlp(self.norm2(x))) # mlp for token mixture
        return x, temp[1]

class LinearTransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512,num_heads=4):
        super().__init__()
        self.norm = norm_layer(dim)
        # self.attn = Attention(dim=dim,num_heads=num_heads)
        self.attn = LinearAttention(dim=dim,num_heads=num_heads)

    def forward(self, x, rope, alibi):
        return x + self.attn(self.norm(x)), None



def get_slopes(n_heads: int):
    n = 2 ** math.floor(math.log2(n_heads))
    m_0 = 2.0 ** (-8.0 / n)
    m = torch.pow(m_0, torch.arange(1, 1 + n))

    if n < n_heads:
        m_hat_0 = 2.0 ** (-4.0 / n)
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        m = torch.cat([m, m_hat])
    return m

class LONG_MIL_V1(nn.Module):
    def __init__(self, num_classes, in_dim,hidden_dim,alibi_positon_embedding_path):
        super(LONG_MIL_V1, self).__init__()
        self.n_heads = 4
        self.in_dim = in_dim
        feat_size = hidden_dim
        self.feat_size = feat_size
        self._fc1 = nn.Sequential(nn.Linear(in_dim, feat_size), nn.ReLU())
        self.n_classes = num_classes

        self.layer1 = TransBlock(dim=feat_size, num_heads=self.n_heads)
        self.layer2 = TransBlock(dim=feat_size, num_heads=self.n_heads)
        self.layer3 = TransBlock(dim=feat_size, num_heads=self.n_heads)

        # optional 1
        # self.layer3 = LinearTransLayer(dim=feat_size,num_heads=self.n_heads,)
        # optional 2
        # self.layer3 = MambaLayer(dim=feat_size)

        self.norm = nn.LayerNorm(feat_size)
        self._fc2 = nn.Linear(feat_size, self.n_classes)
        self.rotary = Rotary2D(dim=feat_size)
        self.alibi = torch.load(alibi_positon_embedding_path,weights_only=True)
        self.slope_m = get_slopes(self.n_heads).to(torch.float16)


    def forward(self, x,  return_WSI_attn = False, return_WSI_feature = False):
        '''
        :param x:  shape = N * (feat_size+2), 2 for x-y 2d-position
        :return:
        '''
        self.alibi = self.alibi.to(x.device)
        self.slope_m = self.slope_m.to(x.device)
        forward_return = {}
        x = x.squeeze(0)
        h = x[:, :self.in_dim].unsqueeze(0)
        h = self._fc1(h)

        # option 1 use local_alibi +global_alibi
        freqs_cis, alibi_local, alibi_global = self.positional_embedding(x, use_alibi=True)
        # option 2 combine alibi and rope
        # freqs_cis, alibi_local, alibi_global = self.positional_embedding(x,use_rope=True,use_alibi=True)
        # option 3 use rope only
        # freqs_cis, alibi_local, alibi_global = self.positional_embedding(x,use_rope=True)


        h, _ = self.layer1(h, freqs_cis, alibi_local)
        h, _ = self.layer2(h, freqs_cis, alibi_local)
        h, _ = self.layer3(h, freqs_cis, alibi_global)
        h_ori = h.clone()
        h = self.norm(h.mean(1))

        # ---->predict
        logits = self._fc2(h)  # [1, n_classes]
        Y_hat = torch.argmax(logits, dim=-1)
        Y_prob = F.softmax(logits, dim=-1)
        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = h
        if return_WSI_attn:
            WSI_attn = torch.matmul(h_ori, h.unsqueeze(-1)).squeeze(0)
            print('WSI_attn',WSI_attn.shape)
            forward_return['WSI_attn'] = WSI_attn
        return forward_return

    def positional_embedding(self, x, use_alibi=False, use_rope=False):
        # scale = 1  # for 20x 224 with 112 overlap (or 40x 224)
        scale = 2  # for 20x 224 with 0 overlap
        shape = 112 # or 128
        # shape = 128
        freqs_cis = None
        alibi_bias = None
        alibi_bias2 = None
        if use_rope or use_alibi:
            abs_pos = x[::, -2:]
            # pdb.set_trace()
            # print(abs_pos)
            x_pos, y_pos = abs_pos[:, 0], abs_pos[:, 1]
            x_pos = torch.round((x_pos - x_pos.min()) / (shape * scale) / 4)
            y_pos = torch.round((y_pos - y_pos.min()) / (shape * scale) / 4)
            H, W = 600 // scale, 600 // scale
            selected_idx = (x_pos * W + y_pos).to(torch.int)

            if use_rope:
                pos_cached = self.rotary.forward(torch.tensor([H, W]))
                freqs_cis = pos_cached[selected_idx].cuda()
            if use_alibi:
                alibi_bias = self.alibi[selected_idx, :][:, selected_idx]

                alibi_bias = alibi_bias[:, :, None] * self.slope_m[None, None, :]
                # pdb.set_trace()
                alibi_bias = alibi_bias.permute(2, 0, 1).unsqueeze(0)#.float()

                shape3 = alibi_bias.shape[3]
                pad_num = 8 - shape3 % 8 # to tackle xformer problems
                padding_bias = torch.zeros(1, alibi_bias.shape[1], alibi_bias.shape[2], pad_num).to(alibi_bias.device)
                alibi_bias = torch.cat([alibi_bias, padding_bias], dim=-1)
                alibi_bias = autograd.Variable(alibi_bias.contiguous())[:, :, :, :shape3]
                alibi_bias2 = copy.deepcopy(alibi_bias)
                temp_min = alibi_bias2.min()
                alibi_bias2[torch.where(alibi_bias2 == temp_min)] = -torch.inf # masked out longer distances

        return freqs_cis, alibi_bias2, alibi_bias


class LONG_MIL(nn.Module):
    def __init__(self, num_classes, in_dim,hidden_dim,alibi_positon_embedding_path,version = 'v1'):
        super(LONG_MIL, self).__init__()
        if version == 'v1':
            self.model = LONG_MIL_V1(num_classes, in_dim,hidden_dim,alibi_positon_embedding_path)
        elif version == 'v2':
            raise NotImplementedError
        
    def forward(self, x,  return_WSI_attn = False, return_WSI_feature = False):
        return self.model(x, return_WSI_attn, return_WSI_feature)