import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import numpy as np
from .nystrom_attention import NystromAttention

# --------------------------------------------------------
# Modified by Swin@Microsoft
# --------------------------------------------------------

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

def region_partition(x, region_size):
    """
    Args:
        x: (B, H, W, C)
        region_size (int): region size
    Returns:
        regions: (num_regions*B, region_size, region_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // region_size, region_size, W // region_size, region_size, C)
    regions = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, region_size, region_size, C)
    return regions

def region_reverse(regions, region_size, H, W):
    """
    Args:
        regions: (num_regions*B, region_size, region_size, C)
        region_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(regions.shape[0] / (H * W / region_size / region_size))
    x = regions.view(B, H // region_size, W // region_size, region_size, region_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class InnerAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted region.
    Args:
        dim (int): Number of input channels.
        region_size (tuple[int]): The height and width of the region.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, head_dim=None, region_size=None, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,conv=True,conv_k=15,conv_2d=False,conv_bias=True,conv_type='attn'):

        super().__init__()
        self.dim = dim
        self.region_size = [region_size,region_size] if region_size is not None else None  # Wh, Ww
        self.num_heads = num_heads
        if head_dim is None:
            head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        if region_size is not None:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * self.region_size[0] - 1) * (2 * self.region_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the region
            coords_h = torch.arange(self.region_size[0])
            coords_w = torch.arange(self.region_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.region_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.region_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.region_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(dim, head_dim * num_heads * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(head_dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.conv_2d = conv_2d
        self.conv_type = conv_type
        if conv:
            kernel_size = conv_k
            padding = kernel_size // 2
            
            if conv_2d:
                if conv_type == 'attn':
                    self.pe = nn.Conv2d(num_heads, num_heads, kernel_size, padding = padding, groups = num_heads, bias = conv_bias)
                else:
                    self.pe = nn.Conv2d(head_dim * num_heads, head_dim * num_heads, kernel_size, padding = padding, groups = head_dim * num_heads, bias = conv_bias)
            else:
                if conv_type == 'attn':
                    self.pe = nn.Conv2d(num_heads, num_heads, (kernel_size, 1), padding = (padding, 0), groups = num_heads, bias = conv_bias)
                else:
                    self.pe = nn.Conv2d(head_dim *num_heads, head_dim *num_heads, (kernel_size, 1), padding = (padding, 0), groups = head_dim *num_heads, bias = conv_bias)
                # self.pe = nn.Conv2d(num_heads, num_heads, (1, kernel_size), padding = (0, padding), groups = num_heads, bias = conv_bias)
        else:
            self.pe = None

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_regions*B, N, C)
            mask: (0/-inf) mask with shape of (num_regions, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        # x = self.pe(x)

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.pe is not None and self.conv_type == 'attn':
            #print(attn.size())
            # B,H,N,N ->B,H,N,N-0.5,N-0.5
            # if self.conv_2d:
            #     pe = self.pe(attn.permute(0,2,1,3).reshape(-1,self.num_heads,int(np.ceil(np.sqrt(N))),int(np.ceil(np.sqrt(N)))))
            #     attn = attn+pe.reshape(B_,N,self.num_heads,N).transpose(1,2)
            # else:
            pe = self.pe(attn)
            attn = attn+pe

        if self.region_size is not None:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.region_size[0] * self.region_size[1], self.region_size[0] * self.region_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        if self.pe is not None and self.conv_type == 'value_bf':
            # B,H,N,C -> B,HC,N-0.5,N-0.5 
            pe = self.pe(v.permute(0,3,1,2).reshape(B_,C,int(np.ceil(np.sqrt(N))),int(np.ceil(np.sqrt(N)))))
            #pe = torch.einsum('ahbd->abhd',pe).flatten(-2,-1)
            v = v + pe.reshape(B_,self.num_heads, self.head_dim,N).permute(0,1,3,2)

        # print(v.size())

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.num_heads*self.head_dim)
        
        if self.pe is not None and self.conv_type == 'value_af':
            #print(v.size())
            pe = self.pe(v.permute(0,3,1,2).reshape(B_,C,int(np.ceil(np.sqrt(N))),int(np.ceil(np.sqrt(N)))))
            # print(pe.size())
            # print(v.size())
            x = x + pe.reshape(B_,self.num_heads*self.head_dim,N).transpose(-1,-2)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, region_size={self.region_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 region with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class RegionAttntion(nn.Module):
    def __init__(self, dim, input_resolution=None, head_dim=None,num_heads=8, region_size=0, shift_size=False, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., region_num=8,conv=False,rpe=False,min_region_num=0,min_region_ratio=0.,region_attn='native',**kawrgs):
        super().__init__()
 
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.region_size = region_size if region_size > 0 else None
        self.shift_size = shift_size
        self.region_num = region_num
        self.min_region_num = min_region_num
        self.min_region_ratio = min_region_ratio
        self.rpe = rpe
        
        if self.region_size is not None:
            self.region_num = None
        self.fused_region_process = False

        if region_attn == 'native':
            self.attn = InnerAttention(
                dim, head_dim=head_dim,num_heads=num_heads,region_size=self.region_size if rpe else None,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,conv=conv,**kawrgs)
        elif region_attn == 'ntrans':
            self.attn = NystromAttention(
                dim=dim,
                dim_head=head_dim,
                heads=num_heads,
                dropout=drop
            )

        self.attn_mask = None

    def padding(self,x):
        B, L, C = x.shape
        if self.region_size is not None:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            # print(L)
            # print(H)
            _n = -H % self.region_size
            H, W = H+_n, W+_n
            region_num = int(H // self.region_size)
            region_size = self.region_size
        else:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % self.region_num
            H, W = H+_n, W+_n
            region_size = int(H // self.region_num)
            region_num = self.region_num
        
        add_length = H * W - L
        if (add_length > L / (self.min_region_ratio+1e-8) or L < self.min_region_num) and not self.rpe:
            H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
            _n = -H % 2
            H, W = H+_n, W+_n
            add_length = H * W - L
            region_size = H
        if add_length > 0:
            x = torch.cat([x, torch.zeros((B,add_length,C),device=x.device)],dim = 1)
        
        return x,H,W,add_length,region_num,region_size

    def forward(self,x,return_attn=False):
        B, L, C = x.shape
        
        # padding
        x,H,W,add_length,region_num,region_size = self.padding(x)

        x = x.view(B, H, W, C)

        # partition regions
        x_regions = region_partition(x, region_size)  # nW*B, region_size, region_size, C

        x_regions = x_regions.view(-1, region_size * region_size, C)  # nW*B, region_size*region_size, C

        # R-MSA
        attn_regions = self.attn(x_regions, mask=self.attn_mask)  # nW*B, region_size*region_size, C

        # merge regions
        attn_regions = attn_regions.view(-1, region_size, region_size, C)

        x = region_reverse(attn_regions, region_size, H, W)  # B H' W' C

        x = x.view(B, H * W, C)

        if add_length >0:
            x = x[:,:-add_length]

        return x
