import torch
import torch.nn as nn
from .nystrom_attention import NystromAttention
from .rmsa import RegionAttntion
from timm.models.layers import DropPath
import numpy as np

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # H, W = self.input_resolution
        B, L, C = x.shape
        # padding
        H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
        _n = -H % 2
        H, W = H+_n, W+_n
        add_length = H * W - L
        if add_length > 0:
            x = torch.cat([x, torch.zeros((B,add_length,C),device=x.device)],dim = 1)

        # H,W = int(L**0.5),int(L**0.5)
        # assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
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

class TransLayer1(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512,head=8,drop_out=0.1,drop_path=0.,need_down=False,need_reduce=False,down_ratio=2,ffn=False,ffn_act='gelu',mlp_ratio=4.,trans_dim=64,n_cycle=1,attn='ntrans',n_region=8,epeg=False,shift_size=False,region_size=0,rpe=False,min_region_num=0,min_region_ratio=0,qkv_bias=True,**kwargs):
        super().__init__()

        if need_reduce:
            self.reduction = nn.Linear(dim, dim//down_ratio, bias=False)
            dim = dim // down_ratio
        else:
            self.reduction = nn.Identity()
        
        self.norm = norm_layer(dim)
        self.norm2 = norm_layer(dim) if ffn else nn.Identity()
        if attn == 'ntrans':
            self.attn = NystromAttention(
                dim = dim,
                dim_head = trans_dim,  # dim // 8
                heads = head,
                num_landmarks = 256,    # number of landmarks dim // 2
                pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
                residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
                dropout=drop_out
            )
        elif attn == 'rrt':
            self.attn = RegionAttntion(
                dim=dim,
                num_heads=head,
                drop=drop_out,
                region_num=n_region,
                head_dim=trans_dim,
                conv=epeg,
                shift_size=shift_size,
                region_size=region_size,
                rpe=rpe,
                min_region_num=min_region_num,
                min_region_ratio=min_region_ratio,
                qkv_bias=qkv_bias,
                **kwargs
            )
        else:
            raise NotImplementedError
        # elif attn == 'rrt1d':
        #     self.attn = RegionAttntion1D(
        #         dim=dim,
        #         num_heads=head,
        #         drop=drop_out,
        #         region_num=n_region,
        #         head_dim=trans_dim,
        #         conv=epeg,
        #         **kwargs
        #     )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = ffn
        act_layer = nn.GELU if ffn_act == 'gelu' else nn.ReLU
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop_out) if ffn else nn.Identity()

        self.downsample = PatchMerging(None,dim) if need_down else nn.Identity()

        self.n_cycle = n_cycle

    def forward(self,x,need_attn=False):
        attn = None
        for i in range(self.n_cycle):
            x,attn = self.forward_trans(x,need_attn=need_attn)
        
        if need_attn:
            return x,attn
        else:
            return x

    def forward_trans(self, x, need_attn=False):
        attn = None

        x = self.reduction(x)
        B, L, C = x.shape
        
        if need_attn:
            z,attn = self.attn(self.norm(x),return_attn=need_attn)
        else:
            z = self.attn(self.norm(x))

        x = x+self.drop_path(z)

        # FFN
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = self.downsample(x)

        return x,attn

