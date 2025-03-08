from math import ceil
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, reduce

# helper functions

def exists(val):
    return val is not None

def moore_penrose_iter_pinv(x, iters = 6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z

# main attention class
class NystromAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        residual = True,
        residual_conv_kernel = 33,
        eps = 1e-8,
        dropout = 0.,
        n_token = 1
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head
        self.n_token = n_token

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x, mask = None, return_attn = False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value = 0)

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value = False)

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # set masked positions to 0 in queries, keys, values

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l = l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l = l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l = l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)

        q_landmarks /= divisor
        k_landmarks /= divisor

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        attn1 = einsum(einops_eq, q, k_landmarks)
        attn2 = einsum(einops_eq, q_landmarks, k_landmarks)
        attn3 = einsum(einops_eq, q_landmarks, k)

        # masking

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (attn1, attn2, attn3))
        attn2 = moore_penrose_iter_pinv(attn2, iters)
        out = (attn1 @ attn2) @ (attn3 @ v)

        # add depth-wise conv residual of values
        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        out = out[:, -n:]
        if return_attn:
            attn1 = attn1[:,:,:self.n_token] @ attn2
            attn1 = (attn1 @ attn3)
        
            return out, attn1.mean(1)

        return out

# transformer

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Nystromformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        attn_values_residual = True,
        attn_values_residual_conv_kernel = 33,
        attn_dropout = 0.,
        ff_dropout = 0.   
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, NystromAttention(dim = dim, dim_head = dim_head, heads = heads, num_landmarks = num_landmarks, pinv_iterations = pinv_iterations, residual = attn_values_residual, residual_conv_kernel = attn_values_residual_conv_kernel, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, dropout = ff_dropout))
            ]))

    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x
        return x