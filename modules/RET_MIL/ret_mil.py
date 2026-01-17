"""
RET_MIL - Retention-based Multiple Instance Learning
Reference: https://github.com/Hongbo-Chu/RetMIL
"""
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


def get_activation_fn(activation):
    if activation == "swish":
        return F.silu
    elif activation == "gelu":
        return F.gelu
    else:
        raise NotImplementedError


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


class RetNetRelPos(nn.Module):
    def __init__(self, embed_dim, retention_heads, hidden_dim):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // retention_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(1 - 2 ** (-5 - torch.arange(retention_heads, dtype=torch.float)))
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)
        self.recurrent_chunk_size = hidden_dim
        
    def forward(self, slen, activate_recurrent=False, chunkwise_recurrent=False):
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen - 1))
            cos = torch.cos(self.angle * (slen - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())
        elif chunkwise_recurrent:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])

            block_index = torch.arange(self.recurrent_chunk_size).to(self.decay)
            mask = torch.tril(torch.ones(self.recurrent_chunk_size, self.recurrent_chunk_size).to(self.decay))
            mask = torch.masked_fill(block_index[:, None] - block_index[None, :], ~mask.bool(), float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            
            value_inner_decay = mask[:, -1] / mask[:, -1].sum(dim=-1, keepdim=True)
            value_inner_decay = value_inner_decay.unsqueeze(-1)
            scale = mask.sum(dim=-1, keepdim=True).sqrt()
            inner_mask = mask / scale

            cross_decay = torch.exp(self.decay * self.recurrent_chunk_size)
            query_inner_decay = torch.exp(self.decay[:, None] * (block_index + 1))
            query_inner_decay = query_inner_decay[:, :, None] / (scale / mask[:, -1].sum(dim=-1)[:, None, None])
            cross_decay = cross_decay[:, None, None]
            retention_rel_pos = ((sin, cos), (inner_mask, cross_decay, query_inner_decay, value_inner_decay))
        else:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])
            mask = torch.tril(torch.ones(slen, slen).to(self.decay))
            mask = torch.masked_fill(index[:, None] - index[None, :], ~mask.bool(), float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=512, D=256, dropout=False, n_classes=1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # chunk * N * n_classes
        return A, x


class Retention(nn.Module):
    def __init__(self, embed_dim, num_heads, gate_fn='swish'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads 

        self.gate_fn = get_activation_fn(activation=str(gate_fn))

        self.ret_rel_pos = RetNetRelPos(embed_dim=embed_dim, retention_heads=num_heads, hidden_dim=self.embed_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.g_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.scaling = self.embed_dim ** -0.5

        self.group_norm = RMSNorm(self.embed_dim, eps=1e-6, elementwise_affine=False)
    
    def parallel_forward(self, qr, kr, v, mask):
        bsz, tgt_len, embed_dim = v.size()

        vr = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        qk_mat = qr @ kr.transpose(-1, -2)  # bsz * m * tgt_len * tgt_len
        qk_mat = qk_mat * mask
        # invariant after normalization
        qk_mat = qk_mat / qk_mat.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1, max=5e4)
        output = torch.matmul(qk_mat, vr)
        output = output.transpose(1, 2)
        return output
    
    def forward(self, x):
        bsz, tgt_len, _ = x.size()

        (sin, cos), inner_mask = self.ret_rel_pos(slen=tgt_len, activate_recurrent=False, chunkwise_recurrent=False)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = self.g_proj(x)

        k *= self.scaling
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        output = self.parallel_forward(qr, kr, v, inner_mask)  # three methods, now just the parallel method

        output = self.group_norm(output).reshape(bsz, tgt_len, self.head_dim * self.num_heads)

        output = self.gate_fn(g) * output

        return output


class RetMIL(nn.Module):
    """
    Original RetMIL model (internal use)
    """
    def __init__(self, embed_dim, num_heads, chunk_size=512, n_classes=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.chunk_size = chunk_size

        self.local_retention = Retention(embed_dim=self.embed_dim, num_heads=self.num_heads, gate_fn='swish')
        self.local_attn_pool = Attn_Net_Gated(L=self.embed_dim, D=self.embed_dim // 2, dropout=0.25, n_classes=1)
        self.global_retention = Retention(embed_dim=self.embed_dim, num_heads=self.num_heads, gate_fn='swish')
        self.global_attn_pool = Attn_Net_Gated(L=self.embed_dim, D=self.embed_dim // 2, dropout=0.25, n_classes=1)
        self.classifier = nn.Linear(embed_dim, n_classes, bias=False)

    def forward(self, x, return_attn=False):
        B, N, C = x.shape

        chunk_size = self.chunk_size
        
        # Chunk splitting
        _N = (N // chunk_size) * chunk_size
        c_x, r_x = x.split((_N, N - _N), dim=1)  # chunk_x, remain_x
        
        if N - _N != 0:
            # Handle remainder: repeat remainder to fill chunk_size
            full_repeats = chunk_size // (N - _N)
            partial_repeat = chunk_size % (N - _N)

            full_repeat_r_x = r_x.repeat(1, full_repeats, 1)
            partial_r_x = r_x[:, :partial_repeat, :]
            added_x = torch.cat([full_repeat_r_x, partial_r_x], dim=1)
            d_x = torch.cat([c_x, added_x], dim=1)
        else:
            d_x = x

        # Reshape to [B, num_chunks, chunk_size, C]
        num_chunks = d_x.shape[1] // chunk_size
        d_x = d_x.reshape(B, num_chunks, chunk_size, C)

        # Local Level: process each chunk
        # Reshape [B, num_chunks, chunk_size, C] to [B*num_chunks, chunk_size, C] for processing
        B_chunks = B * num_chunks
        d_x_flat = d_x.view(B_chunks, chunk_size, C)
        local_x = self.local_retention(d_x_flat)  # [B*num_chunks, chunk_size, C]
        local_x = local_x.view(B, num_chunks, chunk_size, C)  # [B, num_chunks, chunk_size, C]

        # Local attention pooling
        # Process each batch and chunk separately
        local_fe_sq_list = []
        for b in range(B):
            local_x_b = local_x[b]  # [num_chunks, chunk_size, C]
            local_A, local_x_b = self.local_attn_pool(local_x_b)  # local_A: [num_chunks, chunk_size, 1], local_x_b: [num_chunks, chunk_size, C]
            local_A = torch.transpose(local_A, 2, 1)  # [num_chunks, 1, chunk_size]
            local_A = F.softmax(local_A, dim=2)
            local_fe_sq_b = torch.matmul(local_A, local_x_b)  # [num_chunks, 1, C]
            local_fe_sq_b = local_fe_sq_b.squeeze(1)  # [num_chunks, C]
            local_fe_sq_list.append(local_fe_sq_b)
        local_fe_sq = torch.stack(local_fe_sq_list, dim=0)  # [B, num_chunks, C]

        # Global Level: process chunk embeddings
        # Process each batch separately
        global_fe_list = []
        global_A_all = []  # Store global attention weights
        for b in range(B):
            global_x_b = self.global_retention(local_fe_sq[b].unsqueeze(0))  # [1, num_chunks, C]
            global_A, global_x_b = self.global_attn_pool(global_x_b)  # global_A: [1, num_chunks, 1], global_x_b: [1, num_chunks, C]
            global_A = torch.transpose(global_A, 2, 1)  # [1, 1, num_chunks]
            global_A = F.softmax(global_A, dim=2)
            global_A_all.append(global_A.squeeze(0).squeeze(0))  # [num_chunks]
            global_fe_b = torch.matmul(global_A, global_x_b)  # [1, 1, C]
            global_fe_list.append(global_fe_b.squeeze(0))  # [1, C]
        global_fe = torch.stack(global_fe_list, dim=0)  # [B, 1, C]
        global_A_all = torch.stack(global_A_all, dim=0)  # [B, num_chunks]

        # Classification
        output = self.classifier(global_fe.squeeze(1))  # [B, num_classes]

        if return_attn:
            # Return attention scores: use global attention weights
            # Expand global attention weights to original patch level
            # global_A_all shape: [B, num_chunks]
            # Distribute each chunk's attention evenly to all patches in that chunk
            attn_scores_list = []
            for b in range(B):
                attn_b = []
                for chunk_idx in range(num_chunks):
                    chunk_attn = global_A_all[b, chunk_idx].item()
                    # Each chunk has chunk_size patches
                    attn_b.extend([chunk_attn] * chunk_size)
                # Handle remainder part (if exists)
                if N - _N > 0:
                    # Use last chunk's attention for remainder part
                    if num_chunks > 0:
                        remainder_attn = global_A_all[b, -1].item()
                    else:
                        remainder_attn = 1.0 / N
                    attn_b.extend([remainder_attn] * (N - _N))
                attn_scores_list.append(torch.tensor(attn_b[:N], device=output.device))  # Truncate to original length
            attn_scores = torch.stack(attn_scores_list, dim=0)  # [B, N]
            return output, attn_scores
        else:
            return output


class RET_MIL(nn.Module):
    """
    RET_MIL - Retention-based Multiple Instance Learning
    Framework-adapted wrapper class
    """
    def __init__(self, in_dim, embed_dim=None, num_classes=2, chunk_len=512, num_heads=8, dropout=0.25):
        super().__init__()
        # If embed_dim is not specified, use in_dim
        if embed_dim is None:
            embed_dim = in_dim
        
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.chunk_len = chunk_len
        
        # If in_dim != embed_dim, add projection layer
        if in_dim != embed_dim:
            self.proj = nn.Linear(in_dim, embed_dim)
        else:
            self.proj = None
        
        # Create RetMIL model
        self.retmil = RetMIL(
            embed_dim=embed_dim,
            num_heads=num_heads,
            chunk_size=chunk_len,
            n_classes=num_classes
        )
    
    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False):
        """
        Forward pass
        
        Args:
            x: input features, shape [N, D] or [1, N, D]
            return_WSI_attn: whether to return attention weights
            return_WSI_feature: whether to return slide-level feature
        
        Returns:
            dict with keys:
                - 'logits': classification logits, shape [1, num_classes]
                - 'WSI_attn' (optional): attention weights
                - 'WSI_feature' (optional): slide-level feature
        """
        forward_return = {}
        
        # Input adaptation: [N, D] -> [1, N, D]
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        
        B, N, D = x.shape
        
        # Project to embed_dim (if needed)
        if self.proj is not None:
            x = self.proj(x)
        
        # Call RetMIL
        if return_WSI_attn:
            logits, attn_scores = self.retmil(x, return_attn=True)  # [B, num_classes], [B, N]
            # Take first batch (usually B=1)
            if B == 1:
                attn_scores = attn_scores.squeeze(0)  # [N]
            else:
                attn_scores = attn_scores[0]  # [N]
        else:
            logits = self.retmil(x, return_attn=False)  # [B, num_classes]
        
        # Output adaptation
        forward_return['logits'] = logits
        
        if return_WSI_feature:
            # Can extract from RetMIL's intermediate output, but requires modifying RetMIL.forward
            # Return None for now, can be implemented later
            forward_return['WSI_feature'] = None
        
        if return_WSI_attn:
            forward_return['WSI_attn'] = attn_scores
        
        return forward_return
