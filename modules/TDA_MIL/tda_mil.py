"""
TDA_MIL - Top-Down Attention-based Multiple Instance Learning
Reference: Top-Down Attention-based Multiple Instance Learning for Whole Slide Image Analysis

Implementation based on the paper:
- Online aggregation stage only (patch features are pre-extracted)
- Inference Step I: Self-Attention contextualization
- Feature Selection Module: Eq.(3)(4) with task relevance token and channel rescaling
- Inference Step II: Top-Down Injection (only V changes, Q and K remain from Step I)

Key equations:
- Eq.(3): s_i = clamp(cos_sim(x_i,BU, T))
- Eq.(4): x_i,TD = C · s_i · x_i,BU
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class TDAMultiheadAttention(nn.Module):
    """
    Multi-head attention that supports separate value input (for top-down injection)
    
    Standard: Q, K, V all from same input
    Top-down: Q, K from x_qk, V from x_v (which can be x_BU + x_TD)
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_qk, x_v=None, mask=None):
        """
        Args:
            x_qk: (B, N, embed_dim) - input for Q and K
            x_v: (B, N, embed_dim) - input for V (if None, use x_qk)
            mask: (B, N) - attention mask (True for valid tokens)
        Returns:
            out: (B, N, embed_dim)
        """
        B, N, _ = x_qk.shape
        
        if x_v is None:
            x_v = x_qk
        
        # Project to Q, K, V
        Q = self.q_proj(x_qk).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        K = self.k_proj(x_qk).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        V = self.v_proj(x_v).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        
        # Attention scores
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        
        # Apply mask if provided
        if mask is not None:
            # mask: (B, N) -> (B, 1, 1, N) for broadcasting
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn = attn.masked_fill(~mask_expanded, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ V).transpose(1, 2).contiguous().view(B, N, self.embed_dim)  # (B, N, embed_dim)
        out = self.out_proj(out)
        
        return out

class TransformerBlock(nn.Module):
    """
    Standard Transformer encoder block: LN -> MSA -> residual -> MLP -> residual
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = TDAMultiheadAttention(embed_dim, num_heads, dropout=attn_dropout)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, x_v=None, mask=None):
        """
        Args:
            x: (B, N, embed_dim) - input for Q and K
            x_v: (B, N, embed_dim) - input for V (for top-down injection, if None use x)
            mask: (B, N) - attention mask
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        if x_v is None:
            attn_out = self.attn(x_norm, x_norm, mask)
        else:
            x_v_norm = self.norm1(x_v)
            attn_out = self.attn(x_norm, x_v_norm, mask)
        x = x + self.dropout1(attn_out)
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x

class FeatureSelection(nn.Module):
    """
    Feature Selection Module: Eq.(3)(4)
    
    Eq.(3): s_i = clamp(cos_sim(x_i,BU, T))
    Eq.(4): x_i,TD = C · s_i · x_i,BU
    
    Then: x_i,TD = MLP_td(x_i,TD)
    """
    def __init__(self, embed_dim, td_mlp_ratio=2.0, dropout=0.1, clamp_min=0.0, clamp_max=1.0, force_cls_score=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.force_cls_score = force_cls_score
        
        # Learnable task relevance token: T ∈ R^d
        self.task_relevance_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.normal_(self.task_relevance_token, std=0.02)
        
        # Learnable channel rescaling matrix: C ∈ R^{d×d}
        self.channel_rescaling = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # MLP decoder
        mlp_hidden_dim = int(embed_dim * td_mlp_ratio)
        self.mlp_td = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x_bu, mask=None):
        """
        Args:
            x_bu: (B, N, embed_dim) - bottom-up features from Step I
            mask: (B, N) - mask for valid tokens (optional)
        Returns:
            x_td: (B, N, embed_dim) - top-down features
            sel_scores: (B, N) - selection scores s_i
        """
        B, N, d = x_bu.shape
        
        # Normalize for cosine similarity
        x_bu_norm = F.normalize(x_bu, p=2, dim=-1)  # (B, N, d)
        T_norm = F.normalize(self.task_relevance_token, p=2, dim=-1)  # (1, 1, d)
        
        # Compute cosine similarity: cos_sim(x_i, T)
        # x_bu_norm: (B, N, d), T_norm: (1, 1, d)
        cos_sim = torch.sum(x_bu_norm * T_norm, dim=-1)  # (B, N)
        
        # Clamp to [clamp_min, clamp_max]
        sel_scores = torch.clamp(cos_sim, self.clamp_min, self.clamp_max)  # (B, N)
        
        # Force CLS token score (if first token is CLS)
        if self.force_cls_score is not None:
            sel_scores[:, 0] = self.force_cls_score
        
        # Apply mask: set padding tokens to 0
        if mask is not None:
            sel_scores = sel_scores * mask.float()
        
        # Eq.(4): x_i,TD = C · s_i · x_i,BU
        # s_i is scalar, broadcast to (B, N, 1)
        sel_scores_expanded = sel_scores.unsqueeze(-1)  # (B, N, 1)
        x_td = self.channel_rescaling(sel_scores_expanded * x_bu)  # (B, N, d)
        
        # MLP decoder
        x_td = self.mlp_td(x_td)
        
        return x_td, sel_scores

class TDA_MIL(nn.Module):
    """
    Top-Down Attention-based Multiple Instance Learning
    
    Implementation based on paper:
    - (0) Dim reduction + CLS token
    - (1) Inference Step I: Self-Attention contextualization (l transformer blocks)
    - (2) Feature Selection Module: Eq.(3)(4)
    - (3) Inference Step II: Top-Down Injection (only V changes, Q and K from Step I)
    - (4) Output: CLS token -> classifier
    
    Reference: Top-Down Attention-based Multiple Instance Learning for Whole Slide Image Analysis
    """
    def __init__(self, in_dim=1024, embed_dim=512, num_classes=2, num_layers=2, num_heads=8,
                 mlp_ratio=4.0, dropout=0.1, attn_dropout=0.1, td_mlp_ratio=2.0,
                 clamp_min=0.0, clamp_max=1.0, force_cls_score=1.0, share_weights_step12=True,
                 max_seq_len=2048):
        super(TDA_MIL, self).__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.share_weights_step12 = share_weights_step12
        self.max_seq_len = max_seq_len  # Limit sequence length to avoid OOM
        
        # (0) Dim reduction: D -> d
        self.dim_reduction = nn.Linear(in_dim, embed_dim)
        
        # CLS token: learnable CLS ∈ R^{1×d}
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # (1) Inference Step I: Transformer encoder blocks
        self.blocks_step1 = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout)
            for _ in range(num_layers)
        ])
        
        # (2) Feature Selection Module
        self.feature_selection = FeatureSelection(
            embed_dim, td_mlp_ratio, dropout, clamp_min, clamp_max, force_cls_score
        )
        
        # (3) Inference Step II: Transformer blocks (shared or separate)
        if share_weights_step12:
            # Reuse same blocks, but with value injection
            self.blocks_step2 = self.blocks_step1
        else:
            self.blocks_step2 = nn.ModuleList([
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout)
                for _ in range(num_layers)
            ])
        
        # (4) Output: CLS token -> classifier
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        self.apply(initialize_weights)
    
    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False, mask=None):
        """
        Args:
            x: (N, D) or (1, N, D) - patch features
            return_WSI_attn: bool - return attention/selection scores
            return_WSI_feature: bool - return bag-level feature
            mask: (N,) or (1, N) - attention mask (True for valid tokens)
        Returns:
            dict with keys:
                - 'logits': (1, num_classes)
                - 'WSI_attn': (N,) - selection scores (optional)
                - 'WSI_feature': (embed_dim,) - CLS token feature (optional)
                - 'aux': dict with additional info (optional)
        """
        forward_return = {}
        
        # Handle input format
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # (N, D) -> (1, N, D)
        
        B, N_original, D = x.shape
        N = N_original
        
        # Handle mask
        if mask is not None:
            if len(mask.shape) == 1:
                mask = mask.unsqueeze(0)  # (N,) -> (1, N)
        else:
            # Default: all tokens are valid
            mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
        
        # Limit sequence length to avoid OOM
        sampled_indices = None
        if N > self.max_seq_len:
            # Randomly sample max_seq_len patches
            indices = torch.randperm(N, device=x.device)[:self.max_seq_len]
            indices = torch.sort(indices)[0]  # Keep original order
            sampled_indices = indices.cpu().numpy()  # Save for mapping back
            x = x[:, indices, :]
            mask = mask[:, indices]
            N = self.max_seq_len
        
        # (0) Dim reduction: D -> d
        x = self.dim_reduction(x)  # (B, N, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, embed_dim)
        
        # Update mask for CLS token (CLS is always valid)
        mask_cls = torch.ones(B, 1, dtype=torch.bool, device=x.device)
        mask = torch.cat([mask_cls, mask], dim=1)  # (B, N+1)
        
        # (1) Inference Step I: Self-Attention contextualization
        x_bu = x
        for block in self.blocks_step1:
            x_bu = block(x_bu, x_v=None, mask=mask)
        
        # (2) Feature Selection Module: Eq.(3)(4)
        x_td, sel_scores = self.feature_selection(x_bu, mask=mask)  # x_td: (B, N+1, embed_dim), sel_scores: (B, N+1)
        
        # (3) Inference Step II: Top-Down Injection
        # Q, K from x_bu (unchanged), V from (x_bu + x_td)
        x_v_injected = x_bu + x_td  # (B, N+1, embed_dim)
        
        x_step2 = x_bu  # Start from x_bu for residual connections
        for block in self.blocks_step2:
            # Q, K from x_step2 (which is x_bu), V from x_v_injected
            x_step2 = block(x_step2, x_v=x_v_injected, mask=mask)
        
        # (4) Output: Extract CLS token and classify
        x_cls = self.norm(x_step2[:, 0])  # (B, embed_dim) - CLS token
        logits = self.classifier(x_cls)  # (B, num_classes)
        
        forward_return['logits'] = logits
        
        if return_WSI_feature:
            forward_return['WSI_feature'] = x_cls.squeeze(0)  # (embed_dim,)
        
        if return_WSI_attn:
            # Return selection scores for patches (excluding CLS token)
            patch_scores = sel_scores[:, 1:].squeeze(0)  # (N,) - scores for sampled patches
            
            # If we sampled patches, map scores back to original patch count
            if sampled_indices is not None and N_original > self.max_seq_len:
                # Create full score array for all original patches
                full_scores = torch.zeros(N_original, device=patch_scores.device, dtype=patch_scores.dtype)
                # Map sampled scores to their original positions
                full_scores[sampled_indices] = patch_scores
                forward_return['WSI_attn'] = full_scores
            else:
                forward_return['WSI_attn'] = patch_scores
        
        # Additional auxiliary info
        forward_return['aux'] = {
            'sel_score': sel_scores.squeeze(0) if B == 1 else sel_scores,  # (N+1,) or (B, N+1)
            'x_td': x_td.squeeze(0) if B == 1 else x_td,  # (N+1, embed_dim) or (B, N+1, embed_dim)
        }
        
        return forward_return
