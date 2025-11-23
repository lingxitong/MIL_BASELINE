"""
PA_MIL - Policy-Driven Adaptive Multiple Instance Learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class Attention(nn.Module):
    """Multi-head attention mechanism with memory optimization"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        """
        Optimized attention for long sequences
        """
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # For long sequences, use chunked attention
        if n > 1000:
            chunk_size = 512
            out_chunks = []
            attn_chunks = []

            for i in range(0, n, chunk_size):
                end_idx = min(i + chunk_size, n)
                q_chunk = q[:, :, i:end_idx, :]  # (b, heads, chunk_size, dim_head)

                dots = torch.matmul(q_chunk, k.transpose(-1, -2)) * self.scale  # (b, heads, chunk_size, n)
                attn = self.attend(dots)
                attn = self.dropout(attn)

                out_chunk = torch.matmul(attn, v)  # (b, heads, chunk_size, dim_head)
                out_chunks.append(out_chunk)
                attn_chunks.append(attn)

            out = torch.cat(out_chunks, dim=2)  # (b, heads, n, dim_head)
            attn = torch.cat(attn_chunks, dim=2)  # (b, heads, n, n)
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = self.attend(dots)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn

class SimpleAttenLayer(nn.Module):
    """Simple attention layer with residual connection - memory efficient"""
    def __init__(self, dim, dropout=0.1):
        super(SimpleAttenLayer, self).__init__()
        self.dim = dim
        self.dropout = dropout

        # Simple single-head attention
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        """
        Memory-efficient attention: avoid storing full attention matrix
        Optimized for long sequences using chunked attention with CLS token attention tracking
        """
        b, n, d = x.shape

        # Project to Q, K, V
        q = self.query_proj(x)  # (b, n, d)
        k = self.key_proj(x)    # (b, n, d)
        v = self.value_proj(x)  # (b, n, d)

        # For long sequences, compute attention in chunks to avoid O(n²) memory
        if n > 2000:
            chunk_size = 256
            attn_out_chunks = []
            # Store CLS token attention weights (first token) for visualization
            cls_attn_weights = None

            for i in range(0, n, chunk_size):
                end_idx = min(i + chunk_size, n)
                q_chunk = q[:, i:end_idx, :]  # (b, chunk_size, d)

                # Compute attention scores for this chunk
                # For very long sequences, also chunk the keys to save memory
                if n > 5000:
                    # Double chunking: chunk both queries and keys
                    k_chunk_size = 512
                    attn_out_chunk_parts = []
                    attn_weights_chunk_all = []
                    
                    for j in range(0, n, k_chunk_size):
                        k_end = min(j + k_chunk_size, n)
                        k_chunk = k[:, j:k_end, :]  # (b, k_chunk_size, d)
                        v_chunk = v[:, j:k_end, :]  # (b, k_chunk_size, d)
                        
                        # Compute attention scores for this key chunk
                        scores_chunk = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) / (d ** 0.5)  # (b, chunk_size, k_chunk_size)
                        attn_weights_chunk = F.softmax(scores_chunk, dim=-1)
                        attn_weights_chunk = self.dropout_layer(attn_weights_chunk)
                        
                        # Store attention weights for re-normalization
                        attn_weights_chunk_all.append(attn_weights_chunk)
                        
                        # Apply attention
                        attn_out_chunk_part = torch.matmul(attn_weights_chunk, v_chunk)  # (b, chunk_size, d)
                        attn_out_chunk_parts.append(attn_out_chunk_part)
                    
                    # Concatenate attention weights and re-normalize across all key chunks
                    attn_weights_chunk_full = torch.cat(attn_weights_chunk_all, dim=-1)  # (b, chunk_size, n)
                    attn_weights_chunk_full = F.softmax(attn_weights_chunk_full, dim=-1)  # Re-normalize
                    
                    # Re-compute attention output with normalized weights
                    attn_out_chunk = torch.matmul(attn_weights_chunk_full, v)  # (b, chunk_size, d)
                    
                    # Store CLS token attention weights (first token)
                    if i == 0:
                        cls_attn_weights = attn_weights_chunk_full[:, 0:1, :]  # (b, 1, n)
                else:
                    # Single chunking: chunk queries only
                    scores_chunk = torch.matmul(q_chunk, k.transpose(-2, -1)) / (d ** 0.5)  # (b, chunk_size, n)
                    attn_weights_chunk = F.softmax(scores_chunk, dim=-1)
                    attn_weights_chunk = self.dropout_layer(attn_weights_chunk)

                    # Apply attention
                    attn_out_chunk = torch.matmul(attn_weights_chunk, v)  # (b, chunk_size, d)
                    
                    # Store CLS token attention weights (first token)
                    if i == 0:
                        cls_attn_weights = attn_weights_chunk[:, 0:1, :]  # (b, 1, n)

                attn_out_chunks.append(attn_out_chunk)

            attn_out = torch.cat(attn_out_chunks, dim=1)  # (b, n, d)
            # Use CLS token attention weights for visualization
            attn_weights = cls_attn_weights if cls_attn_weights is not None else None
        else:
            # Standard attention for shorter sequences
            scores = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)  # (b, n, n)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout_layer(attn_weights)
            attn_out = torch.matmul(attn_weights, v)  # (b, n, d)

        attn_out = self.out_proj(attn_out)

        # Residual connection
        return x + attn_out, attn_weights

class PA_MIL(nn.Module):
    """
    Policy-Driven Adaptive Multiple Instance Learning
    Simplified version using traditional attention for memory efficiency
    """
    def __init__(self, in_dim=1024, num_classes=2, dropout=0.1, act=nn.ReLU(),
                 embed_dim=128, num_layers=1, **kwargs):
        super(PA_MIL, self).__init__()

        self.in_dim = in_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Feature projection (smaller embed_dim)
        self.feature_proj = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            act,
            nn.Dropout(dropout) if dropout else nn.Identity()
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Simple attention layers (single-head attention to save memory)
        self.atten_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.atten_layers.append(SimpleAttenLayer(dim=embed_dim, dropout=dropout))

        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Classifier
        self.classifier = nn.Linear(embed_dim, num_classes)

        self.apply(initialize_weights)

    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False):
        """
        前向传播
        输入: x - (1, N, D) 形状的tensor
        输出: forward_return - 包含 'logits' 的字典
        """
        forward_return = {}
        
        # Handle input format
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # (N, D) -> (1, N, D)
        
        batch_size = x.shape[0]
        
        # Feature projection
        x = self.feature_proj(x)  # (1, N, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (1, N+1, embed_dim)
        
        # Apply attention layers
        attn_weights_list = []
        for atten_layer in self.atten_layers:
            x, attn_weights = atten_layer(x)
            attn_weights_list.append(attn_weights)
        
        # Layer norm
        x = self.norm(x)
        
        # Extract CLS token
        cls_feat = x[:, 0, :]  # (1, embed_dim)
        
        # Classification
        logits = self.classifier(cls_feat)  # (1, num_classes)
        
        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = cls_feat
        if return_WSI_attn:
            # Use attention from last layer
            if len(attn_weights_list) > 0:
                last_attn = attn_weights_list[-1]  # (batch, 1, n) or (batch, n, n)
                
                # Handle different attention weight shapes
                if last_attn is not None:
                    if len(last_attn.shape) == 3:
                        # Chunked attention: CLS token attention to all patches
                        if last_attn.shape[1] == 1:
                            # Shape: (batch, 1, n) - CLS token attention to all patches
                            # Skip CLS token itself (index 0), extract attention to patches (index 1:)
                            if last_attn.shape[2] == x.shape[1]:
                                # Full sequence including CLS token
                                cls_attn = last_attn[:, 0, 1:].squeeze(0)  # (N,)
                            else:
                                # Only patches (no CLS token in attention weights)
                                cls_attn = last_attn[:, 0, :].squeeze(0)  # (N,)
                        else:
                            # Full attention matrix: (batch, n, n)
                            # Extract CLS token row (first row), skip CLS token itself
                            if last_attn.shape[2] == x.shape[1]:
                                # Full sequence including CLS token
                                cls_attn = last_attn[:, 0, 1:].squeeze(0)  # (N,)
                            else:
                                # Only patches
                                cls_attn = last_attn[:, 0, :].squeeze(0)  # (N,)
                    elif len(last_attn.shape) == 2:
                        # Already flattened: (batch, n) or (n,)
                        if last_attn.shape[0] == batch_size:
                            # (batch, n) - extract first sample and skip CLS token
                            if last_attn.shape[1] == x.shape[1]:
                                cls_attn = last_attn[0, 1:]  # (N,)
                            else:
                                cls_attn = last_attn[0, :]  # (N,)
                        else:
                            # (n,) - already single sample, skip CLS token if present
                            if last_attn.shape[0] == x.shape[1]:
                                cls_attn = last_attn[1:]  # (N,)
                            else:
                                cls_attn = last_attn  # (N,)
                    else:
                        # Fallback: uniform attention
                        num_patches = x.shape[1] - 1
                        cls_attn = torch.ones(num_patches, device=x.device) / num_patches
                    
                    # Ensure correct shape and normalize
                    if cls_attn.shape[0] != (x.shape[1] - 1):
                        # Shape mismatch, use uniform attention
                        num_patches = x.shape[1] - 1
                        cls_attn = torch.ones(num_patches, device=x.device) / num_patches
                    else:
                        # Normalize to ensure sum is 1
                        cls_attn = cls_attn / (cls_attn.sum() + 1e-8)
                    
                    forward_return['WSI_attn'] = cls_attn
                else:
                    # Fallback: uniform attention
                    num_patches = x.shape[1] - 1
                    forward_return['WSI_attn'] = torch.ones(num_patches, device=x.device) / num_patches
            else:
                # Fallback: uniform attention
                num_patches = x.shape[1] - 1
                forward_return['WSI_attn'] = torch.ones(num_patches, device=x.device) / num_patches
        
        return forward_return
