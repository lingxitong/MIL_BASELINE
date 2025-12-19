"""
SC_MIL - Sparse Context-aware Multiple Instance Learning
Reference: Sparse Context-aware Multiple Instance Learning for Predicting Cancer Survival Probability Distribution in Whole Slide Images
Paper: (需要补充论文信息)

Note: This model requires coordinates (coords) as input, similar to LONG_MIL.
The coords should be provided from h5 files.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange

# Try to import cuml (GPU-accelerated KMeans) for faster clustering
CUML_AVAILABLE = False
SKLEARN_AVAILABLE = False

try:
    from cuml import KMeans as cuKMeans
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

# Fallback to sklearn KMeans (CPU-based, slower but always available)
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Print warning only once at module import (suppress by default, user can enable if needed)
# Note: sklearn is sufficient for SC_MIL, cuml is optional for GPU acceleration
if not CUML_AVAILABLE and SKLEARN_AVAILABLE:
    # Suppress warning by default - sklearn works fine
    pass

if not SKLEARN_AVAILABLE:
    raise ImportError("sklearn not available. Please install: pip install scikit-learn")

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class SoftFilterLayer(nn.Module):
    """Soft Filter Layer for instance selection"""
    def __init__(self, dim, hidden_size=256, deep=1):
        super().__init__()
        layers = []
        for i in range(deep):
            layers.append(nn.Linear(dim, hidden_size))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        logits = self.layers(x)  # (B, N, 1)
        h = torch.mul(x, logits)  # Element-wise multiplication
        return h, logits

class SelfAttention(nn.Module):
    """Self-attention mechanism"""
    def __init__(self, dim, num_heads=8, head_size=None, dropout_prob=0.25):
        super().__init__()
        if head_size is None:
            head_size = dim // num_heads
        inner_dim = num_heads * head_size
        self.num_heads = num_heads
        self.head_size = head_size
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3)
        self.attn_dropout = nn.Dropout(dropout_prob)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout_prob)
        )
    
    def forward(self, x):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_size)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        out = torch.matmul(attention_probs, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=self.num_heads)
        out = self.to_out(out)
        return out

class ClusterLocalAttention(nn.Module):
    """Cluster-based Local Attention"""
    def __init__(self, hidden_size=384, n_cluster=None, cluster_size=None, feature_weight=0, dropout_rate=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_cluster = n_cluster
        self.cluster_size = cluster_size
        self.feature_weight = feature_weight
        
        self.atten = SelfAttention(dim=hidden_size, num_heads=8, head_size=hidden_size//8, dropout_prob=dropout_rate)
        
        self.num_heads = 8
        self.head_size = hidden_size // self.num_heads
        inner_dim = self.num_heads * self.head_size
        self.to_qkv = nn.Linear(hidden_size, inner_dim * 3)
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, hidden_size),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x, coords=None, cluster_label=None, return_patch_label=False):
        """
        Args:
            x: (B, L, C) - features
            coords: (B, L, 2) - coordinates
            cluster_label: optional pre-computed cluster labels
            return_patch_label: whether to return patch labels
        """
        B, L, C = x.shape
        
        if cluster_label is not None:
            labels = cluster_label.numpy()[0] if isinstance(cluster_label, torch.Tensor) else cluster_label
        else:
            if self.n_cluster is None and self.cluster_size is not None:
                n_cluster = L // self.cluster_size
                if n_cluster == 0:
                    n_cluster = 1
            elif self.n_cluster is not None:
                n_cluster = self.n_cluster
            else:
                raise ValueError("Either n_cluster or cluster_size must be specified")
            
            if coords is None:
                raise ValueError("coords required for clustering")
            
            # K-means clustering
            coords_np = coords.squeeze().cpu().numpy()
            coords_norm = (coords_np - np.mean(coords_np, axis=0)) / (np.std(coords_np, axis=0) + 1e-8)
            
            if self.feature_weight != 0:
                feats_np = x.squeeze().cpu().numpy()
                feats_norm = (feats_np - np.mean(feats_np, axis=0)) / (np.std(feats_np, axis=0) + 1e-8)
                feats_norm = feats_norm * self.feature_weight
                coords_norm = coords_norm * (1 - self.feature_weight)
                clustering_input = np.concatenate([feats_norm, coords_norm], axis=1)
            else:
                clustering_input = coords_norm
            
            if CUML_AVAILABLE:
                # Use GPU-accelerated cuml KMeans
                kmeans = cuKMeans(n_clusters=n_cluster, init='k-means++', tol=1e-4, max_iter=5, random_state=1)
                labels = kmeans.fit_predict(clustering_input)
                # cuml returns cuDF Series, need to convert to numpy
                labels = labels.get() if hasattr(labels, 'get') else labels
                if hasattr(labels, 'values'):
                    labels = labels.values
                labels = np.asarray(labels)
            elif SKLEARN_AVAILABLE:
                # Use CPU-based sklearn KMeans (slower but functional)
                kmeans = KMeans(n_clusters=n_cluster, init='k-means++', tol=1e-4, max_iter=5, random_state=1, n_init=10)
                labels = kmeans.fit_predict(clustering_input)
                labels = np.asarray(labels)
            else:
                # Fallback: random assignment (should not happen if sklearn is installed)
                np.random.seed(1)
                labels = np.random.randint(0, n_cluster, size=L)
        
        # Sort by cluster labels
        index = np.argsort(labels, kind='stable').tolist()
        index_ori = np.argsort(index, kind='stable').tolist()
        x = x[:, index]
        
        # Compute window sizes
        window_sizes = np.bincount(labels).tolist()
        
        # Prevent large cluster size
        window_sizes_new = []
        for size in window_sizes:
            if size >= self.cluster_size * 2 if self.cluster_size else False:
                num_splits = size // self.cluster_size
                quotient = size // num_splits
                remainder = size % num_splits
                result = [quotient + 1 if i < remainder else quotient for i in range(num_splits)]
                window_sizes_new.extend(result)
            else:
                window_sizes_new.append(size)
        window_sizes = window_sizes_new
        
        # Apply attention within each cluster
        qs, ks, vs = self.to_qkv(x).chunk(3, dim=-1)
        qs, ks, vs = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (qs, ks, vs))
        h = torch.zeros_like(qs)
        
        now = 0
        for i in range(len(window_sizes)):
            q = qs[:, :, now:now+window_sizes[i]]
            k = ks[:, :, now:now+window_sizes[i]]
            v = vs[:, :, now:now+window_sizes[i]]
            
            attention_scores = torch.matmul(q, k.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.head_size)
            attention_probs = F.softmax(attention_scores, dim=-1)
            attention_probs = self.attn_dropout(attention_probs)
            out = torch.matmul(attention_probs, v)
            h[:, :, now:now+window_sizes[i]] = out
            now += window_sizes[i]
        
        h = rearrange(h, 'b h n d -> b n (h d)', h=self.num_heads)
        h = self.to_out(h) + x  # Residual connection
        
        if return_patch_label:
            return h, labels
        else:
            return h, None

class AMIL_layer(nn.Module):
    """Attention-based MIL aggregation layer"""
    def __init__(self, L=1024, D=256, dropout=0.1):
        super().__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(L, D),
            nn.Tanh(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )
        self.attention_b = nn.Sequential(
            nn.Linear(L, D),
            nn.Sigmoid(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )
        self.attention_c = nn.Linear(D, 1)
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        a = self.attention_a(x)  # (B, N, D)
        b = self.attention_b(x)  # (B, N, D)
        A_without_softmax = a.mul(b)  # Element-wise multiplication
        A_without_softmax = self.attention_c(A_without_softmax).squeeze(dim=2)  # (B, N)
        A = F.softmax(A_without_softmax, dim=1).unsqueeze(dim=1)  # (B, 1, N)
        h = torch.bmm(A, x).squeeze(dim=1)  # (B, L)
        return h, A.squeeze(dim=1), A_without_softmax

class SC_MIL(nn.Module):
    """
    Sparse Context-aware Multiple Instance Learning
    
    Uses cluster-based local attention and context-aware aggregation.
    Requires coordinates (coords) as input for clustering.
    
    Reference: Sparse Context-aware Multiple Instance Learning for Predicting Cancer Survival Probability Distribution in Whole Slide Images
    """
    def __init__(self, in_dim=1024, num_classes=2, hidden_size=None, deep=1, 
                 n_cluster=None, cluster_size=None, feature_weight=0,
                 dropout=0.25, with_softfilter=False, use_filter_branch=False, 
                 with_cssa=True, act=nn.ReLU()):
        super(SC_MIL, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.deep = deep
        self.n_cluster = n_cluster
        self.cluster_size = cluster_size
        self.feature_weight = feature_weight
        self.with_softfilter = with_softfilter
        self.with_cssa = with_cssa
        self.use_filter_branch = use_filter_branch or with_softfilter
        
        if hidden_size is None:
            hidden_size = in_dim
        
        if self.with_softfilter:
            self.softfilter = SoftFilterLayer(dim=in_dim, hidden_size=256, deep=1)
        
        self.fc1 = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            act,
            nn.Dropout(dropout)
        )
        
        if with_cssa:
            layers = []
            for i in range(deep):
                layers.append(ClusterLocalAttention(
                    hidden_size=hidden_size,
                    n_cluster=n_cluster,
                    cluster_size=cluster_size,
                    feature_weight=feature_weight,
                    dropout_rate=dropout
                ))
            self.attens = nn.ModuleList(layers)
        
        self.amil = AMIL_layer(L=hidden_size, D=256, dropout=dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        self.apply(initialize_weights)
    
    def forward(self, x, coords=None, return_WSI_attn=False, return_WSI_feature=False):
        """
        Args:
            x: (N, D) or (1, N, D) - patch features
            coords: (N, 2) or (1, N, 2) - patch coordinates (required for clustering)
            return_WSI_attn: bool
            return_WSI_feature: bool
        """
        forward_return = {}
        
        # Handle input format
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # (N, D) -> (1, N, D)
        
        if coords is not None and len(coords.shape) == 2:
            coords = coords.unsqueeze(0)  # (N, 2) -> (1, N, 2)
        
        # Soft filter (optional)
        if self.with_softfilter:
            x, logits_filter = self.softfilter(x)
        
        # Filter branch (optional)
        if self.use_filter_branch and self.with_softfilter:
            idx = torch.where(logits_filter.squeeze() > 0.5)[0]
            if len(idx) == 0:
                idx = torch.arange(x.shape[1], device=x.device)
            h = x[:, idx]
            if coords is not None:
                coords_filtered = coords[:, idx]
            else:
                coords_filtered = None
            idx2 = torch.where(logits_filter.squeeze() <= 0.5)[0]
            if len(idx2) > 0:
                h_app = x[:, idx2]
            else:
                h_app = None
        else:
            h = self.fc1(x)
            coords_filtered = coords
            h_app = None
        
        # Cluster-based local attention
        if self.with_cssa:
            if h.shape[1] > 1:
                for atten in self.attens:
                    h, _ = atten(h, coords_filtered, cluster_label=None, return_patch_label=False)
        
        # Concatenate filtered and unfiltered features if using filter branch
        if self.use_filter_branch and h_app is not None:
            h = torch.cat([h, h_app], dim=1)
        
        # AMIL aggregation
        h, att, _ = self.amil(h)  # h: (1, hidden_size), att: (1, N)
        
        # Classification
        logits = self.classifier(h)  # (1, num_classes)
        
        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = h.squeeze(0)
        if return_WSI_attn:
            forward_return['WSI_attn'] = att.squeeze(0)
        
        return forward_return

