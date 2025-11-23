"""
Micro_MIL - Micro MIL with Graph Attention Network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import Optional

try:
    from dgl.nn import GATConv
    import dgl
except ImportError:
    print("Warning: dgl not installed. Please install it: pip install dgl")
    GATConv = None
    dgl = None

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class DEC(nn.Module):
    """
    Deep Embedded Clustering (DEC) module
    """
    def __init__(self, cluster_number: int, embedding_dimension: int, alpha: float = 1.0, 
                 cluster_centers: Optional[torch.Tensor] = None):
        super(DEC, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        initial_cluster_centers = cluster_centers if cluster_centers is not None else torch.zeros(
            self.cluster_number, self.embedding_dimension, dtype=torch.float)
        nn.init.xavier_uniform_(initial_cluster_centers)
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        cluster_assignment = numerator / torch.sum(numerator, dim=1, keepdim=True)
        return cluster_assignment

class Micro_MIL(nn.Module):
    """
    Micro MIL with Graph Attention Network
    Uses clustering and graph neural networks for instance aggregation
    """
    def __init__(self, in_dim=1024, num_classes=2, dropout=0.5, act=nn.ReLU(), 
                 cluster_number=36, hidden_dim=128, layer=2, alpha=1.0, shuffle=False, **kwargs):
        super(Micro_MIL, self).__init__()
        
        if GATConv is None or dgl is None:
            raise ImportError("dgl is required. Install with: pip install dgl")
        
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.cluster_number = cluster_number
        self.hidden_dim = hidden_dim
        self.layer = layer
        self.alpha = alpha
        self.shuffle = shuffle
        
        # Feature transformation (assuming features are already extracted)
        self.image_feature = nn.Linear(in_dim, in_dim)
        
        # Clustering module
        self.dec = DEC(cluster_number=cluster_number, embedding_dimension=in_dim, alpha=alpha)
        
        # Attention for cluster selection
        self.attn = nn.Linear(in_dim, 1)
        
        # Graph Attention Network layers
        self.layers = nn.ModuleList([GATConv(in_dim, hidden_dim, num_heads=1)])
        for _ in range(layer - 1):
            self.layers.append(GATConv(hidden_dim, hidden_dim, num_heads=1))
        
        # Classifier
        self.classify = nn.Linear(hidden_dim, num_classes)
        
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.apply(initialize_weights)

    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False):
        """
        前向传播
        输入: x - (1, N, D) 或 (N, D) 形状的tensor
        输出: forward_return - 包含 'logits' 的字典
        """
        forward_return = {}
        
        # Handle input format - support both 2D and 3D inputs
        if len(x.shape) == 2:
            # Input is (N, D) - already 2D
            instances = x
        elif len(x.shape) == 3:
            # Input is (1, N, D) or (B, N, D)
            instances = x.squeeze(0)  # (N, D)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Ensure instances is 2D
        if len(instances.shape) == 1:
            instances = instances.unsqueeze(0)
        
        # Store original number of instances for attention return
        original_num_instances = instances.shape[0]
        
        # For Micro_MIL, we need at least 2 instances for graph construction
        # But we should preserve the original instances for attention return
        if instances.shape[0] == 1:
            # Duplicate for graph construction, but we'll handle attention separately
            instances_for_graph = torch.cat([instances, instances])
        else:
            instances_for_graph = instances
        
        # Feature transformation
        batch_instances = self.leaky_relu(self.image_feature(instances_for_graph))
        
        # Clustering
        cluster_assignments = self.dec(batch_instances)  # (N_graph, cluster_number)
        
        # Cluster-based attention
        cluster_assignments_list = []
        for i in range(self.cluster_number):
            cluster_attn = F.gumbel_softmax(
                self.attn(batch_instances * cluster_assignments[:, i:i+1]).squeeze(), 
                dim=0, hard=True
            )
            cluster_assignments_list.append(cluster_attn)
        
        gumbel_scores = torch.stack(cluster_assignments_list, dim=1)
        rep_features = torch.matmul(gumbel_scores.T, batch_instances)
        
        grid, dim = rep_features.size()
        
        # Build graph based on similarity
        similarity_matrix = F.cosine_similarity(
            rep_features.unsqueeze(1), rep_features.unsqueeze(0), dim=2
        )
        attention_scores = F.gumbel_softmax(
            similarity_matrix.view(-1, grid), hard=True
        ).view(-1, grid, grid)
        
        nonzero_indices = attention_scores.nonzero(as_tuple=True)
        x_nodes = nonzero_indices[1]
        
        if self.shuffle:
            x_nodes = x_nodes[torch.randperm(x_nodes.size(0))]
        
        # Create DGL graph
        device = batch_instances.device
        g = dgl.graph((x_nodes, nonzero_indices[2])).to(device)
        g = dgl.add_self_loop(g)
        
        h = rep_features.view(-1, dim)
        
        # Apply GAT layers
        for gat_layer in self.layers:
            h = gat_layer(g, h)
            h = self.dropout(h)
            h = self.leaky_relu(h)
        
        g.ndata['h'] = h
        
        # Aggregate graph nodes
        wsi_feature = dgl.mean_nodes(g, 'h').squeeze()
        
        # Classification
        logits = self.leaky_relu(self.classify(wsi_feature)).unsqueeze(0)  # (1, num_classes)
        
        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = wsi_feature.unsqueeze(0)
        if return_WSI_attn:
            # Return attention weights for each patch (instance)
            # cluster_assignments shape: (N_graph, cluster_number)
            # gumbel_scores shape: (N_graph, cluster_number) - represents importance of each patch to each cluster
            
            # Use gumbel_scores to compute patch importance
            # Sum across clusters gives total importance of each patch
            # This represents how much each patch contributes to the final representation
            patch_attention = gumbel_scores.sum(dim=1)  # (N_graph,)
            
            # If we duplicated instances, only return attention for original instances
            if original_num_instances == 1 and instances_for_graph.shape[0] > 1:
                # Only return attention for the first (original) instance
                patch_attention = patch_attention[:1]  # (1,)
            elif patch_attention.shape[0] != original_num_instances:
                # Ensure we return the correct number of attention weights
                patch_attention = patch_attention[:original_num_instances]
            
            # Normalize attention weights to sum to 1
            patch_attention = patch_attention / (patch_attention.sum() + 1e-8)
            
            # Ensure output is 1D tensor (remove batch dimension if present)
            if len(patch_attention.shape) > 1:
                patch_attention = patch_attention.squeeze()
            
            forward_return['WSI_attn'] = patch_attention  # (N,) or (original_num_instances,)
        
        return forward_return
