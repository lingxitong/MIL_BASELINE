"""
PGCN_MIL - Patch Graph Convolutional Network Multiple Instance Learning
Reference: https://github.com/mahmoodlab/Patch-GCN
Paper: Context-Aware Survival Prediction using Patch-based Graph Convolutional Networks (MICCAI 2021)

Note: Requires torch-geometric or dgl for graph operations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, global_mean_pool
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    try:
        import dgl
        from dgl.nn import GraphConv
        DGL_AVAILABLE = True
        TORCH_GEOMETRIC_AVAILABLE = False
    except ImportError:
        TORCH_GEOMETRIC_AVAILABLE = False
        DGL_AVAILABLE = False
        print("Warning: Neither torch-geometric nor dgl is available. PGCN_MIL requires one of them.")

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

def build_knn_graph(features, k=5):
    """
    Build k-NN graph from features
    Args:
        features: (N, D) tensor
        k: number of neighbors
    Returns:
        edge_index: (2, E) tensor for torch-geometric or edge list for dgl
    """
    N = features.shape[0]
    if N <= k:
        k = max(1, N - 1)
    
    # Compute pairwise distances
    distances = torch.cdist(features, features)  # (N, N)
    
    # Get k nearest neighbors
    _, indices = torch.topk(distances, k=k+1, dim=1, largest=False)  # (N, k+1)
    indices = indices[:, 1:]  # Remove self-connections
    
    # Create edge list
    src = torch.arange(N).repeat_interleave(k).to(features.device)
    dst = indices.flatten().to(features.device)
    
    # Bidirectional edges
    edge_index = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src])
    ])
    
    return edge_index

class PGCN_MIL(nn.Module):
    """
    Patch Graph Convolutional Network Multiple Instance Learning
    
    Builds a graph from patch features (k-NN) and uses GCN to extract bag-level representation.
    
    Reference: https://github.com/mahmoodlab/Patch-GCN
    """
    def __init__(self, in_dim=1024, num_classes=2, dropout=0.1, act=nn.ReLU(), 
                 hidden_dim=256, n_layers=2, k=5, use_dgl=False):
        super(PGCN_MIL, self).__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE and not DGL_AVAILABLE:
            raise ImportError("PGCN_MIL requires either torch-geometric or dgl. Install with: pip install torch-geometric or pip install dgl")
        
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.k = k
        self.use_dgl = use_dgl and DGL_AVAILABLE
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act,
            nn.Dropout(dropout)
        )
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        if TORCH_GEOMETRIC_AVAILABLE and not self.use_dgl:
            for i in range(n_layers):
                self.gcn_layers.append(
                    GCNConv(hidden_dim if i == 0 else hidden_dim, hidden_dim)
                )
        elif DGL_AVAILABLE and self.use_dgl:
            for i in range(n_layers):
                self.gcn_layers.append(
                    GraphConv(hidden_dim if i == 0 else hidden_dim, hidden_dim, activation=act)
                )
        
        self.dropout = nn.Dropout(dropout)
        self.act = act
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        self.apply(initialize_weights)
    
    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False):
        forward_return = {}
        
        # Handle input format
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # (N, D) -> (1, N, D)
        
        x = x.squeeze(0)  # (N, D)
        N = x.shape[0]
        
        # Feature projection
        x_proj = self.feature_proj(x)  # (N, hidden_dim)
        x = x_proj  # Keep reference for attention computation
        
        # Build graph (use projected features for graph construction)
        edge_index = build_knn_graph(x_proj, k=self.k)  # (2, E)
        
        # Apply GCN layers
        if TORCH_GEOMETRIC_AVAILABLE and not self.use_dgl:
            for i, gcn_layer in enumerate(self.gcn_layers):
                x = gcn_layer(x, edge_index)
                if i < len(self.gcn_layers) - 1:
                    x = self.act(x)
                    x = self.dropout(x)
            
            # Global pooling
            batch = torch.zeros(N, dtype=torch.long, device=x.device)
            bag_feature = global_mean_pool(x, batch)  # (1, hidden_dim)
        elif DGL_AVAILABLE and self.use_dgl:
            import dgl
            # Create DGL graph
            g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=N).to(x.device)
            g = dgl.add_self_loop(g)
            
            for i, gcn_layer in enumerate(self.gcn_layers):
                x = gcn_layer(g, x)
                if i < len(self.gcn_layers) - 1:
                    x = self.dropout(x)
            
            # Set node features for pooling
            g.ndata['h'] = x
            # Global pooling
            bag_feature = dgl.mean_nodes(g, 'h').unsqueeze(0)
        
        # Keep batch dimension for training compatibility
        # bag_feature shape: (1, hidden_dim)
        
        # Classification
        logits = self.classifier(bag_feature)  # (1, num_classes)
        
        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = bag_feature.unsqueeze(0)
        if return_WSI_attn:
            # Compute node importance scores for visualization
            # Use GCN output features (x) which contain graph structure information
            # Method: Use dot product between node features and classifier weights
            # This measures how much each node contributes to the final prediction
            
            with torch.no_grad():
                # Get classifier weights for the predicted class
                if logits.shape[1] == 2:
                    # Binary classification: use the positive class weights
                    # This measures how much each node contributes to positive prediction
                    class_weights = self.classifier.weight[1]  # (hidden_dim,)
                else:
                    # Multi-class: use the predicted class weights
                    pred_class = logits.argmax(dim=1).item()
                    class_weights = self.classifier.weight[pred_class]  # (hidden_dim,)
                
                # Compute importance as dot product with classifier weights
                # Nodes with features aligned with classifier weights are more important
                # This directly reflects contribution to the prediction
                node_importance = (x * class_weights.unsqueeze(0)).sum(dim=1)  # (N,)
            
            # Apply softmax-like normalization to get attention scores
            # Use temperature scaling to increase discrimination
            temperature = 2.0  # Higher temperature = more uniform, lower = more peaked
            node_importance_scaled = node_importance / temperature
            
            # Softmax normalization
            node_importance_exp = torch.exp(node_importance_scaled - node_importance_scaled.max())
            node_importance = node_importance_exp / (node_importance_exp.sum() + 1e-8)
            
            forward_return['WSI_attn'] = node_importance  # (N,)
        
        return forward_return

