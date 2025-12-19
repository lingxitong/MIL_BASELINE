"""
MICO_MIL - Multiple Instance Learning with Context-Aware Clustering
Reference: https://github.com/MiCo-main
Paper: Multiple Instance Learning with Context-Aware Clustering for Whole Slide Image Analysis

Key idea: Context-aware clustering with multi-scale enhancement
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class GatedAttention(nn.Module):
    """Gated attention mechanism"""
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=1, drop=0.25):
        super().__init__()
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(drop) if drop != 0 else nn.Identity()
        )
        
        self.attention_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(drop) if drop != 0 else nn.Identity()
        )
        
        self.attention_scorer = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, features):
        transformed = self.feature_transform(features)
        gate = self.attention_gate(features)
        attention_weights = self.attention_scorer(transformed * gate)
        return attention_weights, features

class Mlp(nn.Module):
    """MLP module"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop) if drop != 0 else nn.Identity()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ClusterReducer(Mlp):
    """Cluster reducer for multi-scale enhancement"""
    def forward(self, x):
        """
        Args:
            x: (num_clusters, embedding_dim)
        """
        return super().forward(x.transpose(0, 1)).transpose(0, 1)

class MICO_MIL(nn.Module):
    """
    Multiple Instance Learning with Context-Aware Clustering
    
    Uses context-aware clustering with multi-scale enhancement and gated attention.
    Reference: https://github.com/MiCo-main
    """
    def __init__(self, in_dim=768, embedding_dim=512, num_clusters=64, num_classes=2,
                 num_enhancers=3, drop=0.25, hard=False, similarity_method='l2',
                 cluster_init_path=None):
        super(MICO_MIL, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.hard_assignment = hard
        self.similarity_method = similarity_method
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        
        # Initialize cluster centers
        if cluster_init_path:
            initial_centers = torch.load(cluster_init_path)
            self.cluster_centers = nn.Parameter(torch.from_numpy(initial_centers), requires_grad=True)
        else:
            self.cluster_centers = nn.Parameter(torch.randn(num_clusters, in_dim), requires_grad=True)
        
        # Projectors
        self.patch_feature_projector = nn.Sequential(
            nn.Linear(in_dim, embedding_dim),
            nn.LeakyReLU(inplace=True)
        )
        self.cluster_center_projector = nn.Sequential(
            nn.Linear(in_dim, embedding_dim),
            nn.LeakyReLU(inplace=True)
        )
        
        # Multi-scale context enhancement
        self.dynamic_num_clusters = [num_clusters // (2 ** i) for i in range(num_enhancers + 1)]
        
        self.context_enhancers = nn.ModuleList([
            Mlp(embedding_dim, embedding_dim, embedding_dim, nn.ReLU, drop)
            for _ in range(num_enhancers)
        ])
        self.cluster_reducers = nn.ModuleList([
            ClusterReducer(
                in_features=self.dynamic_num_clusters[i],
                hidden_features=self.dynamic_num_clusters[i],
                out_features=self.dynamic_num_clusters[i+1],
                act_layer=nn.ReLU,
                drop=drop
            )
            for i in range(num_enhancers)
        ])
        self.enhancer_norm_layers = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(num_enhancers)
        ])
        
        # Feature processing and attention
        self.feature_processor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(drop) if drop != 0 else nn.Identity()
        )
        self.attention_network = GatedAttention(
            input_dim=embedding_dim,
            hidden_dim=embedding_dim,
            num_classes=1,
            drop=drop
        )
        self.aggregation_norm_layer = nn.LayerNorm(embedding_dim)
        
        # Classifier
        self.final_projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(drop) if drop > 0 else nn.Identity()
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        # Similarity scaling
        self.similarity_scale = nn.Parameter(torch.ones(1), requires_grad=True)
        self.similarity_bias = nn.Parameter(torch.zeros(1), requires_grad=True)
        
        self.apply(initialize_weights)
    
    def _straight_through_softmax(self, logits, hard_assignment=True, dim=-1):
        """Straight-through softmax for cluster assignment"""
        y_soft = F.softmax(logits / self.similarity_scale, dim=1)
        if hard_assignment:
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        return ret
    
    def _get_contextual_features(self, patch_embeddings, cluster_embeddings):
        """
        Compute contextual features based on similarity between patches and clusters
        """
        if self.similarity_method == 'l2':
            similarity_scores = -torch.cdist(patch_embeddings, cluster_embeddings)
        else:  # dot product
            similarity_scores = patch_embeddings @ cluster_embeddings.transpose(-2, -1)
        
        assignment_weights = self._straight_through_softmax(similarity_scores, self.hard_assignment, dim=1)
        contextual_features = torch.matmul(assignment_weights, cluster_embeddings)
        return contextual_features
    
    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False):
        """
        Args:
            x: (N, D) or (1, N, D) - patch features
            return_WSI_attn: bool
            return_WSI_feature: bool
        """
        forward_return = {}
        
        # Handle input format
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # (N, D) -> (1, N, D)
        
        patch_features = x.squeeze(0)  # (N, D)
        
        # Project to embedding space
        patch_embeddings = self.patch_feature_projector(patch_features)  # (N, embedding_dim)
        cluster_embeddings = self.cluster_center_projector(self.cluster_centers)  # (num_clusters, embedding_dim)
        
        # Multi-scale context enhancement
        for i, enhancer_mlp in enumerate(self.context_enhancers):
            # Compute contextual features
            contextual_features = self._get_contextual_features(patch_embeddings, cluster_embeddings)
            
            # Fusion: add context to patch embeddings (residual connection)
            patch_embeddings = patch_embeddings + contextual_features
            
            # Normalize
            patch_embeddings = self.enhancer_norm_layers[i](patch_embeddings)
            
            # Enhancement: further processing via MLP (residual connection)
            patch_embeddings = patch_embeddings + enhancer_mlp(patch_embeddings)
            
            # Reduce: decrease number of cluster centers for next stage
            cluster_embeddings = self.cluster_reducers[i](cluster_embeddings)
        
        # Concatenate patch embeddings and remaining cluster embeddings
        enhanced_embeddings = torch.cat([patch_embeddings, cluster_embeddings], dim=0)  # (N + num_clusters_final, embedding_dim)
        
        # Feature processing
        processed_embeddings = self.feature_processor(enhanced_embeddings)
        processed_embeddings = self.aggregation_norm_layer(processed_embeddings)
        
        # Attention pooling
        attention_scores, _ = self.attention_network(processed_embeddings)  # (N + num_clusters_final, 1)
        attention_weights = F.softmax(attention_scores.transpose(0, 1), dim=1)  # (1, N + num_clusters_final)
        
        # Weighted aggregation
        slide_level_representation = torch.mm(attention_weights, processed_embeddings)  # (1, embedding_dim)
        
        # Classification
        final_features = self.final_projector(slide_level_representation).squeeze(0)  # (embedding_dim,)
        logits = self.classifier(final_features).unsqueeze(0)  # (1, num_classes)
        
        forward_return['logits'] = logits
        
        if return_WSI_feature:
            forward_return['WSI_feature'] = final_features
        
        if return_WSI_attn:
            # Return attention weights for patches only (exclude cluster embeddings)
            patch_attn = attention_weights.squeeze(0)[:patch_features.shape[0]]  # (N,)
            forward_return['WSI_attn'] = patch_attn
        
        return forward_return


