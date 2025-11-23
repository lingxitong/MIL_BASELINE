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

class ContextAwareAttention(nn.Module):
    """
    Context-aware attention mechanism that considers relationships between instances
    """
    def __init__(self, L, D):
        super(ContextAwareAttention, self).__init__()
        self.L = L
        self.D = D
        
        # Query, Key, Value projections
        self.query = nn.Linear(L, D)
        self.key = nn.Linear(L, D)
        self.value = nn.Linear(L, D)
        
        # Attention weights
        self.attention_weights = nn.Sequential(
            nn.Linear(D, D),
            nn.Tanh(),
            nn.Linear(D, 1)
        )
        
    def forward(self, x):
        """
        x: [N, L] where N is number of instances
        Memory-efficient version with chunked attention for long sequences
        """
        N, L = x.shape
        Q = self.query(x)  # [N, D]
        K = self.key(x)    # [N, D]
        V = self.value(x)  # [N, D]
        
        # For long sequences, use chunked attention to avoid O(N^2) memory
        if N > 2000:
            chunk_size = 512
            context_features_chunks = []
            
            for i in range(0, N, chunk_size):
                end_idx = min(i + chunk_size, N)
                Q_chunk = Q[i:end_idx, :]  # [chunk_size, D]
                
                # Compute attention scores for this chunk
                # For very long sequences, also chunk the keys
                if N > 5000:
                    k_chunk_size = 512
                    context_chunk_parts = []
                    attn_scores_chunks = []
                    
                    for j in range(0, N, k_chunk_size):
                        k_end = min(j + k_chunk_size, N)
                        K_chunk = K[j:k_end, :]  # [k_chunk_size, D]
                        V_chunk = V[j:k_end, :]  # [k_chunk_size, D]
                        
                        # Compute attention scores for this key chunk
                        scores_chunk = torch.matmul(Q_chunk, K_chunk.transpose(-2, -1)) / (self.D ** 0.5)  # [chunk_size, k_chunk_size]
                        attn_scores_chunks.append(scores_chunk)
                        
                        # Apply attention
                        context_chunk_part = torch.matmul(F.softmax(scores_chunk, dim=-1), V_chunk)  # [chunk_size, D]
                        context_chunk_parts.append(context_chunk_part)
                    
                    # Concatenate attention scores and re-normalize
                    all_scores = torch.cat(attn_scores_chunks, dim=-1)  # [chunk_size, N]
                    attn_weights_full = F.softmax(all_scores, dim=-1)  # [chunk_size, N]
                    
                    # Re-compute context features with normalized weights
                    context_chunk = torch.matmul(attn_weights_full, V)  # [chunk_size, D]
                else:
                    # Single chunking: chunk queries only
                    scores_chunk = torch.matmul(Q_chunk, K.transpose(-2, -1)) / (self.D ** 0.5)  # [chunk_size, N]
                    attn_weights_chunk = F.softmax(scores_chunk, dim=-1)
                    context_chunk = torch.matmul(attn_weights_chunk, V)  # [chunk_size, D]
                
                context_features_chunks.append(context_chunk)
            
            context_features = torch.cat(context_features_chunks, dim=0)  # [N, D]
        else:
            # Standard attention for shorter sequences
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.D ** 0.5)  # [N, N]
            context_weights = F.softmax(scores, dim=-1)  # [N, N]
            context_features = torch.matmul(context_weights, V)  # [N, D]
        
        # Compute instance attention weights
        A = self.attention_weights(context_features)  # [N, 1]
        A = torch.transpose(A, -1, -2)  # [1, N]
        A = F.softmax(A, dim=-1)
        
        return A, context_features

class CA_MIL(nn.Module):
    """
    Context-Aware Multiple Instance Learning for WSI Classification (ICLR 2024)
    Key idea: Incorporates spatial and contextual relationships between patches
    using self-attention mechanisms.
    """
    def __init__(self, L=512, D=128, num_classes=2, dropout=0, act=nn.ReLU(), in_dim=512, rrt=None):
        super(CA_MIL, self).__init__()
        self.rrt = rrt
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.L = L
        self.D = D
        
        # Feature extraction layers
        self.feature = [nn.Linear(in_dim, self.L)]
        self.feature += [act]
        
        if dropout:
            self.feature += [nn.Dropout(dropout)]
            
        if self.rrt is not None:
            self.feature += [self.rrt]
        
        self.feature = nn.Sequential(*self.feature)
        
        # Context-aware attention mechanism
        self.context_attention = ContextAwareAttention(self.L, self.D)
        
        # Feature aggregation
        self.feature_aggregation = nn.Sequential(
            nn.Linear(self.D, self.L),
            act
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.L, self.num_classes)
        )
        
        self.apply(initialize_weights)
    
    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False):
        forward_return = {}
        
        # Handle input format - support both 2D and 3D inputs
        if len(x.shape) == 2:
            # Input is (N, D) - already 2D
            x = x.unsqueeze(0)  # (1, N, D)
        elif len(x.shape) == 3:
            # Input is (1, N, D) or (B, N, D)
            pass  # Already 3D
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Extract features - feature layer expects (B, N, D) or (N, D)
        # Reshape to (B*N, D) for feature extraction, then reshape back
        batch_size, num_instances, feat_dim = x.shape
        x_flat = x.view(-1, feat_dim)  # (B*N, D)
        feature_flat = self.feature(x_flat)  # (B*N, L)
        feature = feature_flat.view(batch_size, num_instances, -1)  # (B, N, L)
        feature = feature.squeeze(0)  # [N, L]
        
        # Apply context-aware attention
        A, context_features = self.context_attention(feature)  # A: [1, N], context_features: [N, D]
        
        # Aggregate context features
        aggregated_context = torch.mm(A, context_features)  # [1, D]
        
        # Transform back to feature space
        bag_feature = self.feature_aggregation(aggregated_context)  # [1, L]
        
        # Classification
        logits = self.classifier(bag_feature)  # [1, num_classes]
        
        forward_return['logits'] = logits
        
        if return_WSI_feature:
            forward_return['WSI_feature'] = bag_feature
        
        if return_WSI_attn:
            forward_return['WSI_attn'] = A.transpose(-1, -2)
        
        return forward_return
