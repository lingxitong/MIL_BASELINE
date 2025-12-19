"""
AEM_MIL - Attention Entropy Maximization Multiple Instance Learning
Reference: Attention Entropy Maximization for Multiple Instance Learning based Whole Slide Image Classification
Paper: (需要补充论文信息)

Key idea: Maximize attention entropy to encourage diverse attention patterns
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

class AttentionEntropyMaximization(nn.Module):
    """
    Attention mechanism with entropy maximization
    Encourages diverse attention patterns by maximizing entropy
    """
    def __init__(self, L=512, D=128, K=1, temperature=1.0):
        super().__init__()
        self.L = L
        self.D = D
        self.K = K
        self.temperature = temperature
        
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )
        
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )
        
        self.attention_weights = nn.Linear(self.D, self.K)
    
    def forward(self, x, return_entropy=False):
        """
        Args:
            x: (N, L) - patch features
            return_entropy: whether to return entropy value
        Returns:
            A: (K, N) - attention weights
            entropy: (optional) entropy value
        """
        A_V = self.attention_V(x)  # (N, D)
        A_U = self.attention_U(x)  # (N, D)
        A = self.attention_weights(A_V * A_U)  # (N, K)
        A = torch.transpose(A, 0, 1)  # (K, N)
        
        # Apply temperature scaling
        A_scaled = A / self.temperature
        
        # Softmax normalization
        A = F.softmax(A_scaled, dim=1)
        
        if return_entropy:
            # Calculate entropy: -sum(p * log(p))
            entropy = -torch.sum(A * torch.log(A + 1e-8), dim=1).mean()
            return A, entropy
        return A

class AEM_MIL(nn.Module):
    """
    Attention Entropy Maximization Multiple Instance Learning
    
    Maximizes attention entropy to encourage diverse attention patterns.
    Reference: Attention Entropy Maximization for Multiple Instance Learning based Whole Slide Image Classification
    """
    def __init__(self, L=512, D=128, num_classes=2, dropout=0.1, act=nn.ReLU(), in_dim=512, temperature=1.0, lambda_entropy=0.1):
        super(AEM_MIL, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.L = L
        self.D = D
        self.K = 1
        self.temperature = temperature
        self.lambda_entropy = lambda_entropy  # Weight for entropy regularization
        
        # Feature encoder
        self.feature = nn.Sequential(
            nn.Linear(in_dim, self.L),
            act
        )
        
        if dropout:
            self.feature.add_module('dropout', nn.Dropout(dropout))
        
        # Attention with entropy maximization
        self.attention = AttentionEntropyMaximization(L=self.L, D=self.D, K=self.K, temperature=temperature)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, self.num_classes),
        )
        
        self.apply(initialize_weights)
    
    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False, return_entropy=False):
        """
        Args:
            x: (N, D) or (1, N, D) - patch features
            return_WSI_attn: bool
            return_WSI_feature: bool
            return_entropy: bool - return attention entropy for loss computation
        """
        forward_return = {}
        
        # Handle input format
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # (N, D) -> (1, N, D)
        
        # Feature encoding
        feature = self.feature(x)  # (1, N, L)
        feature = feature.squeeze(0)  # (N, L)
        
        # Attention with entropy maximization
        if return_entropy:
            A, entropy = self.attention(feature, return_entropy=True)
            forward_return['entropy'] = entropy
        else:
            A = self.attention(feature, return_entropy=False)
        
        A_ori = A.clone()
        # A shape: (K, N) where K=1
        
        # Bag-level representation
        M = torch.mm(A, feature)  # (K, L)
        
        # Classification
        logits = self.classifier(M)  # (K, num_classes)
        
        forward_return['logits'] = logits
        
        if return_WSI_feature:
            forward_return['WSI_feature'] = M.squeeze(0)
        
        if return_WSI_attn:
            forward_return['WSI_attn'] = A_ori.squeeze(0)  # (N,)
        
        return forward_return


