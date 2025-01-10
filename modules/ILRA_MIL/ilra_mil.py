import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Exploring Low-Rank Property in Multiple Instance Learning for Whole Slide Image Classification
Jinxi Xiang et al. ICLR 2023
"""


class MultiHeadAttention(nn.Module):
    """
    multi-head attention block
    """

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, gated=False):
        super(MultiHeadAttention, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(dim_V, num_heads)
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

        self.gate = None
        if gated:
            self.gate = nn.Sequential(nn.Linear(dim_Q, dim_V), nn.SiLU())

    def forward(self, Q, K):

        Q0 = Q

        Q = self.fc_q(Q).transpose(0, 1)
        K, V = self.fc_k(K).transpose(0, 1), self.fc_v(K).transpose(0, 1)

        A, attn_weights = self.multihead_attn(Q, K, V)

        O = (Q + A).transpose(0, 1)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)

        if self.gate is not None:
            O = O.mul(self.gate(Q0))

        return O, attn_weights


class GAB(nn.Module):
    """
    equation (16) in the paper
    """

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(GAB, self).__init__()
        self.latent = nn.Parameter(torch.Tensor(1, num_inds, dim_out))  # low-rank matrix L

        nn.init.xavier_uniform_(self.latent)

        self.project_forward = MultiHeadAttention(dim_out, dim_in, dim_out, num_heads, ln=ln, gated=True)
        self.project_backward = MultiHeadAttention(dim_in, dim_out, dim_out, num_heads, ln=ln, gated=True)

    def forward(self, X):
        """
        This process, which utilizes 'latent_mat' as a proxy, has relatively low computational complexity.
        In some respects, it is equivalent to the self-attention function applied to 'X' with itself,
        denoted as self-attention(X, X), which has a complexity of O(n^2).
        """
        latent_mat = self.latent.repeat(X.size(0), 1, 1)
        H,_ = self.project_forward(latent_mat, X)  # project the high-dimensional X into low-dimensional H
        X_hat,_ = self.project_backward(X, H)  # recover to high-dimensional space X_hat

        return X_hat




def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            # m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class NLP(nn.Module):
    """
    To obtain global features for classification, Non-Local Pooling is a more effective method
    than simple average pooling, which may result in degraded performance.
    """

    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(NLP, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mha = MultiHeadAttention(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        global_embedding = self.S.repeat(X.size(0), 1, 1)
        ret,attn_weights = self.mha(global_embedding, X)
        return ret,attn_weights


class ILRA_MIL(nn.Module):
    def __init__(self, num_layers=2, in_dim=768, num_classes=2, hidden_feat=256, num_heads=8, topk=1, ln=False):
        super().__init__()
        # stack multiple GAB block
        gab_blocks = []
        for idx in range(num_layers):
            block = GAB(dim_in=in_dim if idx == 0 else hidden_feat,
                        dim_out=hidden_feat,
                        num_heads=num_heads,
                        num_inds=topk,
                        ln=ln)
            gab_blocks.append(block)

        self.gab_blocks = nn.ModuleList(gab_blocks)

        # non-local pooling for classification
        self.pooling = NLP(dim=hidden_feat, num_heads=num_heads, num_seeds=topk, ln=ln)

        # classifier
        self.classifier = nn.Linear(in_features=hidden_feat, out_features=num_classes)

        initialize_weights(self)

    def forward(self, x, return_WSI_attn = False, return_WSI_feature = False):
        forward_return = {}
        for block in self.gab_blocks:
            x = block(x)

        feat,attn_weights = self.pooling(x)

        feat = feat.squeeze(1)
        attn_weights = attn_weights.squeeze(1).transpose(0, 1)
        logits = self.classifier(feat)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = feat
        if return_WSI_attn:
            forward_return['WSI_attn'] = attn_weights
        return forward_return
    
