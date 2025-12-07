import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
try:
    from torch_geometric.nn.aggr import AttentionalAggregation
    GlobalAttention = AttentionalAggregation
except ImportError:
    from torch_geometric.nn import GlobalAttention

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_normal_(m.weight)
            if isinstance(m, (nn.Conv2d, nn.Linear)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class GDF_MIL(nn.Module):
    """
    Graph-based Dynamic Feature MIL (GDF_MIL)
    Based on GDAMIL from GDF-MIL-AAAI26
    """
    def __init__(self, in_dim=512, num_classes=2, hid_dim=256, out_dim=128, k_components=10, k_neighbors=10,
                 dropout=0.1, lambda_smooth=0.0, lambda_nce=0.0, act='leaky_relu'):
        super().__init__()

        self.encoder = nn.Sequential(nn.Linear(in_dim, hid_dim), nn.LeakyReLU())
        self.bag_partition = BagPartition(hid_dim, hid_dim, out_dim, k_components)
        self.gnn = GraphAttention(hid_dim, in_channels=out_dim, k_neighbors=k_neighbors)
        self.basic = Attention(hid_dim, hid_dim, out_dim, dropout=dropout, act=act)
        self.feature_fusion = FeatureFusion(dim=out_dim)
        # Use AttentionalAggregation for newer torch_geometric versions
        try:
            from torch_geometric.nn.aggr import AttentionalAggregation
            self.bag_embedding = AttentionalAggregation(
                nn.Sequential(
                    nn.Linear(out_dim, out_dim // 2),
                    nn.LeakyReLU(),
                    nn.Linear(out_dim // 2, 1)
                )
            )
        except ImportError:
            # Fallback to GlobalAttention for older versions
            self.bag_embedding = GlobalAttention(
                nn.Sequential(
                    nn.Linear(out_dim, out_dim // 2),
                    nn.LeakyReLU(),
                    nn.Linear(out_dim // 2, 1)
                )
            )
        self.lambda_smooth = lambda_smooth
        self.lambda_nce = lambda_nce

        self.basic_linear = nn.Sequential(nn.Linear(hid_dim, out_dim), nn.LeakyReLU())
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, hid_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, num_classes)
        )
        
        self.apply(initialize_weights)

    def forward(self, X, return_WSI_attn=False, return_WSI_feature=False):
        """
        Forward pass for GDF_MIL model
        
        Args:
            X: input features [B, N, in_dim] or [N, in_dim] (will be unsqueezed)
            return_WSI_attn: whether to return attention weights
            return_WSI_feature: whether to return WSI-level features
            
        Returns:
            forward_return: dictionary containing:
                - 'logits': classification logits
                - 'WSI_attn': attention weights (if return_WSI_attn=True)
                - 'WSI_feature': WSI-level features (if return_WSI_feature=True)
        """
        forward_return = {}
        
        # Handle input shape: if 2D, add batch dimension
        if len(X.shape) == 2:
            X = X.unsqueeze(0)
        
        X = self.encoder(X)
        X_soft = self.bag_partition(X)

        X_soft, edge_index_soft = self.gnn(X_soft)  # k_components x 128

        b_gnn = self.bag_embedding(X_soft)  # 1 x 128
        b_basic, A = self.basic(X)
        b_basic = self.basic_linear(b_basic)

        b = self.feature_fusion(b_gnn, b_basic)

        logits = self.classifier(b)
        
        forward_return['logits'] = logits
        
        if return_WSI_attn:
            forward_return['WSI_attn'] = A
        
        if return_WSI_feature:
            forward_return['WSI_feature'] = b

        return forward_return


class BagPartition(nn.Module):
    """Bag Partition module using Gumbel Softmax for soft clustering"""
    def __init__(self, in_dim=512, hid_dim=512, out_dim=128, k_components=10):
        super().__init__()

        self.k_components = k_components
        self.attention = Attention(in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim)
        self.cluster_logits = nn.Linear(in_dim, k_components)

    def forward(self, X):
        X = X.squeeze(0)
        assert len(X.shape) == 2
        P = F.gumbel_softmax(self.cluster_logits(X), tau=0.5, hard=False)
        partitions = P.T @ X  # n x 512

        return partitions


class DynamicGraphBuilder(nn.Module):
    """Dynamic Graph Builder for constructing graph structure"""
    def __init__(self, dim, topk=10):
        super().__init__()
        self.W_head = nn.Linear(dim, dim)
        self.W_tail = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        self.topk = topk

    def forward(self, X):
        # n: The number of instances in the new bag
        e_h = self.W_head(X)  # n x 512
        e_t = self.W_tail(X)  # n x 512
        logits = (e_h @ e_t.T) * self.scale  # n x n

        # topk for each row
        topk_val, topk_idx = torch.topk(logits, k=self.topk, dim=-1)
        weights = F.softmax(topk_val, dim=-1)  # n' * topk

        # Construct the edge_index and edge_weight
        n = X.size(0)
        src = torch.arange(n).unsqueeze(1).expand(-1, self.topk).reshape(-1).to(X.device)
        dst = topk_idx.reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)
        edge_weight = weights.reshape(-1)
        return edge_index, edge_weight


class GraphAttention(nn.Module):
    """Graph Attention Network module"""
    def __init__(self, in_dim, in_channels, num_layers=1, k_neighbors=10):
        super().__init__()

        self.convs_list = nn.ModuleList()
        self.convs_list.append(SAGEConv(in_channels=in_dim, out_channels=in_channels))
        self.graph_builder = DynamicGraphBuilder(in_dim, topk=k_neighbors)

        for _ in range(num_layers - 1):
            self.convs_list.append(SAGEConv(in_channels=in_channels, out_channels=in_channels))

        self.lin_sum = nn.Linear(in_channels, in_channels)
        self.lin_bi = nn.Linear(in_channels, in_channels)
        self.gate_U = nn.Linear(in_channels, in_channels // 2)
        self.gate_V = nn.Linear(in_channels, in_channels // 2)
        self.gate_W = nn.Linear(in_channels // 2, in_channels)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, X):
        edge_index, edge_weight = self.graph_builder(X)

        for conv in self.convs_list:
            X = F.leaky_relu(conv(X, edge_index))

        row, col = edge_index
        # summed = Σ_j w_ij * X[j]
        summed = torch.zeros_like(X).index_add_(0, row, X[col] * edge_weight.unsqueeze(1))  # n x 128
        sum_msg = self.lin_sum(X + summed)
        bi_msg = self.lin_bi(X * summed)

        u = self.gate_U(X)  # n x 64
        v = self.gate_V(summed)  # n x 64
        g = torch.sigmoid(self.gate_W(u + v))  # n x 128

        out = F.leaky_relu(g * sum_msg + (1 - g) * bi_msg)  # n x 128
        out = self.norm(out + X)

        return out, edge_index


class FeatureFusion(nn.Module):
    """Feature Fusion module for combining GNN and attention features"""
    def __init__(self, dim=32):
        super(FeatureFusion, self).__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU()
        )

    def forward(self, x1, x2):
        combined = torch.cat([x1, x2], dim=1)
        gate = self.gate(combined)
        transformed = self.transform(combined)
        return gate * x1 + (1 - gate) * transformed


class Attention(nn.Module):
    """Attention module for basic attention mechanism"""
    def __init__(self, in_dim=512, hid_dim=512, out_dim=128, act='relu', bias=False, dropout=0):
        super(Attention, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.L = hid_dim
        self.D = out_dim
        self.K = 1

        self.feature = [nn.Linear(in_dim, hid_dim)]
        if act == 'gelu':
            self.feature += [nn.GELU()]
        elif act == 'relu':
            self.feature += [nn.ReLU()]
        elif act == 'tanh':
            self.feature += [nn.Tanh()]
        elif act == 'leaky_relu':
            self.feature += [nn.LeakyReLU()]
        else:
            self.feature += [nn.ReLU()]
        
        if dropout:
            self.feature += [nn.Dropout(self.dropout)]
        self.feature = nn.Sequential(*self.feature)

        self.attention_a = [
            nn.Linear(self.L, self.D, bias=bias),
        ]
        if act == 'gelu':
            self.attention_a += [nn.GELU()]
        elif act == 'relu':
            self.attention_a += [nn.ReLU()]
        elif act == 'tanh':
            self.attention_a += [nn.Tanh()]
        elif act == 'leaky_relu':
            self.attention_a += [nn.LeakyReLU()]
        else:
            self.attention_a += [nn.ReLU()]

        self.attention_b = [nn.Linear(self.L, self.D, bias=bias),
                            nn.Sigmoid()]

        if dropout:
            self.attention_a += [nn.Dropout(0.25)]
            self.attention_b += [nn.Dropout(0.25)]

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(self.D, self.K, bias=bias)

    def forward(self, x):
        x = self.feature(x.squeeze(0))

        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)

        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A, x)

        return x, A

