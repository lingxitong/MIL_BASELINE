"""
IIB_MIL - Interventional Instance-Bag Multiple Instance Learning
Reference: https://github.com/TencentAILabHealthcare/IIB-MIL
Paper: Interventional Instance-Bag Multiple Instance Learning for Whole Slide Image Classification
"""
import copy
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

def _get_clones(module, N):
    return nn.ModuleList([nn.ModuleList([copy.deepcopy(module) for _ in range(N)])])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

import copy

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu", n_heads=8):
        super().__init__()
        # Cross attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    
    def forward(self, tgt, src):
        q = k = v = tgt
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        q = tgt
        k = v = src
        tgt2 = self.cross_attn(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        tgt = self.forward_ffn(tgt)
        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
    
    def forward(self, tgt, src):
        output = tgt
        for layer in self.layers:
            output = layer(output, src)
        return output

class IIBMIL_Encoder(nn.Module):
    def __init__(self, num_classes, patch_dim, dim, depth, dropout=0.0, emb_dropout=0.0):
        super().__init__()
        self.patch_dim = patch_dim
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.z_dim = dim
        self.dropout = nn.Dropout(emb_dropout)
        self.depth = depth
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.z_dim, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        for i in range(self.depth):
            if i == 0:
                input_dim = patch_dim
            else:
                input_dim = self.z_dim
            
            self.add_module("depth_{}".format(i), nn.Sequential(
                nn.Linear(input_dim, self.z_dim * 2),
                nn.Tanh(),
                nn.LayerNorm(self.z_dim * 2),
                nn.Dropout(dropout),
                nn.Linear(self.z_dim * 2, self.z_dim * 4),
                nn.Tanh(),
                nn.LayerNorm(self.z_dim * 4),
                nn.Dropout(dropout),
                nn.Linear(self.z_dim * 4, self.z_dim * 2),
                nn.Tanh(),
                nn.LayerNorm(self.z_dim * 2),
                nn.Dropout(dropout),
                nn.Linear(self.z_dim * 2, self.z_dim * 2),
                nn.Tanh(),
                nn.LayerNorm(self.z_dim * 2),
                nn.Dropout(dropout),
                nn.Linear(self.z_dim * 2, self.z_dim * 1),
                nn.Tanh(),
                nn.LayerNorm(self.z_dim * 1),
                nn.Dropout(dropout),
            ))
            self.add_module("encode_{}".format(i), nn.TransformerEncoderLayer(d_model=self.z_dim, nhead=2))
            self.add_module("trans_{}".format(i), nn.TransformerEncoder(self._modules["encode_{}".format(i)], num_layers=1))
            self.add_module("res_{}".format(i), nn.Linear(patch_dim, self.z_dim))
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.z_dim * 1),
            nn.Linear(self.z_dim * 1, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, N, D) - patch features
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        
        img = x.clone()
        b, n, dimen = img.shape
        
        for i in range(self.depth):
            x = self._modules["depth_{}".format(i)](x)
            x = self._modules["trans_{}".format(i)](x.transpose(0, 1))
            x = x.transpose(0, 1)
            x = x + torch.relu(self._modules["res_{}".format(i)](img))
        
        patch_classifier_output = self.mlp_head(x)
        return patch_classifier_output, x

class IIBMIL_Decoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_decoder_layers=2, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(d_model, dim_feedforward, dropout, activation, nhead)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
    
    def forward(self, tgt, memory):
        hs = self.decoder(tgt, memory)
        return hs

class IIB_MIL(nn.Module):
    """
    Interventional Instance-Bag Multiple Instance Learning
    
    Uses Transformer Encoder-Decoder architecture with query embeddings.
    Reference: https://github.com/TencentAILabHealthcare/IIB-MIL
    """
    def __init__(self, in_dim=1024, num_classes=2, dim=256, depth=3, num_queries=5, dropout=0.1):
        super(IIB_MIL, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.num_queries = num_queries
        
        self.patch_encoder = IIBMIL_Encoder(
            num_classes=num_classes,
            patch_dim=in_dim,
            dim=dim,
            depth=depth,
            dropout=dropout,
            emb_dropout=dropout
        )
        
        self.wsi_aggregator = IIBMIL_Decoder(
            d_model=dim,
            nhead=4,
            num_decoder_layers=2,
            dim_feedforward=dim * 2,
            dropout=dropout,
            activation="relu"
        )
        
        self.query_embed = nn.Embedding(num_queries, dim)
        self.wsi_classifier = nn.Linear(dim * num_queries, num_classes)
        
        self.apply(initialize_weights)
    
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
        
        bs, patch_num, _ = x.shape
        
        # Patch encoder
        patch_classifier_output, x_instance = self.patch_encoder(x)
        # x_instance: (bs, patch_num, dim)
        # patch_classifier_output: (bs, patch_num, num_classes)
        
        # Query embeddings
        tgt = self.query_embed.weight  # (num_queries, dim)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)  # (bs, num_queries, dim)
        
        # WSI aggregator (decoder)
        hs = self.wsi_aggregator(tgt, x_instance)  # (bs, num_queries, dim)
        
        # Classification
        wsi_classifier_output = self.wsi_classifier(hs.view(bs, -1))  # (bs, num_classes)
        
        forward_return['logits'] = wsi_classifier_output
        
        if return_WSI_feature:
            forward_return['WSI_feature'] = hs.mean(dim=1).squeeze(0)  # Average over queries
        
        if return_WSI_attn:
            # Use patch classifier output as attention (softmax over patches)
            attn = F.softmax(patch_classifier_output.max(dim=-1)[0], dim=1)  # (bs, patch_num)
            forward_return['WSI_attn'] = attn.squeeze(0)
        
        return forward_return

