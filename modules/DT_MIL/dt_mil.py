
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

# Add the ops directory to path for MSDeformAttn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ops'))

try:
    from ops.modules import MSDeformAttn
    MSDEFORM_AVAILABLE = True
except ImportError:
    print("Warning: MSDeformAttn not available. Please compile the CUDA extension:")
    print("  cd modules/DT_MIL/ops && bash make.sh")
    MSDEFORM_AVAILABLE = False
    MSDeformAttn = None

import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def inverse_sigmoid(x, eps=1e-5):
    """Inverse sigmoid function"""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=1, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        if MSDEFORM_AVAILABLE:
            self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        else:
            raise RuntimeError("MSDeformAttn is required but not available. Please compile the CUDA extension.")
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index,
                              padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                                          indexing='ij')
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None,
                return_intermediate=False):
        output = src
        encoder_layers_output = []
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
            intermediate_output = output.detach()
            if return_intermediate:
                encoder_layers_output.append(intermediate_output)

        return output, encoder_layers_output


class NonDeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=1, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, src, src_padding_mask=None):
        q = k = v = tgt

        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        q = tgt
        k = v = src

        tgt2 = self.cross_attn(q.transpose(0, 1),
                               k.transpose(0, 1),
                               v.transpose(0, 1))[0].transpose(0, 1)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class NonDeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(self, tgt, src, src_padding_mask=None):
        output = tgt

        for lid, layer in enumerate(self.layers):
            output = layer(output, src, src_padding_mask)

        return output


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=1, dec_n_points=4, enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300,
                 num_queries=10,
                 drop_decoder=False):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = NonDeformableTransformerDecoderLayer(d_model, dim_feedforward, dropout,
                                                             activation, num_feature_levels, nhead, dec_n_points)

        self.drop_decoder = drop_decoder
        if drop_decoder:
            raise NotImplementedError("PlaneDecoder not implemented")
        else:
            self.decoder = NonDeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = None  # Not used for single-level features

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        nn.init.normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        """For feature vectors, we assume all are valid"""
        if mask is None:
            return torch.ones((mask.shape[0] if mask is not None else 1, 1, 2), device=mask.device if mask is not None else 'cpu')
        N, L = mask.shape
        valid_ratio = torch.ones((N, 1, 2), device=mask.device)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)  # (bs, h*w, c)
            mask = mask.flatten(1)  # (bs, h*w)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # (bs, h*w, c)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  # (bs, sum(h*w), c)
        mask_flatten = torch.cat(mask_flatten, 1)  # (bs, sum(h*w))
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # (bs, sum(h*w), c)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        return_intermediate = True
        memory, encoder_intermediate_output = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios,
                                                           lvl_pos_embed_flatten,
                                                           mask_flatten, return_intermediate)

        # prepare input for decoder
        bs, _, c = memory.shape
        tgt = query_embed
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)

        hs = self.decoder(tgt, memory, mask_flatten)

        return hs, encoder_intermediate_output


class DT_MIL(nn.Module):
    """
    Deformable Transformer Multiple Instance Learning (Original Implementation)
    Adapted to work with feature vectors by converting them to feature maps
    """
    def __init__(self, in_dim=1024, num_classes=2, dropout=0.1, act=nn.ReLU(), 
                 d_model=512, n_heads=8, num_encoder_layers=2, num_decoder_layers=2, 
                 dim_feedforward=2048, num_queries=10, num_feature_levels=1, 
                 enc_n_points=4, dec_n_points=4, **kwargs):
        super(DT_MIL, self).__init__()
        
        if not MSDEFORM_AVAILABLE:
            raise RuntimeError(
                "MSDeformAttn is required but not available. Please compile the CUDA extension:\n"
                "  cd modules/DT_MIL/ops && bash make.sh"
            )
        
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        
        # Input projection: feature vector -> d_model
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            act,
            nn.Dropout(dropout) if dropout else nn.Identity()
        )
        
        # Position embedding for feature vectors
        # We'll create a virtual spatial structure
        self.pos_embed = nn.Parameter(torch.randn(1, 1000, d_model))  # Max 1000 patches
        
        # Query embeddings
        self.query_embed = nn.Embedding(num_queries, d_model)
        
        # Transformer
        self.transformer = DeformableTransformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            return_intermediate_dec=False,
            num_feature_levels=num_feature_levels,
            dec_n_points=dec_n_points,
            enc_n_points=enc_n_points,
            two_stage=False,
            num_queries=num_queries,
            drop_decoder=False
        )
        
        # Classifier
        self.classifier = nn.Linear(d_model * num_queries, num_classes)
        
        # Initialize
        self._reset_parameters()
        
        # Initialize bias for classifier
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.classifier.bias.data = torch.ones(num_classes) * bias_value
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.pos_embed)
    
    def _convert_features_to_map(self, x):
        """
        Convert feature vectors (B, N, D) to feature map format (B, C, H, W)
        Creates a virtual spatial structure
        """
        batch_size, num_patches, feat_dim = x.shape
        
        # Calculate grid size (make it roughly square)
        grid_size = int(math.ceil(math.sqrt(num_patches)))
        total_size = grid_size * grid_size
        
        # Pad if necessary
        if num_patches < total_size:
            padding = torch.zeros(batch_size, total_size - num_patches, feat_dim, 
                                 device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)
        
        # Reshape to (B, C, H, W)
        x = x.view(batch_size, grid_size, grid_size, feat_dim)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Create mask (all valid)
        mask = torch.zeros(batch_size, grid_size, grid_size, dtype=torch.bool, device=x.device)
        
        # Create position embedding
        pos_embed = self.pos_embed[:, :total_size, :].unsqueeze(0).expand(batch_size, -1, -1)
        pos_embed = pos_embed.view(batch_size, grid_size, grid_size, self.d_model)
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        return x, mask, pos_embed, grid_size, num_patches
    
    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False):
        
        forward_return = {}
        
        # Handle input format
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # (N, D) -> (1, N, D)
        
        batch_size = x.shape[0]
        num_patches = x.shape[1]
        
        # Project input features
        x_proj = self.input_proj(x)  # (B, N, d_model)
        
        # Convert to feature map format
        srcs, masks, pos_embeds, grid_size, original_num_patches = self._convert_features_to_map(x_proj)
        
        # Prepare inputs for transformer (as lists for multi-level)
        srcs_list = [srcs]
        masks_list = [masks]
        pos_embeds_list = [pos_embeds]
        
        # Get query embeddings
        query_embeds = self.query_embed.weight  # (num_queries, d_model)
        
        # Forward through transformer
        hs, encoder_intermediate_output = self.transformer(
            srcs_list, masks_list, pos_embeds_list, query_embeds
        )
        
        # hs shape: (B, num_queries, d_model)
        # Flatten query features
        hs_flat = hs.view(batch_size, -1)  # (B, num_queries * d_model)
        
        # Classification
        logits = self.classifier(hs_flat)  # (B, num_classes)
        
        forward_return['logits'] = logits
        
        if return_WSI_feature:
            forward_return['WSI_feature'] = hs.mean(dim=1)  # Average over queries
        
        if return_WSI_attn:
            # For attention, we can use encoder output
            # Use uniform attention as placeholder (can be improved)
            forward_return['WSI_attn'] = torch.ones(original_num_patches, device=x.device) / original_num_patches
        
        return forward_return

