import copy
from typing import Any

from transformers import PretrainedConfig

class ConchConfig(PretrainedConfig):
    model_type = "conch"

    def __init__(
        self,
        patch_size: int = 16,
        context_dim: int = 1024,
        embed_dim: int = 768,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        init_values: float = 1e-6,
        pooler_n_queries_contrast: int = 1,
        **kwargs: Any,
    ):
        self.patch_size = patch_size
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.init_values = init_values
        self.pooler_n_queries_contrast = pooler_n_queries_contrast

        super().__init__(**kwargs)