import pdb

import torch, math

# from .sinusoidal import position_encoding_1d

def position_encoding_1d(d_model: int, length: int, base: float = 10000):
    assert d_model % 2 == 0, f"Cannot use sin/cos positional encoding with odd dim (got dim={d_model})"

    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(base) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe

def apply_rotary_position_embeddings_nystrom(sinusoidal: torch.Tensor, *tensors):
    assert len(tensors) > 0, "at least one input tensor"
    N = sinusoidal.shape[0]
    # pdb.set_trace()
    cos_pos = sinusoidal[..., 1::2].repeat_interleave(2, 1).view(1,N,-1)
    sin_pos = sinusoidal[..., 0::2].repeat_interleave(2, 1).view(1,N,-1)
    cos_pos = cos_pos.expand_as(tensors[0])
    sin_pos = sin_pos.expand_as(tensors[0])

    outputs = []
    for t in tensors:
        t_r = torch.empty_like(t)
        t_r[..., 0::2] = -t[..., 1::2]
        t_r[..., 1::2] = t[..., 0::2]
        outputs.append(t * cos_pos + t_r * sin_pos)

    return outputs if len(tensors) > 1 else outputs[0]

def apply_rotary_position_embeddings(sinusoidal: torch.Tensor, *tensors):
    assert len(tensors) > 0, "at least one input tensor"
    N = sinusoidal.shape[0]
    # pdb.set_trace()
    cos_pos = sinusoidal[..., 1::2].repeat_interleave(2, 1).view(1,N,1,-1)
    sin_pos = sinusoidal[..., 0::2].repeat_interleave(2, 1).view(1,N,1,-1)
    cos_pos = cos_pos.expand_as(tensors[0])
    sin_pos = sin_pos.expand_as(tensors[0])

    outputs = []
    for t in tensors:
        t_r = torch.empty_like(t)
        t_r[..., 0::2] = -t[..., 1::2]
        t_r[..., 1::2] = t[..., 0::2]
        outputs.append(t * cos_pos + t_r * sin_pos)

    return outputs if len(tensors) > 1 else outputs[0]


class Rotary2D:

    def __init__(self, dim: int, base: float = 10000):
        self.dim = dim
        self.base = base
        self.pos_cached = None
        self.w_size_cached = None
        self.h_size_cached = None

    def forward(self, x_shape):
        H, W = int(x_shape[0].item()), int(x_shape[1].item())
        # pdb.set_trace()
        if self.pos_cached is None or self.w_size_cached != W or self.h_size_cached != H:
            # pdb.set_trace()
            print('forward')
            self.h_size_cached = H
            self.w_size_cached = W

            position_x = position_encoding_1d(H, self.dim // 2, self.base)
            position_y = position_encoding_1d(W, self.dim // 2, self.base)

            position_x = position_x.reshape(H, -1, 2)
            position_y = position_y.reshape(W, -1, 2)

            self.pos_cached = torch.empty(H * W, self.dim, dtype=torch.float).cuda()
            for i in range(H):
                for j in range(W):
                    emb = torch.cat([
                        position_x[i, 0::2],
                        position_y[j, 0::2],
                        position_x[i, 1::2],
                        position_y[j, 1::2]
                    ], 0).flatten(-2)
                    self.pos_cached[i * W + j] = emb.to(torch.float).cuda()
        return self.pos_cached

if __name__ == "__main__":
    rotary = Rotary2D(dim=384)
    data = torch.randn((1, 384, 512, 512)).cuda()
    pos_cached = rotary.forward(data)
    # print(model.eval())
    # results_dict = model(data = data)
    # pdb.set_trace()
    # print(results_dict)