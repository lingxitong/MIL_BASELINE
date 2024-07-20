import torch
def get_act(act):
    if act.lower() == 'relu':
        return torch.nn.ReLU()
    elif act.lower() == 'gelu':
        return torch.nn.GELU()
    elif act.lower() == 'leakyrelu':
        return torch.nn.LeakyReLU()
    elif act.lower() == 'sigmoid':
        return torch.nn.Sigmoid()
    elif act.lower() == 'tanh':
        return torch.nn.Tanh()
    elif act.lower() == 'silu':
        return torch.nn.SiLU()
    else:
        raise ValueError(f'Invalid activation function: {act}')