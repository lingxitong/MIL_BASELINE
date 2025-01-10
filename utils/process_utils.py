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
    
def get_process_pipeline(val_set,test_set):
    val_is_None = val_set.is_None_Dataset()
    test_is_None = test_set.is_None_Dataset()
    if (not val_is_None) and (not test_is_None):
        return 'Train_Val_Test'
    elif (not val_is_None) and test_is_None:
        return 'Train_Val'
    elif val_is_None and (not test_is_None):
        return 'Train_Test'
    else:
        raise ValueError('Both Val and Test dataset are None!')
    
