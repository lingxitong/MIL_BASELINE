import torch
from torch.utils.data import DataLoader
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


def build_train_dataloader(args, train_dataset, generator):
    num_workers = args.General.num_workers
    balanced_sampler_cfg = getattr(args.Dataset, 'balanced_sampler', False)

    dataloader_kwargs = {
        'dataset': train_dataset,
        'batch_size': 1,
        'num_workers': num_workers,
        'generator': generator
    }

    replacement = True

    if isinstance(balanced_sampler_cfg, bool):
        use_balanced_sampler = balanced_sampler_cfg
    elif balanced_sampler_cfg is None:
        use_balanced_sampler = False
    else:
        if hasattr(balanced_sampler_cfg, 'to_dict'):
            cfg_dict = balanced_sampler_cfg.to_dict()
        else:
            cfg_dict = dict(balanced_sampler_cfg)
        use_balanced_sampler = cfg_dict.get('use', False)
        replacement = cfg_dict.get('replacement', True)

    if use_balanced_sampler:
        sampler = train_dataset.get_balanced_sampler(replacement=replacement)
        dataloader_kwargs['sampler'] = sampler
        dataloader_kwargs['shuffle'] = False
    else:
        dataloader_kwargs['shuffle'] = True

    return DataLoader(**dataloader_kwargs)
    
