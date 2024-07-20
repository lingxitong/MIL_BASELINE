import torch
import os
import pandas as pd
import glob
import pickle
from modules.AB_MIL.ab_mil import AB_MIL
from modules.CLAM_MB_MIL.clam_mb_mil import CLAM_MB_MIL
from modules.CLAM_SB_MIL.clam_sb_mil import CLAM_SB_MIL
from modules.MAX_MIL.max_mil import MAX_MIL
from modules.MEAN_MIL.mean_mil import MEAN_MIL
from modules.RRT_MIL.rrt_mil import RRT_MIL
from modules.TRANS_MIL.trans_mil import TRANS_MIL
from modules.WIKG_MIL.wikg_mil import WIKG_MIL
import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler

    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, warmup_epochs,base_lr):
        self.warmup_epochs = warmup_epochs
        self.init_lr = base_lr/self.warmup_epochs
        super().__init__(optimizer)


    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [self.init_lr * epoch for epoch in range(1,self.warmup_epochs+1)]


def get_criterion(criterion):
    if criterion == 'ce':
        return torch.nn.CrossEntropyLoss()


def get_optimizer(args,model):
    opt = args.Model.optimizer.which
    if opt == 'adam':
       lr = args.Model.optimizer.adam_config.lr
       weight_decay = args.Model.optimizer.adam_config.weight_decay
       optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
       return optimizer,lr
    elif opt == 'adamw':
       lr = args.Model.optimizer.adamw_config.lr
       weight_decay = args.Model.optimizer.adamw_config.weight_decay
       optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
       return optimizer,lr
   
   
def get_param_optimizer(args,params):
    # trainable_parameters = filter(lambda p: p.requires_grad, mil_model.parameters())
    trainable_parameters = params
    opt = args.Model.optimizer.which
    if opt == 'adam':
       lr = args.Model.optimizer.adam_config.lr
       weight_decay = args.Model.optimizer.adam_config.weight_decay
       optimizer = torch.optim.Adam(trainable_parameters, lr=lr, weight_decay=weight_decay)
       return optimizer,lr
    elif opt == 'adamw':
       lr = args.Model.optimizer.adamw_config.lr
       weight_decay = args.Model.optimizer.adamw_config.weight_decay
       optimizer = torch.optim.AdamW(trainable_parameters, lr=lr, weight_decay=weight_decay)
       return optimizer,lr


def get_scheduler(args,optimizer,base_lr):
    sch = args.Model.scheduler.which
    warmup = args.Model.scheduler.warmup
    warmup_scheduler = WarmUpLR(optimizer,warmup,base_lr)
    if sch == 'step':
        step_size = args.Model.scheduler.step_config.step_size
        gamma = args.Model.scheduler.step_config.gamma
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        return scheduler,warmup_scheduler
    elif sch == 'multi_step':
        milestones = args.Model.scheduler.multi_step_config.milestones
        gamma = args.Model.scheduler.multi_step_config.gamma
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        return scheduler,warmup_scheduler
    elif sch == 'exponential':
        gamma = args.Model.scheduler.exponential_config.gamma
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        return scheduler,warmup_scheduler
    elif sch == 'cosine':
        T_max = args.Model.scheduler.cosine_config.T_max
        eta_min = args.Model.scheduler.cosine_config.eta_min
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        return scheduler,warmup_scheduler
    elif sch == 'none':
        return None,warmup_scheduler



def _delete_best_pth_files(now_log_dir):
    pth_files = glob.glob(os.path.join(now_log_dir, '*.pth'))

    for file in pth_files:
        if 'Best' in os.path.basename(file):
            os.remove(file)
    
def save_best_model(args,model,best_epoch):
    save_path = os.path.join(args.Logs.now_log_dir,f'Best_EPOCH_{best_epoch}.pth')
    _delete_best_pth_files(args.Logs.now_log_dir)
    torch.save(model.state_dict(),save_path)
    
def save_last_model(args,model,last_epoch):
    save_path = os.path.join(args.Logs.now_log_dir,f'Last_EPOCH_{last_epoch}.pth')
    torch.save(model.state_dict(),save_path)
    
    
def dtfd_save_best_model(args,state_dict,best_epoch):
    save_path = os.path.join(args.Logs.now_log_dir,f'Best_EPOCH_{best_epoch}.pth')
    _delete_best_pth_files(args.Logs.now_log_dir)
    torch.save(state_dict,save_path)
    
def dtfd_save_last_model(args,state_dict,last_epoch):
    save_path = os.path.join(args.Logs.now_log_dir,f'Last_EPOCH_{last_epoch}.pth')
    torch.save(state_dict,save_path)

def save_log(args,epoch_info_log,best_epoch):
    log_df = pd.DataFrame(epoch_info_log)

    log_df.to_csv(os.path.join(args.Logs.now_log_dir,f'Log_seed{args.General.seed}_{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}.csv'),index=False)
    print('Global Log CSV Saved!')
    best_df = log_df[log_df['epoch'] == best_epoch]
    best_df.to_csv(os.path.join(args.Logs.now_log_dir,f'Best_Log_seed{args.General.seed}_{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}.csv'),index=False)
    print('Best Log CSV Saved!')
    
def get_model(yaml_args):
    model_name = yaml_args.General.MODEL_NAME
    if model_name == 'AB_MIL':
        mil_model = AB_MIL(yaml_args.General.num_classes,yaml_args.Model.dropout,yaml_args.Model.act,yaml_args.Model.in_dim)
        return mil_model
    elif model_name == 'CLAM_MB_MIL':
        instance_loss_fn = get_criterion(yaml_args.Model.instance_loss_fn)
        mil_model = CLAM_MB_MIL(yaml_args.Model.gate,yaml_args.Model.size_arg,yaml_args.Model.dropout,yaml_args.Model.k_sample,yaml_args.General.num_classes,
                                instance_loss_fn,yaml_args.Model.subtyping,yaml_args.Model.in_dim)
        return mil_model
    elif model_name == 'CLAM_SB_MIL':
        instance_loss_fn = get_criterion(yaml_args.Model.instance_loss_fn)
        mil_model = CLAM_SB_MIL(yaml_args.Model.gate,yaml_args.Model.size_arg,yaml_args.Model.dropout,yaml_args.Model.k_sample,yaml_args.General.num_classes,
                                instance_loss_fn,yaml_args.Model.subtyping,yaml_args.Model.in_dim)
        return mil_model
    elif model_name == 'MAX_MIL':
        mil_model = MAX_MIL(yaml_args.General.num_classes,yaml_args.Model.dropout,yaml_args.Model.act,yaml_args.Model.in_dim)
        return mil_model
    elif model_name == 'MEAN_MIL':
        mil_model = MEAN_MIL(yaml_args.General.num_classes,yaml_args.Model.dropout,yaml_args.Model.act,yaml_args.Model.in_dim)
        return mil_model
    elif model_name == 'TRANS_MIL':
        mil_model = TRANS_MIL(yaml_args.General.num_classes,yaml_args.Model.dropout,yaml_args.Model.act,yaml_args.Model.in_dim)
        return mil_model
    elif model_name == 'WIKG_MIL':
        mil_model = WIKG_MIL(yaml_args.Model.in_dim,yaml_args.Model.dim_hidden,yaml_args.Model.topk,yaml_args.General.num_classes,yaml_args.Model.agg_type,yaml_args.Model.dropout,yaml_args.Model.pool)
        return mil_model
    elif model_name == 'RRT_MIL':
        model_params = {
    'in_dim': yaml_args.Model.in_dim,
    'n_classes': yaml_args.General.num_classes,
    'dropout': yaml_args.Model.dropout,
    'act': yaml_args.Model.act,
    'region_num': yaml_args.Model.region_num,
    'pos': yaml_args.Model.pos,
    'pos_pos': yaml_args.Model.pos_pos,
    'pool':   yaml_args.Model.pool,
    'peg_k': yaml_args.Model.peg_k,
    'drop_path': yaml_args.Model.drop_path,
    'n_layers': yaml_args.Model.n_layers,
    'n_heads': yaml_args.Model.n_heads,
    'attn': yaml_args.Model.attn,
    'da_act': yaml_args.Model.da_act,
    'trans_dropout': yaml_args.Model.trans_dropout,
    'ffn': yaml_args.Model.ffn,
    'mlp_ratio': yaml_args.Model.mlp_ratio,
    'trans_dim': yaml_args.Model.trans_dim,
    'epeg': yaml_args.Model.epeg,
    'min_region_num': yaml_args.Model.min_region_num,
    'qkv_bias': yaml_args.Model.qkv_bias,
    'conv_k': yaml_args.Model.conv_k,
    'conv_2d':  yaml_args.Model.conv_2d,
    'conv_bias': yaml_args.Model.conv_bias,
    'conv_type': yaml_args.Model.conv_type,
    'region_attn': yaml_args.Model.region_attn,
    'peg_1d': yaml_args.Model.peg_1d,}
        mil_model = RRT_MIL(**model_params)
        return mil_model
    else:
        raise ValueError(f'Invalid model name: {model_name}')
    
    
