import torch
import os
import pandas as pd
import glob
import torch
from torch.optim.lr_scheduler import _LRScheduler
from .process_utils import get_act


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
    elif criterion == 'bce':
        return torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown criterion: {criterion}. Supported: 'ce', 'bce'")


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
    
# def save_best_model(args,model,best_epoch):
#     save_path = os.path.join(args.Logs.now_log_dir,f'Best_EPOCH_{best_epoch}.pth')
#     _delete_best_pth_files(args.Logs.now_log_dir)
#     torch.save(model.state_dict(),save_path)
    
# def save_last_model(args,model,last_epoch):
#     save_path = os.path.join(args.Logs.now_log_dir,f'Last_EPOCH_{last_epoch}.pth')
#     torch.save(model.state_dict(),save_path)
    
def save_best_model(args,model_state_dict,best_epoch):
    save_path = os.path.join(args.Logs.now_log_dir,f'Best_EPOCH_{best_epoch}.pth')
    _delete_best_pth_files(args.Logs.now_log_dir)
    torch.save(model_state_dict,save_path)
    
def save_last_model(args,model_state_dict,last_epoch):
    save_path = os.path.join(args.Logs.now_log_dir,f'Last_EPOCH_{last_epoch}.pth')
    torch.save(model_state_dict,save_path)
    
def dtfd_save_best_model(args,state_dict,best_epoch):
    save_path = os.path.join(args.Logs.now_log_dir,f'Best_EPOCH_{best_epoch}.pth')
    _delete_best_pth_files(args.Logs.now_log_dir)
    torch.save(state_dict,save_path)
    
def dtfd_save_last_model(args,state_dict,last_epoch):
    save_path = os.path.join(args.Logs.now_log_dir,f'Last_EPOCH_{last_epoch}.pth')
    torch.save(state_dict,save_path)

def save_log(args,epoch_info_log,best_epoch,process_pipeline):
    log_df = pd.DataFrame(epoch_info_log)
    log_df.to_csv(os.path.join(args.Logs.now_log_dir,f'Log_seed{args.General.seed}_{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}.csv'),index=False)
    print('Global Log CSV Saved!')
    if process_pipeline == 'Train_Test':
        return
    best_df = log_df[log_df['epoch'] == best_epoch]
    best_df.to_csv(os.path.join(args.Logs.now_log_dir,f'Best_Log_seed{args.General.seed}_{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}.csv'),index=False)
    print('Best Log CSV Saved!')
    
def get_model_from_yaml(yaml_args):
    model_name = yaml_args.General.MODEL_NAME
    if model_name == 'AB_MIL' or model_name in ['MIXUP_MIL', 'REMIX_MIL', 'RANKMIX_MIL', 'PSEBMIX_MIL', 'INSMIX_MIL']:
        from modules.AB_MIL.ab_mil import AB_MIL
        mil_model = AB_MIL(yaml_args.Model.L,yaml_args.Model.D,yaml_args.General.num_classes,yaml_args.Model.dropout,get_act(yaml_args.Model.act),yaml_args.Model.in_dim)
        return mil_model
    elif model_name == 'GATE_AB_MIL':
        from modules.GATE_AB_MIL.gate_ab_mil import GATE_AB_MIL
        mil_model = GATE_AB_MIL(yaml_args.Model.L,yaml_args.Model.D,yaml_args.General.num_classes,get_act(yaml_args.Model.act),yaml_args.Model.dropout,yaml_args.Model.bias,yaml_args.Model.in_dim)
        return mil_model
    elif model_name == 'CLAM_MB_MIL':
        from modules.CLAM_MB_MIL.clam_mb_mil import CLAM_MB_MIL
        instance_loss_fn = get_criterion(yaml_args.Model.instance_loss_fn)
        mil_model = CLAM_MB_MIL(yaml_args.Model.gate,yaml_args.Model.size_arg,yaml_args.Model.dropout,yaml_args.Model.k_sample,yaml_args.General.num_classes,
                                instance_loss_fn,yaml_args.Model.subtyping,yaml_args.Model.in_dim,get_act(yaml_args.Model.act),yaml_args.Model.instance_eval)
        return mil_model
    elif model_name == 'CLAM_SB_MIL':
        from modules.CLAM_SB_MIL.clam_sb_mil import CLAM_SB_MIL
        instance_loss_fn = get_criterion(yaml_args.Model.instance_loss_fn)
        mil_model = CLAM_SB_MIL(yaml_args.Model.gate,yaml_args.Model.size_arg,yaml_args.Model.dropout,yaml_args.Model.k_sample,yaml_args.General.num_classes,
                                instance_loss_fn,yaml_args.Model.subtyping,yaml_args.Model.in_dim,get_act(yaml_args.Model.act),yaml_args.Model.instance_eval)
        return mil_model
    elif model_name == 'FR_MIL':
        from modules.FR_MIL.fr_mil import FR_MIL
        mil_model = FR_MIL(yaml_args.General.num_classes,yaml_args.Model.num_heads,yaml_args.Model.in_dim,yaml_args.Model.k,get_act(yaml_args.Model.act),yaml_args.Model.hidden_dim)
        return mil_model
    elif model_name == 'DS_MIL':
        from modules.DS_MIL.ds_mil import FCLayer,BClassifier,DS_MIL
        device = torch.device(f'cuda:{yaml_args.General.device}')
        num_classes = yaml_args.General.num_classes
        in_dim = yaml_args.Model.in_dim
        dropout = yaml_args.Model.dropout
        act = yaml_args.Model.act
        i_classifier = FCLayer(in_dim,num_classes)
        b_classifier = BClassifier(in_dim,num_classes,dropout)
        mil_model = DS_MIL(i_classifier,b_classifier)
        return mil_model
    elif model_name == 'CDP_MIL':
        from modules.CDP_MIL.cdp_mil import CDP_MIL
        mil_model = CDP_MIL(yaml_args.Model.in_dim,yaml_args.General.num_classes,yaml_args.Model.eta,yaml_args.Model.n_sample)
        return mil_model
    elif model_name == 'LONG_MIL':
        from modules.LONG_MIL.long_mil import LONG_MIL
        mil_model = LONG_MIL(yaml_args.General.num_classes, yaml_args.Model.in_dim,yaml_args.Model.hidden_dim,yaml_args.Model.alibi_position_embedding_path,yaml_args.Model.version)
        return mil_model
    elif model_name == 'DGR_MIL':
        from modules.DGR_MIL.dgr_mil import DGR_MIL
        mil_model = DGR_MIL(yaml_args.Model.in_dim,yaml_args.General.num_classes,yaml_args.Model.L,yaml_args.Model.D,yaml_args.Model.n_lesion,yaml_args.Model.attn_mode,yaml_args.Model.dropout_node,yaml_args.Model.dropout_patch,yaml_args.Model.initialize)
        return mil_model
    elif model_name == 'MAX_MIL':
        from modules.MAX_MIL.max_mil import MAX_MIL
        mil_model = MAX_MIL(yaml_args.General.num_classes,yaml_args.Model.dropout,get_act(yaml_args.Model.act),yaml_args.Model.in_dim)
        return mil_model
    elif model_name == 'ILRA_MIL':
        from modules.ILRA_MIL.ilra_mil import ILRA_MIL
        mil_model = ILRA_MIL(yaml_args.Model.num_layers,yaml_args.Model.in_dim,yaml_args.General.num_classes,yaml_args.Model.hidden_feat,yaml_args.Model.num_heads,yaml_args.Model.topk,yaml_args.Model.ln)
        return mil_model
    elif model_name == 'MEAN_MIL':
        from modules.MEAN_MIL.mean_mil import MEAN_MIL
        mil_model = MEAN_MIL(yaml_args.General.num_classes,yaml_args.Model.dropout,get_act(yaml_args.Model.act),yaml_args.Model.in_dim)
        return mil_model
    elif model_name == 'TRANS_MIL':
        from modules.TRANS_MIL.trans_mil import TRANS_MIL
        mil_model = TRANS_MIL(yaml_args.General.num_classes,yaml_args.Model.dropout,get_act(yaml_args.Model.act),yaml_args.Model.in_dim)
        return mil_model
    elif model_name == 'WIKG_MIL':
        from modules.WIKG_MIL.wikg_mil import WIKG_MIL
        mil_model = WIKG_MIL(yaml_args.Model.in_dim,get_act(yaml_args.Model.act),yaml_args.Model.dim_hidden,yaml_args.Model.topk,yaml_args.General.num_classes,yaml_args.Model.agg_type,yaml_args.Model.dropout,yaml_args.Model.pool)
        return mil_model
    elif model_name == 'AMD_MIL':
        from modules.AMD_MIL.amd_mil import AMD_MIL
        mil_model = AMD_MIL(yaml_args.General.num_classes,yaml_args.Model.in_dim,yaml_args.Model.embed_dim,yaml_args.Model.dropout,get_act(yaml_args.Model.act),yaml_args.Model.agent_num)
        return mil_model
    elif model_name == 'AC_MIL':
        from modules.AC_MIL.ac_mil import AC_MIL
        mil_model = AC_MIL(yaml_args.Model.in_dim, yaml_args.Model.hidden_dim, yaml_args.General.num_classes,yaml_args.Model.n_token, yaml_args.Model.n_masked_patch, yaml_args.Model.mask_prob)
        return mil_model
    elif model_name == 'DTFD_MIL':
        from modules.DTFD_MIL.dtfd_mil import Classifier_1fc,Attention,DimReduction,Attention_with_Classifier
        classifier = Classifier_1fc(yaml_args.Model.mdim, yaml_args.General.num_classes, yaml_args.Model.classifier_dropout)
        attention = Attention(yaml_args.Model.mdim)
        dimReduction = DimReduction(yaml_args.Model.in_dim, yaml_args.Model.mdim, numLayer_Res=yaml_args.Model.numLayer_Res)
        attCls = Attention_with_Classifier(L=yaml_args.Model.mdim, num_cls=yaml_args.General.num_classes, droprate=yaml_args.Model.attCls_dropout)
        return (classifier,attention,dimReduction,attCls)
    elif model_name == 'DyHG_MIL':
        from modules.DyHG_MIL.dyhg_mil import DyHG_MIL
        mil_model = DyHG_MIL(
            in_dim=yaml_args.Model.in_dim,
            emb_dim=yaml_args.Model.emb_dim,
            num_classes=yaml_args.General.num_classes,
            dropout=yaml_args.Model.dropout,
            hyper_num=yaml_args.Model.hyper_num,
            num_layers=yaml_args.Model.num_layers,
            tau=yaml_args.Model.tau,
            act=get_act(yaml_args.Model.act)
        )
        return mil_model
    elif model_name == 'DG_MIL':
        from modules.DG_MIL.dg_mil import DG_MIL
        projection_dim = yaml_args.Model.projection_dim if hasattr(yaml_args.Model, 'projection_dim') else 768
        mil_model = DG_MIL(
            in_dim=yaml_args.Model.in_dim,
            num_classes=yaml_args.General.num_classes,
            dropout=yaml_args.Model.dropout,
            act=get_act(yaml_args.Model.act),
            projection_dim=projection_dim
        )
        return mil_model
    elif model_name == 'DT_MIL':
        from modules.DT_MIL.dt_mil import DT_MIL
        mil_model = DT_MIL(
            in_dim=yaml_args.Model.in_dim,
            num_classes=yaml_args.General.num_classes,
            dropout=yaml_args.Model.dropout,
            act=get_act(yaml_args.Model.act)
        )
        return mil_model
    elif model_name == 'Mamba_MIL':
        from modules.Mamba_MIL.mamba_mil import Mamba_MIL
        layer = yaml_args.Model.layer if hasattr(yaml_args.Model, 'layer') else 2
        rate = yaml_args.Model.rate if hasattr(yaml_args.Model, 'rate') else 10
        mamba_type = yaml_args.Model.mamba_type if hasattr(yaml_args.Model, 'mamba_type') else 'SRMamba'
        mil_model = Mamba_MIL(
            in_dim=yaml_args.Model.in_dim,
            num_classes=yaml_args.General.num_classes,
            dropout=yaml_args.Model.dropout,
            act=get_act(yaml_args.Model.act),
            layer=layer,
            rate=rate,
            mamba_type=mamba_type
        )
        return mil_model
    elif model_name == 'MHIM_MIL':
        from modules.MHIM_MIL.mhim_mil import MHIM_MIL
        mil_model = MHIM_MIL(
            in_dim=yaml_args.Model.in_dim,
            num_classes=yaml_args.General.num_classes,
            dropout=yaml_args.Model.dropout,
            act=get_act(yaml_args.Model.act)
        )
        return mil_model
    elif model_name == 'Micro_MIL':
        from modules.Micro_MIL.micro_mil import Micro_MIL
        cluster_number = yaml_args.Model.cluster_number if hasattr(yaml_args.Model, 'cluster_number') else 36
        hidden_dim = yaml_args.Model.hidden_dim if hasattr(yaml_args.Model, 'hidden_dim') else 128
        layer = yaml_args.Model.layer if hasattr(yaml_args.Model, 'layer') else 2
        alpha = yaml_args.Model.alpha if hasattr(yaml_args.Model, 'alpha') else 1.0
        shuffle = yaml_args.Model.shuffle if hasattr(yaml_args.Model, 'shuffle') else False
        mil_model = Micro_MIL(
            in_dim=yaml_args.Model.in_dim,
            num_classes=yaml_args.General.num_classes,
            dropout=yaml_args.Model.dropout,
            act=get_act(yaml_args.Model.act),
            cluster_number=cluster_number,
            hidden_dim=hidden_dim,
            layer=layer,
            alpha=alpha,
            shuffle=shuffle
        )
        return mil_model
    elif model_name == 'MSM_MIL':
        from modules.MSM_MIL.msm_mil import MSM_MIL
        layer = yaml_args.Model.layer if hasattr(yaml_args.Model, 'layer') else 2
        rate = yaml_args.Model.rate if hasattr(yaml_args.Model, 'rate') else 10
        mamba_type = yaml_args.Model.mamba_type if hasattr(yaml_args.Model, 'mamba_type') else 'SRMamba'
        mil_model = MSM_MIL(
            in_dim=yaml_args.Model.in_dim,
            num_classes=yaml_args.General.num_classes,
            dropout=yaml_args.Model.dropout,
            act=get_act(yaml_args.Model.act),
            layer=layer,
            rate=rate,
            mamba_type=mamba_type
        )
        return mil_model
    elif model_name == 'MSMMIL_MIL':
        from modules.MSMMIL_MIL.msmmil_mil import MSMMIL_MIL
        layer = yaml_args.Model.layer if hasattr(yaml_args.Model, 'layer') else 2
        rate = yaml_args.Model.rate if hasattr(yaml_args.Model, 'rate') else 10
        mamba_type = yaml_args.Model.mamba_type if hasattr(yaml_args.Model, 'mamba_type') else 'SRMamba'
        mil_model = MSMMIL_MIL(
            in_dim=yaml_args.Model.in_dim,
            num_classes=yaml_args.General.num_classes,
            dropout=yaml_args.Model.dropout,
            act=get_act(yaml_args.Model.act),
            layer=layer,
            rate=rate,
            mamba_type=mamba_type
        )
        return mil_model
    elif model_name == 'PA_MIL':
        from modules.PA_MIL.pa_mil import PA_MIL
        embed_dim = yaml_args.Model.embed_dim if hasattr(yaml_args.Model, 'embed_dim') else 512
        num_layers = yaml_args.Model.num_layers if hasattr(yaml_args.Model, 'num_layers') else 2
        num_heads = yaml_args.Model.num_heads if hasattr(yaml_args.Model, 'num_heads') else 8
        dim_head = yaml_args.Model.dim_head if hasattr(yaml_args.Model, 'dim_head') else 64
        mil_model = PA_MIL(
            in_dim=yaml_args.Model.in_dim,
            num_classes=yaml_args.General.num_classes,
            dropout=yaml_args.Model.dropout,
            act=get_act(yaml_args.Model.act),
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_head=dim_head
        )
        return mil_model
    elif model_name == 'DT_MIL':
        from modules.DT_MIL.dt_mil import DT_MIL
        d_model = yaml_args.Model.d_model if hasattr(yaml_args.Model, 'd_model') else 512
        n_heads = yaml_args.Model.n_heads if hasattr(yaml_args.Model, 'n_heads') else 8
        num_encoder_layers = yaml_args.Model.num_encoder_layers if hasattr(yaml_args.Model, 'num_encoder_layers') else 2
        dim_feedforward = yaml_args.Model.dim_feedforward if hasattr(yaml_args.Model, 'dim_feedforward') else 2048
        num_queries = yaml_args.Model.num_queries if hasattr(yaml_args.Model, 'num_queries') else 10
        mil_model = DT_MIL(
            in_dim=yaml_args.Model.in_dim,
            num_classes=yaml_args.General.num_classes,
            dropout=yaml_args.Model.dropout,
            act=get_act(yaml_args.Model.act),
            d_model=d_model,
            n_heads=n_heads,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            num_queries=num_queries
        )
        return mil_model
    elif model_name == 'CA_MIL':
        from modules.CA_MIL.ca_mil import CA_MIL
        L = yaml_args.Model.L if hasattr(yaml_args.Model, 'L') else 512
        D = yaml_args.Model.D if hasattr(yaml_args.Model, 'D') else 128
        mil_model = CA_MIL(
            L=L,
            D=D,
            num_classes=yaml_args.General.num_classes,
            dropout=yaml_args.Model.dropout,
            act=get_act(yaml_args.Model.act),
            in_dim=yaml_args.Model.in_dim
        )
        return mil_model
    elif model_name == 'ADD_MIL':
        from modules.ADD_MIL.add_mil import ADD_MIL
        L = yaml_args.Model.L if hasattr(yaml_args.Model, 'L') else 512
        D = yaml_args.Model.D if hasattr(yaml_args.Model, 'D') else 128
        mil_model = ADD_MIL(
            L=L,
            D=D,
            num_classes=yaml_args.General.num_classes,
            dropout=yaml_args.Model.dropout,
            act=get_act(yaml_args.Model.act),
            in_dim=yaml_args.Model.in_dim
        )
        return mil_model
    elif model_name == 'MHIM_MIL':
        from modules.MHIM_MIL.mhim_mil import MHIM_MIL
        mlp_dim = yaml_args.Model.mlp_dim if hasattr(yaml_args.Model, 'mlp_dim') else 512
        head = yaml_args.Model.head if hasattr(yaml_args.Model, 'head') else 8
        mask_ratio = yaml_args.Model.mask_ratio if hasattr(yaml_args.Model, 'mask_ratio') else 0.0
        baseline = yaml_args.Model.baseline if hasattr(yaml_args.Model, 'baseline') else 'selfattn'
        mil_model = MHIM_MIL(
            in_dim=yaml_args.Model.in_dim,
            num_classes=yaml_args.General.num_classes,
            dropout=yaml_args.Model.dropout,
            act=get_act(yaml_args.Model.act),
            mlp_dim=mlp_dim,
            head=head,
            mask_ratio=mask_ratio,
            baseline=baseline
        )
        return mil_model
    elif model_name == 'IB_MIL':
        from modules.IB_MIL.ib_mil import IB_MIL
        L = yaml_args.Model.L if hasattr(yaml_args.Model, 'L') else 512
        D = yaml_args.Model.D if hasattr(yaml_args.Model, 'D') else 128
        beta = yaml_args.Model.beta if hasattr(yaml_args.Model, 'beta') else 0.1
        mil_model = IB_MIL(
            L=L,
            D=D,
            num_classes=yaml_args.General.num_classes,
            dropout=yaml_args.Model.dropout,
            act=get_act(yaml_args.Model.act),
            in_dim=yaml_args.Model.in_dim,
            beta=beta
        )
        return mil_model
    elif model_name == 'RRT_MIL':
        from modules.RRT_MIL.rrt_mil import RRT_MIL
        L = yaml_args.Model.L if hasattr(yaml_args.Model, 'L') else 512
        D = yaml_args.Model.D if hasattr(yaml_args.Model, 'D') else 128
        mil_model = RRT_MIL(
            L=L,
            D=D,
            num_classes=yaml_args.General.num_classes,
            dropout=yaml_args.Model.dropout,
            act=get_act(yaml_args.Model.act),
            in_dim=yaml_args.Model.in_dim
        )
        return mil_model
    elif model_name == 'S4_MIL':
        from modules.S4_MIL.s4_mil import S4_MIL
        d_model = yaml_args.Model.d_model if hasattr(yaml_args.Model, 'd_model') else 512
        d_state = yaml_args.Model.d_state if hasattr(yaml_args.Model, 'd_state') else 32
        n_layers = yaml_args.Model.n_layers if hasattr(yaml_args.Model, 'n_layers') else 1
        mil_model = S4_MIL(
            in_dim=yaml_args.Model.in_dim,
            num_classes=yaml_args.General.num_classes,
            dropout=yaml_args.Model.dropout,
            act=get_act(yaml_args.Model.act),
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers
        )
        return mil_model
    elif model_name == 'PGCN_MIL':
        from modules.PGCN_MIL.pgcn_mil import PGCN_MIL
        hidden_dim = yaml_args.Model.hidden_dim if hasattr(yaml_args.Model, 'hidden_dim') else 256
        n_layers = yaml_args.Model.n_layers if hasattr(yaml_args.Model, 'n_layers') else 2
        k = yaml_args.Model.k if hasattr(yaml_args.Model, 'k') else 5
        use_dgl = yaml_args.Model.use_dgl if hasattr(yaml_args.Model, 'use_dgl') else False
        mil_model = PGCN_MIL(
            in_dim=yaml_args.Model.in_dim,
            num_classes=yaml_args.General.num_classes,
            dropout=yaml_args.Model.dropout,
            act=get_act(yaml_args.Model.act),
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            k=k,
            use_dgl=use_dgl
        )
        return mil_model
    elif model_name == 'NCIE_MIL':
        from modules.NCIE_MIL.nc_ie_mil import NcIEMIL
        in_dim = yaml_args.Model.in_dim
        in_chans = yaml_args.Model.in_chans if hasattr(yaml_args.Model, 'in_chans') else 256
        latent_dim = yaml_args.Model.latent_dim if hasattr(yaml_args.Model, 'latent_dim') else 1024
        num_heads = yaml_args.Model.num_heads if hasattr(yaml_args.Model, 'num_heads') else 4
        ratio = yaml_args.Model.ratio if hasattr(yaml_args.Model, 'ratio') else 32
        qkv_bias = yaml_args.Model.qkv_bias if hasattr(yaml_args.Model, 'qkv_bias') else False
        qk_scale = yaml_args.Model.qk_scale if hasattr(yaml_args.Model, 'qk_scale') else None
        attn_drop = yaml_args.Model.attn_drop if hasattr(yaml_args.Model, 'attn_drop') else 0.0
        proj_drop = yaml_args.Model.proj_drop if hasattr(yaml_args.Model, 'proj_drop') else 0.0
        conv_drop = yaml_args.Model.conv_drop if hasattr(yaml_args.Model, 'conv_drop') else 0.0
        mode = yaml_args.Model.mode if hasattr(yaml_args.Model, 'mode') else 'cross'
        mil_model = NcIEMIL(
            in_dim=in_dim,
            in_chans=in_chans,
            latent_dim=latent_dim,
            n_classes=yaml_args.General.num_classes,
            num_heads=num_heads,
            ratio=ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            conv_drop=conv_drop,
            mode=mode
        )
        return mil_model
    elif model_name == 'MAMBA2D_MIL':
        import importlib
        mamba2d_module = importlib.import_module('modules.MAMBA2D_MIL.mamba2d_mil')
        Mamba2D_MIL = mamba2d_module.Mamba2D_MIL
        d_model = yaml_args.Model.d_model if hasattr(yaml_args.Model, 'd_model') else 512
        d_state = yaml_args.Model.d_state if hasattr(yaml_args.Model, 'd_state') else 16
        n_layers = yaml_args.Model.n_layers if hasattr(yaml_args.Model, 'n_layers') else 2
        grid_size = yaml_args.Model.grid_size if hasattr(yaml_args.Model, 'grid_size') else None
        mil_model = Mamba2D_MIL(
            in_dim=yaml_args.Model.in_dim,
            num_classes=yaml_args.General.num_classes,
            dropout=yaml_args.Model.dropout,
            act=get_act(yaml_args.Model.act),
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            grid_size=grid_size
        )
        return mil_model
    elif model_name == 'GDF_MIL':
        from modules.GDF_MIL.gdf_mil import GDF_MIL
        in_dim = yaml_args.Model.in_dim
        hid_dim = yaml_args.Model.hid_dim if hasattr(yaml_args.Model, 'hid_dim') else 256
        out_dim = yaml_args.Model.out_dim if hasattr(yaml_args.Model, 'out_dim') else 128
        k_components = yaml_args.Model.k_components if hasattr(yaml_args.Model, 'k_components') else 10
        k_neighbors = yaml_args.Model.k_neighbors if hasattr(yaml_args.Model, 'k_neighbors') else 10
        dropout = yaml_args.Model.dropout if hasattr(yaml_args.Model, 'dropout') else 0.1
        lambda_smooth = yaml_args.Model.lambda_smooth if hasattr(yaml_args.Model, 'lambda_smooth') else 0.0
        lambda_nce = yaml_args.Model.lambda_nce if hasattr(yaml_args.Model, 'lambda_nce') else 0.0
        act = yaml_args.Model.act if hasattr(yaml_args.Model, 'act') else 'leaky_relu'
        mil_model = GDF_MIL(
            in_dim=in_dim,
            num_classes=yaml_args.General.num_classes,
            hid_dim=hid_dim,
            out_dim=out_dim,
            k_components=k_components,
            k_neighbors=k_neighbors,
            dropout=dropout,
            lambda_smooth=lambda_smooth,
            lambda_nce=lambda_nce,
            act=act
        )
        return mil_model
    else:
        raise ValueError(f'Invalid model name: {model_name}')
    
def model_select(REVERSE,args,mil_model_state_dict,val_metrics,best_model_metric,best_val_metric,epoch,best_epoch):
    if val_metrics == None:
        return best_val_metric,best_epoch
    if REVERSE and val_metrics[best_model_metric] < best_val_metric:
        best_epoch = epoch+1
        best_val_metric = val_metrics[best_model_metric]
        save_best_model(args,mil_model_state_dict,best_epoch)

    elif not REVERSE and val_metrics[best_model_metric] > best_val_metric:
        best_epoch = epoch+1
        best_val_metric = val_metrics[best_model_metric]
        save_best_model(args,mil_model_state_dict,best_epoch)
    return best_val_metric,best_epoch
        

def dtfd_model_select():
    pass
