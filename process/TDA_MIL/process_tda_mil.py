"""
Process script for TDA_MIL
"""
import torch
from torch.utils.data import DataLoader
from modules.TDA_MIL.tda_mil import TDA_MIL
from utils.process_utils import get_process_pipeline, get_act
from utils.wsi_utils import WSI_Dataset
from utils.general_utils import set_global_seed, init_epoch_info_log, add_epoch_info_log, early_stop
from utils.model_utils import get_optimizer, get_scheduler, get_criterion, save_last_model, save_log, model_select
from utils.loop_utils import train_loop, val_loop
from tqdm import tqdm

def process_TDA_MIL(args):
    train_dataset = WSI_Dataset(args.Dataset.dataset_csv_path, 'train')
    val_dataset = WSI_Dataset(args.Dataset.dataset_csv_path, 'val')
    test_dataset = WSI_Dataset(args.Dataset.dataset_csv_path, 'test')
    process_pipeline = get_process_pipeline(val_dataset, test_dataset)
    args.General.process_pipeline = process_pipeline
    
    generator = torch.Generator()
    generator.manual_seed(args.General.seed)
    set_global_seed(args.General.seed)
    num_workers = args.General.num_workers
    use_balanced_sampler = args.Dataset.balanced_sampler.use
    if use_balanced_sampler:
        sampler = train_dataset.get_balanced_sampler(replacement=args.Dataset.balanced_sampler.replacement)
        train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=num_workers, generator=generator, sampler=sampler)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=num_workers, generator=generator)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    
    print('DataLoader Ready!')
    
    device = torch.device(f'cuda:{args.General.device}')
    num_classes = args.General.num_classes
    in_dim = args.Model.in_dim
    embed_dim = args.Model.embed_dim if hasattr(args.Model, 'embed_dim') else 512
    num_layers = args.Model.num_layers if hasattr(args.Model, 'num_layers') else 2
    num_heads = args.Model.num_heads if hasattr(args.Model, 'num_heads') else 8
    mlp_ratio = args.Model.mlp_ratio if hasattr(args.Model, 'mlp_ratio') else 4.0
    dropout = args.Model.dropout if hasattr(args.Model, 'dropout') else 0.1
    attn_dropout = args.Model.attn_dropout if hasattr(args.Model, 'attn_dropout') else 0.1
    td_mlp_ratio = args.Model.td_mlp_ratio if hasattr(args.Model, 'td_mlp_ratio') else 2.0
    clamp_min = args.Model.clamp_min if hasattr(args.Model, 'clamp_min') else 0.0
    clamp_max = args.Model.clamp_max if hasattr(args.Model, 'clamp_max') else 1.0
    force_cls_score = args.Model.force_cls_score if hasattr(args.Model, 'force_cls_score') else 1.0
    share_weights_step12 = args.Model.share_weights_step12 if hasattr(args.Model, 'share_weights_step12') else True
    max_seq_len = args.Model.max_seq_len if hasattr(args.Model, 'max_seq_len') else 2048
    
    mil_model = TDA_MIL(
        in_dim=in_dim,
        embed_dim=embed_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        attn_dropout=attn_dropout,
        td_mlp_ratio=td_mlp_ratio,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        force_cls_score=force_cls_score,
        share_weights_step12=share_weights_step12,
        max_seq_len=max_seq_len
    )
    mil_model.to(device)
    
    print('Model Ready!')
    
    optimizer, base_lr = get_optimizer(args, mil_model)
    scheduler, warmup_scheduler = get_scheduler(args, optimizer, base_lr)
    criterion = get_criterion(args.Model.criterion)
    warmup_epoch = args.Model.scheduler.warmup
    
    epoch_info_log = init_epoch_info_log()
    best_model_metric = args.General.best_model_metric
    REVERSE = False
    best_val_metric = 0
    if best_model_metric == 'val_loss':
        REVERSE = True
        best_val_metric = 9999
    best_epoch = 1
    print('Start Process!')
    print('Using Process Pipeline:', process_pipeline)
    
    for epoch in tqdm(range(args.General.num_epochs), colour='GREEN'):
        if epoch + 1 <= warmup_epoch:
            now_scheduler = warmup_scheduler
        else:
            now_scheduler = scheduler
        train_loss, cost_time = train_loop(device, mil_model, train_dataloader, criterion, optimizer, now_scheduler)
        if process_pipeline == 'Train_Val_Test':
            val_loss, val_metrics = val_loop(device, num_classes, mil_model, val_dataloader, criterion)
            test_loss, test_metrics = val_loop(device, num_classes, mil_model, test_dataloader, criterion)
        elif process_pipeline == 'Train_Val':
            val_loss, val_metrics = val_loop(device, num_classes, mil_model, val_dataloader, criterion)
            test_loss, test_metrics = None, None
        elif process_pipeline == 'Train_Test':
            val_loss, val_metrics, test_loss, test_metrics = None, None, None, None
            if epoch + 1 == args.General.num_epochs:
                test_loss, test_metrics = val_loop(device, num_classes, mil_model, test_dataloader, criterion)
        
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        print('----------------INFO----------------\n')
        print(f'{FAIL}EPOCH:{ENDC}{epoch+1},  Train_Loss:{train_loss},  Val_Loss:{val_loss},  Test_Loss:{test_loss},  Cost_Time:{cost_time}\n')
        print(f'{FAIL}Val_Metrics:  {ENDC}{val_metrics}\n')
        print(f'{FAIL}Test_Metrics:  {ENDC}{test_metrics}\n')
        add_epoch_info_log(epoch_info_log, epoch, train_loss, val_loss, test_loss, val_metrics, test_metrics)
        
        best_val_metric, best_epoch = model_select(REVERSE, args, mil_model.state_dict(), val_metrics, best_model_metric, best_val_metric, epoch, best_epoch)
        
        if early_stop(args, epoch_info_log, process_pipeline, epoch, mil_model.state_dict(), best_epoch):
            break
        
        if epoch + 1 == args.General.num_epochs:
            save_last_model(args, mil_model.state_dict(), epoch + 1)
            save_log(args, epoch_info_log, best_epoch, process_pipeline)

