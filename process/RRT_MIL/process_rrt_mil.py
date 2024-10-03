import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
from modules.RRT_MIL.rrt_mil import RRT_MIL
from utils.wsi_utils import *
from utils.general_utils import *
from utils.model_utils import *
from utils.loop_utils import *
from tqdm import tqdm

    
def process_RRT_MIL(args):
    print_args(args)

    train_dataset = WSI_Dataset(args.Dataset.dataset_csv_path,'train')
    val_dataset = WSI_Dataset(args.Dataset.dataset_csv_path,'val')
    test_dataset = WSI_Dataset(args.Dataset.dataset_csv_path,'test')
    '''
    generator settings
    '''
    
    generator = torch.Generator()
    generator.manual_seed(args.General.seed) 
    set_global_seed(args.General.seed)
    num_workers = args.General.num_workers
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers = num_workers,generator=generator)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    
    print('DataLoader Ready!')
    model_params = {
    'in_dim': args.Model.in_dim,
    'num_classes': args.General.num_classes,
    'dropout': args.Model.dropout,
    'act': args.Model.act,
    'region_num': args.Model.region_num,
    'pos': args.Model.pos,
    'pos_pos': args.Model.pos_pos,
    'pool':   args.Model.pool,
    'peg_k': args.Model.peg_k,
    'drop_path': args.Model.drop_path,
    'n_layers': args.Model.n_layers,
    'n_heads': args.Model.n_heads,
    'attn': args.Model.attn,
    'da_act': args.Model.da_act,
    'trans_dropout': args.Model.trans_dropout,
    'ffn': args.Model.ffn,
    'mlp_ratio': args.Model.mlp_ratio,
    'trans_dim': args.Model.trans_dim,
    'epeg': args.Model.epeg,
    'min_region_num': args.Model.min_region_num,
    'qkv_bias': args.Model.qkv_bias,
    'conv_k': args.Model.conv_k,
    'conv_2d':  args.Model.conv_2d,
    'conv_bias': args.Model.conv_bias,
    'conv_type': args.Model.conv_type,
    'region_attn': args.Model.region_attn,
    'peg_1d': args.Model.peg_1d,}
    num_classes = args.General.num_classes
    device = torch.device(f'cuda:{args.General.device}')
    mil_model = RRT_MIL(**model_params)
    mil_model.to(device)
    
    print('Model Ready!')
    
    optimizer,base_lr = get_optimizer(args,mil_model)
    scheduler,warmup_scheduler = get_scheduler(args,optimizer,base_lr)
    criterion = get_criterion(args.Model.criterion)
    warmup_epoch = args.Model.scheduler.warmup
    
    '''
    start training
    '''
    epoch_info_log = init_epoch_info_log()
    best_model_metric = args.General.best_model_metric
    REVERSE = False
    best_val_metric = 0
    if best_model_metric == 'val_loss':
        REVERSE = True
        best_val_metric = 999
    best_model_metric = best_model_metric.replace('val_','')
    best_epoch = 1
    print('Start Process!')
    for epoch in tqdm(range(args.General.num_epochs),colour='GREEN'):
        if epoch+1 <= warmup_epoch:
            now_scheduler = warmup_scheduler
        else:
            now_scheduler = scheduler
        train_loss,cost_time = train_loop(device,mil_model,train_dataloader,criterion,optimizer,now_scheduler)
        val_loss,val_metrics = val_loop(device,num_classes,mil_model,val_dataloader,criterion)
        if args.Dataset.VIST == True:
            test_loss,test_metrics = val_loss,val_metrics
        else:
            test_loss,test_metrics = val_loop(device,num_classes,mil_model,test_dataloader,criterion)
        print(f'EPOCH:{epoch+1},  Train_Loss:{train_loss},  Val_Loss:{val_loss},  Test_Loss:{test_loss},  Cost_Time:{cost_time}')
        print(f'Val_Metrics:{val_metrics}')
        print(f'Test_Metrics:{test_metrics}')
        add_epoch_info_log(epoch_info_log,epoch,train_loss,val_loss,test_loss,val_metrics,test_metrics)
        
        if REVERSE and val_metrics[best_model_metric] < best_val_metric:
            best_epoch = epoch+1
            best_val_metric = val_metrics[best_model_metric]
            save_best_model(args,mil_model,best_epoch)

        elif not REVERSE and val_metrics[best_model_metric] > best_val_metric:
            best_epoch = epoch+1
            best_val_metric = val_metrics[best_model_metric]
            save_best_model(args,mil_model,best_epoch)

        '''
        early stop
        '''
        is_stop = cal_is_stopping(args,epoch_info_log)
        if is_stop:
            print(f'Early Stop In EPOCH {epoch+1}!')
            save_last_model(args,mil_model,epoch+1)
            save_log(args,epoch_info_log,best_epoch)
            break
        if epoch+1 == args.General.num_epochs:
            save_last_model(args,mil_model,epoch+1)
            save_log(args,epoch_info_log,best_epoch)
