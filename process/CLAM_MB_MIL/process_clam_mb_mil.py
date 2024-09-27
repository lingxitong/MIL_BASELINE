import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
import argparse 
import os
import modules
from modules.CLAM_MB_MIL.clam_mb_mil import *
from utils.wsi_utils import *
from utils.general_utils import *
from utils.model_utils import *
from utils.loop_utils import *
from tqdm import tqdm

    
def process_CLAM_MB_MIL(args):
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
    in_dim = args.Model.in_dim
    subtyping = args.Model.subtyping
    k_sample = args.Model.k_sample
    size_arg = args.Model.size_arg
    dropout = args.Model.dropout
    num_classes = args.General.num_classes
    gate = args.Model.gate
    act = args.Model.act
    instance_eval = args.Model.instance_eval
    device = torch.device(f'cuda:{args.General.device}')
    instance_loss_fn = args.Model.instance_loss_fn
    instance_loss_fn = get_criterion(instance_loss_fn)
    bag_weight = args.Model.bag_weight
    mil_model = CLAM_MB_MIL(gate, size_arg, dropout, k_sample, num_classes, instance_loss_fn, subtyping, in_dim,act,instance_eval)
    mil_model.to(device)
    
    print('Model Ready!')
    
    optimizer,base_lr = get_optimizer(args,mil_model)
    scheduler,warmup_scheduler = get_scheduler(args,optimizer,base_lr)
    criterion = get_criterion(args.Model.criterion)
    warmup_epoch = args.Model.scheduler.warmup
    
    '''
    begin training
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
        train_loss,cost_time = clam_train_loop(device,mil_model,train_dataloader,criterion,optimizer,now_scheduler,bag_weight)
        val_loss,val_metrics = clam_val_loop(device,num_classes,mil_model,val_dataloader,criterion,bag_weight)
        if args.Dataset.VIST == True:
            test_loss,test_metrics = val_loss,val_metrics
        else:
            test_loss,test_metrics = clam_val_loop(device,num_classes,mil_model,test_dataloader,criterion,bag_weight)
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
            save_last_model(args,mil_model,epoch + 1)
            save_log(args,epoch_info_log,best_epoch)
            break
        if epoch+1 == args.General.num_epochs:
            save_last_model(args,mil_model,epoch + 1)
            save_log(args,epoch_info_log,best_epoch)
        






