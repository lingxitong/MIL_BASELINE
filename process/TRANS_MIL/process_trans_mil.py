import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
import argparse 
import os
from modules.TRANS_MIL.trans_mil import *
from utils.wsi_utils import *
from utils.general_utils import *
from utils.model_utils import *
from utils.loop_utils import *
from tqdm import tqdm
    
def process_TRANS_MIL(args):
    print_args(args)

    train_dataset = WSI_Dataset(args.Dataset.dataset_csv_path,'train')
    val_dataset = WSI_Dataset(args.Dataset.dataset_csv_path,'val')
    test_dataset = WSI_Dataset(args.Dataset.dataset_csv_path,'test')
    '''
    generator设置seed用于保证shuffle的一致性
    '''
    
    generator = torch.Generator()
    generator.manual_seed(args.General.seed) 
    set_global_seed(args.General.seed)
    num_workers = args.General.num_workers
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers = num_workers,generator=generator)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    
    print('DataLoader Ready!')
    
    device = torch.device(f'cuda:{args.General.device}')
    mil_model = TRANSMIL(args)
    mil_model.to(device)
    
    print('Model Ready!')
    
    optimizer = get_optimizer(args,mil_model)
    scheduler = get_scheduler(args,optimizer)
    criterion = get_criterion(args.Model.criterion)
    
    '''
    开始循环epoch进行训练
    '''
    epoch_info_log = {'epoch':[],'train_loss':[],'val_loss':[],'test_loss':[],'val_bacc':[],
                      'val_acc':[],'val_auc':[],'val_pre':[],'val_recall':[],'val_f1':[],'test_bacc':[],
                      'test_acc':[],'test_auc':[],'test_pre':[],'test_recall':[],'test_f1':[]}
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
        train_loss,cost_time = train_loop(args,mil_model,train_dataloader,criterion,optimizer,scheduler)
        val_loss,val_metrics = val_loop(args,mil_model,val_dataloader,criterion)
        if args.Dataset.VIST == True:
            test_loss,test_metrics = val_loss,val_metrics
        else:
            test_loss,test_metrics = test_loop(args,mil_model,test_dataloader,criterion)
        print(f'EPOCH:{epoch+1},  Train_Loss:{train_loss},  Val_Loss:{val_loss},  Test_Loss:{test_loss},  Cost_Time:{cost_time}')
        print(f'Val_Metrics:{val_metrics}')
        print(f'Test_Metrics:{test_metrics}')
        add_epoch_info_log(epoch_info_log,epoch,train_loss,val_loss,test_loss,val_metrics,test_metrics)
        
        if REVERSE and val_metrics[best_model_metric] < best_val_metric:
            best_val_metric = val_metrics[best_model_metric]
            save_best_model(args,mil_model,epoch_info_log)
            best_epoch = epoch+1
        elif not REVERSE and val_metrics[best_model_metric] > best_val_metric:
            best_val_metric = val_metrics[best_model_metric]
            save_best_model(args,mil_model,epoch_info_log)
            best_epoch = epoch+1
        '''
        判断是否需要早停
        '''
        is_stop = cal_is_stopping(args,epoch_info_log)
        if is_stop:
            print(f'Early Stop In EPOCH {epoch+1}!')
            save_last_model(args,mil_model,epoch_info_log)
            save_log(args,epoch_info_log,best_epoch)
            break
        if epoch+1 == args.General.num_epochs:
            save_last_model(args,mil_model,epoch_info_log)
            save_log(args,epoch_info_log,best_epoch)


