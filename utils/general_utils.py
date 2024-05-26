import random
import numpy as np
import torch
import pytz
from datetime import datetime
import shutil
import os
def save_dataset_csv(args):
    dataset_csv_path = args.Dataset.dataset_csv_path
    dst_path = os.path.join(args.Logs.now_log_dir,os.path.basename(dataset_csv_path))
    shutil.copyfile(dataset_csv_path,dst_path)
    
def save_yaml(args,yaml_path):
    dst_path = os.path.join(args.Logs.now_log_dir,os.path.basename(yaml_path))
    shutil.copyfile(yaml_path,dst_path)

def print_args(args):
    print(f'Seed Info:{args.General.seed}')
    print(f'Dataset Info:{args.Dataset.DATASET_NAME}')
    print(f'Model Info:{args.General.MODEL_NAME}')
    print(f'Device Info:CUDA:{args.General.device}')
    print(f'Epoch Info:{args.General.num_epochs}')
    print(f'Fold Info:{args.Dataset.now_fold}')
    
def get_time():

    tz = pytz.timezone('Asia/Shanghai')

    now = datetime.now(tz)

    return now.strftime("%Y-%m-%d-%H-%M")
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  
    print('set_global_seed:{}'.format(seed))
    
    
def add_epoch_info_log(epoch_info_log,epoch,train_loss,val_loss,test_loss,val_metrics,test_metrics):
    epoch_info_log['epoch'].append(epoch+1)
    epoch_info_log['train_loss'].append(train_loss)
    epoch_info_log['val_loss'].append(val_loss)
    epoch_info_log['test_loss'].append(test_loss)
    epoch_info_log['val_bacc'].append(val_metrics['bacc'])
    epoch_info_log['val_acc'].append(val_metrics['acc'])
    epoch_info_log['val_auc'].append(val_metrics['auc'])
    epoch_info_log['val_pre'].append(val_metrics['pre'])
    epoch_info_log['val_recall'].append(val_metrics['recall'])
    epoch_info_log['val_f1'].append(val_metrics['f1'])
    epoch_info_log['test_bacc'].append(test_metrics['bacc'])
    epoch_info_log['test_acc'].append(test_metrics['acc'])
    epoch_info_log['test_auc'].append(test_metrics['auc'])
    epoch_info_log['test_pre'].append(test_metrics['pre'])
    epoch_info_log['test_recall'].append(test_metrics['recall'])
    epoch_info_log['test_f1'].append(test_metrics['f1'])

def cal_is_stopping(args,epoch_info_log):

    if args.General.earlystop.use == None or args.General.earlystop.use == False:
        return False
    
    patience = int(args.General.earlystop.patience)
    print('patience:',patience)
    judge_metric = args.General.earlystop.metric
    
    if epoch_info_log['epoch'][-1] <= patience:
        return False
    judge_metric_list = epoch_info_log[judge_metric]
    if judge_metric == 'val_loss':
        judge_metric_list = -np.array(judge_metric_list)
    last_epoch = epoch_info_log['epoch'][-1]
    for i in range(last_epoch,last_epoch-patience,-1):
        if judge_metric_list[last_epoch-patience-1] <= judge_metric_list[i-1]:
            return False
    return True