import random
import numpy as np
import torch
import pytz
from datetime import datetime
import shutil
import os
def save_dataset_csv(args):
    dataset_csv_path = args.Dataset.dataset_csv_path
    os.makedirs(args.Logs.now_log_dir, exist_ok=True)
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
    if args.Dataset.now_fold != {}:
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
    
def init_epoch_info_log():
    epoch_info_log = {'epoch':[],'train_loss':[],'val_loss':[],'test_loss':[],
                      'val_acc':[],'val_bacc':[],'val_macro_auc':[],'val_micro_auc':[],'val_weighted_auc':[],
                       'val_macro_f1':[],'val_micro_f1':[],'val_weighted_f1':[],
                       'val_macro_recall':[],'val_micro_recall':[],'val_weighted_recall':[],
                       'val_macro_pre':[],'val_micro_pre':[],'val_weighted_pre':[],
                       'val_quadratic_kappa':[],'val_linear_kappa':[],'val_confusion_mat':[],
                       'test_acc':[],'test_bacc':[],'test_macro_auc':[],'test_micro_auc':[],'test_weighted_auc':[],
                       'test_macro_f1':[],'test_micro_f1':[],'test_weighted_f1':[],
                       'test_macro_recall':[],'test_micro_recall':[],'test_weighted_recall':[],
                       'test_macro_pre':[],'test_micro_pre':[],'test_weighted_pre':[],
                       'test_quadratic_kappa':[],'test_linear_kappa':[],'test_confusion_mat':[]}
    return epoch_info_log
    
def add_epoch_info_log(epoch_info_log,epoch,train_loss,val_loss,test_loss,val_metrics,test_metrics):
    epoch_info_log['epoch'].append(epoch+1)
    epoch_info_log['train_loss'].append(train_loss)
    epoch_info_log['val_loss'].append(val_loss)
    epoch_info_log['test_loss'].append(test_loss)
    for key in val_metrics.keys():
        epoch_info_log['val_'+key].append(val_metrics[key])
    for key in test_metrics.keys():
        epoch_info_log['test_'+key].append(test_metrics[key])
def cal_is_stopping(args,epoch_info_log):

    if args.General.earlystop.use == None or args.General.earlystop.use == False:
        return False
    
    patience = int(args.General.earlystop.patience)
    print('patience:',patience)
    judge_metric = args.General.earlystop.metric
    
    if epoch_info_log['epoch'][-1] <= patience:
        return False
    if 'val' not in judge_metric:
        judge_metric = 'val_'+judge_metric
    judge_metric_list = epoch_info_log[judge_metric]
    if judge_metric == 'val_loss':
        judge_metric_list = -np.array(judge_metric_list)
    last_epoch = epoch_info_log['epoch'][-1]
    for i in range(last_epoch,last_epoch-patience,-1):
        if judge_metric_list[last_epoch-patience-1] <= judge_metric_list[i-1]:
            return False
    return True
