import torch
import os
import pandas as pd
import glob
import pickle
def get_criterion(criterion):
    if criterion == 'ce':
        return torch.nn.CrossEntropyLoss()


def get_optimizer(args,model):
    opt = args.Model.optimizer.which
    if opt == 'adam':
       lr = args.Model.optimizer.adam_config.lr
       weight_decay = args.Model.optimizer.adam_config.weight_decay
       optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
       return optimizer
    elif opt == 'adamw':
       lr = args.Model.optimizer.adamw_config.lr
       weight_decay = args.Model.optimizer.adamw_config.weight_decay
       optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
       return optimizer
   
def get_scheduler(args,optimizer):
    sch = args.Model.scheduler.which
    if sch == 'step':
        step_size = args.Model.scheduler.step_config.step_size
        gamma = args.Model.scheduler.step_config.gamma
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        return scheduler
    elif sch == 'none':
        return None

def _delete_best_pth_files(now_log_dir):
    pth_files = glob.glob(os.path.join(now_log_dir, '*.pth'))

    for file in pth_files:
        if 'Best' in os.path.basename(file):
            os.remove(file)
    
def _set_round(str_value_list,round_num):
    return [round(float(x),round_num) for x in str_value_list]
def save_best_model(args,model,epoch_info_log):
    t_bacc = epoch_info_log['test_bacc'][-1]
    tacc = epoch_info_log['test_acc'][-1]
    tauc = epoch_info_log['test_auc'][-1]
    tpre = epoch_info_log['test_pre'][-1]
    trc = epoch_info_log['test_recall'][-1]
    tf1 = epoch_info_log['test_f1'][-1]
    best_metric_value = epoch_info_log[args.General.best_model_metric][-1]
    t_bacc,tacc,tauc,tpre,trc,tf1,best_metric_value = _set_round([t_bacc,tacc,tauc,tpre,trc,tf1,best_metric_value],5)
    save_path = os.path.join(args.Logs.now_log_dir,f'Best_EPOCH_{epoch_info_log["epoch"][-1]}_seed_{args.General.seed}_{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}_{args.General.best_model_metric}_{best_metric_value}_t_bacc{t_bacc}_tacc{tacc}_tauc{tauc}_tpre{tpre}_tf1score{tf1}_trc{trc}.pth')
    _delete_best_pth_files(args.Logs.now_log_dir)
    torch.save(model.state_dict(),save_path)
    
def save_last_model(args,model,epoch_info_log):
    t_bacc = epoch_info_log['test_bacc'][-1]
    tacc = epoch_info_log['test_acc'][-1]
    tauc = epoch_info_log['test_auc'][-1]
    tpre = epoch_info_log['test_pre'][-1]
    trc = epoch_info_log['test_recall'][-1]
    tf1 = epoch_info_log['test_f1'][-1]
    last_metric_value = epoch_info_log[args.General.best_model_metric][-1]
    t_bacc,tacc,tauc,tpre,trc,tf1,last_metric_value = _set_round([t_bacc,tacc,tauc,tpre,trc,tf1,last_metric_value],5)
    save_path = os.path.join(args.Logs.now_log_dir,f'Last_EPOCH_{epoch_info_log["epoch"][-1]}_seed_{args.General.seed}_{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}_{args.General.best_model_metric}_{last_metric_value}_tb_acc_{t_bacc}_tacc{tacc}_tauc{tauc}_tpre{tpre}_tf1score{tf1}_trc{trc}.pth')
    torch.save(model.state_dict(),save_path)

def save_log(args,epoch_info_log,best_epoch):
    log_df = pd.DataFrame(epoch_info_log)

    log_df.to_csv(os.path.join(args.Logs.now_log_dir,f'Log_seed{args.General.seed}_{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}.csv'),index=False)
    print('Global Log CSV Saved!')
    best_df = log_df[log_df['epoch'] == best_epoch]
    best_df.to_csv(os.path.join(args.Logs.now_log_dir,f'Best_Log_seed{args.General.seed}_{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}.csv'),index=False)
    print('Best Log CSV Saved!')