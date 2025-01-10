import torch
from torch.utils.data import DataLoader
from modules.DTFD_MIL.dtfd_mil import Classifier_1fc, Attention, DimReduction, Attention_with_Classifier
from utils.process_utils import get_process_pipeline,get_act
from utils.wsi_utils import WSI_Dataset
from utils.general_utils import set_global_seed,init_epoch_info_log,add_epoch_info_log,early_stop
from utils.model_utils import get_optimizer,get_scheduler,get_criterion,save_last_model,save_log,model_select,get_param_optimizer
from utils.loop_utils import dtfd_train_loop,dtfd_val_loop
from tqdm import tqdm
    
def process_DTFD_MIL(args):

    train_dataset = WSI_Dataset(args.Dataset.dataset_csv_path,'train')
    val_dataset = WSI_Dataset(args.Dataset.dataset_csv_path,'val')
    test_dataset = WSI_Dataset(args.Dataset.dataset_csv_path,'test')
    process_pipeline = get_process_pipeline(val_dataset,test_dataset) 
    args.General.process_pipeline = process_pipeline
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
    
    device = torch.device(f'cuda:{args.General.device}')
    num_classes = args.General.num_classes
    in_dim = args.Model.in_dim
    classifier_dropout = args.Model.classifier_dropout
    attCls_dropout = args.Model.attCls_dropout
    act = args.Model.act
    total_instance = args.Model.total_instance
    num_Group = args.Model.num_Group
    grad_clipping = args.Model.grad_clipping
    distill = args.Model.distill
    mdim = args.Model.mdim
    numLayer_Res = args.Model.numLayer_Res
    classifier = Classifier_1fc(mdim, num_classes, classifier_dropout).to(device)
    attention = Attention(mdim).to(device)
    dimReduction = DimReduction(in_dim, mdim, numLayer_Res=numLayer_Res).to(device)
    attCls = Attention_with_Classifier(L=mdim, num_cls=num_classes, droprate=attCls_dropout).to(device)
    
    print('Model Ready!')
    
    trainable_parameters_A = []
    trainable_parameters_A += list(classifier.parameters())
    trainable_parameters_A += list(attention.parameters())
    trainable_parameters_A += list(dimReduction.parameters())
    
    trainable_parameters_B = attCls.parameters()
    
    optimizer_A,base_lr = get_param_optimizer(args,trainable_parameters_A)
    scheduler_A,warmup_scheduler_A = get_scheduler(args,optimizer_A,base_lr)
    
    optimizer_B,base_lr = get_param_optimizer(args,trainable_parameters_B)
    scheduler_B,warmup_scheduler_B = get_scheduler(args,optimizer_B,base_lr)
    
    criterion = get_criterion(args.Model.criterion)
    warmup_epoch = args.Model.scheduler.warmup
    
    model_list = [classifier,attention,dimReduction,attCls]
    optimizer_list = [optimizer_A,optimizer_B]
    '''
    begin training
    '''
    epoch_info_log = init_epoch_info_log()
    best_model_metric = args.General.best_model_metric
    REVERSE = False
    best_val_metric = 0
    if best_model_metric == 'val_loss':
        REVERSE = True
        best_val_metric = 9999
    best_epoch = 1
    print('Start Process!')
    print('Using Process Pipeline:',process_pipeline)
    for epoch in tqdm(range(args.General.num_epochs),colour='GREEN'):
        if epoch+1 <= warmup_epoch:
            now_scheduler_A = warmup_scheduler_A
            now_scheduler_B = warmup_scheduler_B
        else:
            now_scheduler_A = scheduler_A
            now_scheduler_B = scheduler_B
        scheduler_list = [now_scheduler_A,now_scheduler_B]
        
        train_loss,cost_time = dtfd_train_loop(device,model_list,train_dataloader,criterion,optimizer_list,scheduler_list,num_Group,grad_clipping,distill,total_instance)
        if process_pipeline == 'Train_Val_Test':
            val_loss,val_metrics = dtfd_val_loop(device,num_classes,model_list,val_dataloader,criterion,num_Group,grad_clipping,distill,total_instance)
            test_loss,test_metrics = dtfd_val_loop(device,num_classes,model_list,test_dataloader,criterion,num_Group,grad_clipping,distill,total_instance)
        elif process_pipeline == 'Train_Val':
            val_loss,val_metrics = dtfd_val_loop(device,num_classes,model_list,val_dataloader,criterion,num_Group,grad_clipping,distill,total_instance)
            test_loss,test_metrics = None,None
        elif process_pipeline == 'Train_Test':
            val_loss,val_metrics,test_loss,test_metrics = None,None,None,None
            if epoch+1 == args.General.num_epochs:
                test_loss,test_metrics = dtfd_val_loop(device,num_classes,model_list,test_dataloader,criterion,num_Group,grad_clipping,distill,total_instance)


        FAIL = '\033[91m'
        ENDC = '\033[0m'
        print('----------------INFO----------------\n')
        print(f'{FAIL}EPOCH:{ENDC}{epoch+1},  Train_Loss:{train_loss},  Val_Loss:{val_loss},  Test_Loss:{test_loss},  Cost_Time:{cost_time}\n')
        print(f'{FAIL}Val_Metrics:  {ENDC}{val_metrics}\n')
        print(f'{FAIL}Test_Metrics:  {ENDC}{test_metrics}\n')
        add_epoch_info_log(epoch_info_log,epoch,train_loss,val_loss,test_loss,val_metrics,test_metrics)
        mil_model_state_dict = {'classifier':classifier.state_dict(),'attention':attention.state_dict(),
                      'dimReduction':dimReduction.state_dict(),'attCls':attCls.state_dict()}
        
        # model selection, it only works when process_pipeline is 'Train_Val_Test' or 'Train_Val'
        best_val_metric,best_epoch = model_select(REVERSE,args,mil_model_state_dict,val_metrics,best_model_metric,best_val_metric,epoch,best_epoch)

        '''
        early stop
        '''
        if early_stop(args,epoch_info_log,process_pipeline,epoch,mil_model_state_dict,best_epoch):
            break

        if epoch+1 == args.General.num_epochs:
            save_last_model(args,mil_model_state_dict,epoch+1)
            save_log(args,epoch_info_log,best_epoch,process_pipeline)



