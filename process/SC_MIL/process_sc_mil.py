"""
Process script for SC_MIL
SC_MIL requires coords input, similar to LONG_MIL
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from modules.SC_MIL.sc_mil import SC_MIL
from utils.process_utils import get_process_pipeline, get_act
from utils.wsi_utils import LONG_MIL_WSI_Dataset, SC_MIL_WSI_Dataset  # SC_MIL dataset with dummy coords support
from utils.general_utils import set_global_seed, init_epoch_info_log, add_epoch_info_log, early_stop
from utils.model_utils import get_optimizer, get_scheduler, get_criterion, save_last_model, save_log, model_select
from utils.loop_utils import train_loop, val_loop, cal_scores
from tqdm import tqdm

def process_SC_MIL(args):
    """
    Training process for SC_MIL
    SC_MIL can work with or without h5 files:
    - If h5_csv_path is provided and exists, uses coords from h5 files
    - Otherwise, generates dummy coords based on patch indices
    """
    h5_csv_path = getattr(args.Dataset, 'h5_csv_path', None)
    use_dummy_coords = getattr(args.Dataset, 'use_dummy_coords', True)
    
    # Use SC_MIL_WSI_Dataset which supports dummy coords
    train_dataset = SC_MIL_WSI_Dataset(args.Dataset.dataset_csv_path, h5_csv_path, 'train', use_dummy_coords=use_dummy_coords)
    val_dataset = SC_MIL_WSI_Dataset(args.Dataset.dataset_csv_path, h5_csv_path, 'val', use_dummy_coords=use_dummy_coords)
    test_dataset = SC_MIL_WSI_Dataset(args.Dataset.dataset_csv_path, h5_csv_path, 'test', use_dummy_coords=use_dummy_coords)
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
    hidden_size = args.Model.hidden_size if hasattr(args.Model, 'hidden_size') else in_dim
    deep = args.Model.deep if hasattr(args.Model, 'deep') else 1
    n_cluster = args.Model.n_cluster if hasattr(args.Model, 'n_cluster') else None
    cluster_size = args.Model.cluster_size if hasattr(args.Model, 'cluster_size') else None
    feature_weight = args.Model.feature_weight if hasattr(args.Model, 'feature_weight') else 0
    dropout = args.Model.dropout
    act = get_act(args.Model.act)
    with_softfilter = args.Model.with_softfilter if hasattr(args.Model, 'with_softfilter') else False
    use_filter_branch = args.Model.use_filter_branch if hasattr(args.Model, 'use_filter_branch') else False
    with_cssa = args.Model.with_cssa if hasattr(args.Model, 'with_cssa') else True
    
    mil_model = SC_MIL(
        in_dim=in_dim,
        num_classes=num_classes,
        hidden_size=hidden_size,
        deep=deep,
        n_cluster=n_cluster,
        cluster_size=cluster_size,
        feature_weight=feature_weight,
        dropout=dropout,
        with_softfilter=with_softfilter,
        use_filter_branch=use_filter_branch,
        with_cssa=with_cssa,
        act=act
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
        
        # Custom train loop for SC_MIL (needs coords)
        train_loss_total = 0.0
        train_count = 0
        mil_model.train()
        for batch_idx, (data, label) in enumerate(train_dataloader):
            data = data.to(device)
            label = label.to(device)
            
            # Extract features and coords from concatenated input
            # LONG_MIL_WSI_Dataset returns (N, D+2) where last 2 dims are coords
            coords = data[:, :, -2:]  # (1, N, 2)
            features = data[:, :, :-2]  # (1, N, D)
            
            optimizer.zero_grad()
            output = mil_model(features, coords=coords)
            logits = output['logits']
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            if now_scheduler is not None:
                now_scheduler.step()
            
            train_loss_total += loss.item()
            train_count += 1
        
        train_loss = train_loss_total / train_count if train_count > 0 else 0.0
        
        # Validation and test loops
        if process_pipeline == 'Train_Val_Test':
            val_loss, val_metrics = val_loop_SC_MIL(device, num_classes, mil_model, val_dataloader, criterion)
            test_loss, test_metrics = val_loop_SC_MIL(device, num_classes, mil_model, test_dataloader, criterion)
        elif process_pipeline == 'Train_Val':
            val_loss, val_metrics = val_loop_SC_MIL(device, num_classes, mil_model, val_dataloader, criterion)
            test_loss, test_metrics = None, None
        elif process_pipeline == 'Train_Test':
            val_loss, val_metrics, test_loss, test_metrics = None, None, None, None
            if epoch + 1 == args.General.num_epochs:
                test_loss, test_metrics = val_loop_SC_MIL(device, num_classes, mil_model, test_dataloader, criterion)
        
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        print('----------------INFO----------------\n')
        print(f'{FAIL}EPOCH:{ENDC}{epoch+1},  Train_Loss:{train_loss},  Val_Loss:{val_loss},  Test_Loss:{test_loss}\n')
        print(f'{FAIL}Val_Metrics:  {ENDC}{val_metrics}\n')
        print(f'{FAIL}Test_Metrics:  {ENDC}{test_metrics}\n')
        add_epoch_info_log(epoch_info_log, epoch, train_loss, val_loss, test_loss, val_metrics, test_metrics)
        
        best_val_metric, best_epoch = model_select(REVERSE, args, mil_model.state_dict(), val_metrics, best_model_metric, best_val_metric, epoch, best_epoch)
        
        if early_stop(args, epoch_info_log, process_pipeline, epoch, mil_model.state_dict(), best_epoch):
            break
        
        if epoch + 1 == args.General.num_epochs:
            save_last_model(args, mil_model.state_dict(), epoch + 1)
            save_log(args, epoch_info_log, best_epoch, process_pipeline)

def val_loop_SC_MIL(device, num_classes, model, dataloader, criterion):
    """Validation loop for SC_MIL (handles coords)"""
    model.eval()
    val_loss_total = 0.0
    val_count = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            label = label.to(device)
            
            # Extract features and coords
            coords = data[:, :, -2:]  # (1, N, 2)
            features = data[:, :, :-2]  # (1, N, D)
            
            output = model(features, coords=coords)
            logits = output['logits']
            loss = criterion(logits, label)
            
            val_loss_total += loss.item()
            val_count += 1
            
            # Format predictions for cal_scores: (batch_size, num_classes)
            preds = torch.softmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(label.cpu().numpy())
    
    val_loss = val_loss_total / val_count if val_count > 0 else 0.0
    
    if len(all_preds) > 0:
        # Concatenate predictions and labels
        all_preds = np.concatenate(all_preds, axis=0)  # (batch_size, num_classes)
        all_labels = np.concatenate(all_labels, axis=0)  # (batch_size,)
        # Use cal_scores from loop_utils (same as standard val_loop)
        metrics = cal_scores(all_preds, all_labels, num_classes)
    else:
        metrics = {}
    
    return val_loss, metrics

