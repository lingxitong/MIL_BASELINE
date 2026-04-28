import torch
import numpy as np
from torch.utils.data import DataLoader
from modules.DAG_MIL.dag_mil import DeformableGraphGNN
from utils.process_utils import get_process_pipeline
from utils.wsi_utils import LONG_MIL_WSI_Dataset
from utils.general_utils import set_global_seed, init_epoch_info_log, add_epoch_info_log, early_stop
from utils.model_utils import get_optimizer, get_scheduler, get_criterion, save_last_model, save_log, model_select
from utils.loop_utils import cal_scores
from tqdm import tqdm


def train_loop_DAG_MIL(device, model, loader, criterion, optimizer, scheduler):
    model.train()
    train_loss_log = 0.0

    for data, label in loader:
        optimizer.zero_grad()
        data = data.to(device).float()
        label = label.long().to(device)

        coords = data[:, :, -2:]
        bag = data[:, :, :-2]
        forward_return = model(bag, coords)
        train_logits = forward_return['logits']
        train_loss = criterion(train_logits, label)
        train_loss_log += train_loss.item()
        train_loss.backward()
        optimizer.step()

    if scheduler is not None:
        scheduler.step()

    train_loss_log /= len(loader)
    return train_loss_log


def val_loop_DAG_MIL(device, num_classes, model, loader, criterion, return_WSI_feature=False):
    model.eval()
    val_loss_log = 0.0
    labels = []
    bag_predictions_after_normal = []
    wsi_features = []

    with torch.no_grad():
        for data, label in loader:
            data = data.to(device).float()
            label = label.long().to(device)
            coords = data[:, :, -2:]
            bag = data[:, :, :-2]

            if return_WSI_feature:
                wsi_feature = model(bag, coords, return_WSI_feature=True)['WSI_feature']
                wsi_features.append(wsi_feature)
                continue

            forward_return = model(bag, coords)
            val_logits = forward_return['logits']
            labels.append(label.cpu().numpy())
            bag_predictions_after_normal.append(torch.softmax(val_logits.squeeze(0), dim=0).cpu().numpy())
            val_loss = criterion(val_logits, label)
            val_loss_log += val_loss.item()

    if return_WSI_feature:
        wsi_features = torch.cat(wsi_features, dim=0).cpu().numpy()
        return wsi_features

    val_loss_log /= len(loader)
    val_metrics = cal_scores(bag_predictions_after_normal, labels, num_classes)
    return val_loss_log, val_metrics


def process_DAG_MIL(args):
    h5_csv_path = args.Dataset.h5_csv_path

    train_dataset = LONG_MIL_WSI_Dataset(args.Dataset.dataset_csv_path, h5_csv_path, 'train')
    val_dataset = LONG_MIL_WSI_Dataset(args.Dataset.dataset_csv_path, h5_csv_path, 'val')
    test_dataset = LONG_MIL_WSI_Dataset(args.Dataset.dataset_csv_path, h5_csv_path, 'test')
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
    mil_model = DeformableGraphGNN(
        dim_in=args.Model.in_dim,
        dim_hidden=args.Model.dim_hidden,
        n_classes=num_classes,
        topk=args.Model.topk,
        stride=args.Model.stride,
        agg_type=args.Model.agg_type,
        dropout=args.Model.dropout
    )
    mil_model.to(device)

    print('Model Ready!')

    optimizer, base_lr = get_optimizer(args, mil_model)
    scheduler, warmup_scheduler = get_scheduler(args, optimizer, base_lr)
    criterion = get_criterion(args.Model.criterion)
    warmup_epoch = args.Model.scheduler.warmup

    epoch_info_log = init_epoch_info_log()
    best_model_metric = args.General.best_model_metric
    reverse = best_model_metric == 'val_loss'
    best_val_metric = 9999 if reverse else 0
    best_epoch = 1
    print('Start Process!')
    print('Using Process Pipeline:', process_pipeline)

    for epoch in tqdm(range(args.General.num_epochs), colour='GREEN'):
        now_scheduler = warmup_scheduler if epoch + 1 <= warmup_epoch else scheduler
        train_loss = train_loop_DAG_MIL(device, mil_model, train_dataloader, criterion, optimizer, now_scheduler)

        if process_pipeline == 'Train_Val_Test':
            val_loss, val_metrics = val_loop_DAG_MIL(device, num_classes, mil_model, val_dataloader, criterion)
            test_loss, test_metrics = val_loop_DAG_MIL(device, num_classes, mil_model, test_dataloader, criterion)
        elif process_pipeline == 'Train_Val':
            val_loss, val_metrics = val_loop_DAG_MIL(device, num_classes, mil_model, val_dataloader, criterion)
            test_loss, test_metrics = None, None
        elif process_pipeline == 'Train_Test':
            val_loss, val_metrics, test_loss, test_metrics = None, None, None, None
            if epoch + 1 == args.General.num_epochs:
                test_loss, test_metrics = val_loop_DAG_MIL(device, num_classes, mil_model, test_dataloader, criterion)

        fail = '\033[91m'
        endc = '\033[0m'
        print('----------------INFO----------------\n')
        print(f'{fail}EPOCH:{endc}{epoch + 1},  Train_Loss:{train_loss},  Val_Loss:{val_loss},  Test_Loss:{test_loss}\n')
        print(f'{fail}Val_Metrics:  {endc}{val_metrics}\n')
        print(f'{fail}Test_Metrics:  {endc}{test_metrics}\n')
        add_epoch_info_log(epoch_info_log, epoch, train_loss, val_loss, test_loss, val_metrics, test_metrics)

        best_val_metric, best_epoch = model_select(reverse, args, mil_model.state_dict(), val_metrics, best_model_metric, best_val_metric, epoch, best_epoch)

        if early_stop(args, epoch_info_log, process_pipeline, epoch, mil_model.state_dict(), best_epoch):
            break

        if epoch + 1 == args.General.num_epochs:
            save_last_model(args, mil_model.state_dict(), epoch + 1)
            save_log(args, epoch_info_log, best_epoch, process_pipeline)
