import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.CAR_MIL.car_mil import CAR_MIL, car_mil_loss
from utils.general_utils import (
    add_epoch_info_log,
    early_stop,
    init_epoch_info_log,
    set_global_seed,
)
from utils.loop_utils import val_loop
from utils.model_utils import (
    get_criterion,
    get_optimizer,
    get_scheduler,
    model_select,
    save_last_model,
    save_log,
)
from utils.process_utils import get_process_pipeline
from utils.wsi_utils import WSI_Dataset


def _train_car_mil(
    device,
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    args,
):
    start = time.time()
    model.train()
    loss_sum = 0.0
    for bag, label in loader:
        bag = bag.to(device).float()
        label = label.to(device).long()
        optimizer.zero_grad()
        output = model(bag)
        loss = car_mil_loss(
            output,
            label,
            criterion,
            alpha_effect=args.Model.get("alpha_effect", 0.2),
            alpha_attention=args.Model.get("alpha_attention", 0.2),
            attention_loss=args.Model.get("attention_loss", "cosine"),
        )
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    if scheduler is not None:
        scheduler.step()
    return loss_sum / len(loader), time.time() - start


def process_CAR_MIL(args):
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
        sampler = train_dataset.get_balanced_sampler(
            replacement=args.Dataset.balanced_sampler.replacement
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=1,
            num_workers=num_workers,
            generator=generator,
            sampler=sampler,
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
        )
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=num_workers
    )
    print('DataLoader Ready!')

    device = torch.device(f'cuda:{args.General.device}')
    num_classes = args.General.num_classes
    mil_model = CAR_MIL(
        in_dim=args.Model.in_dim,
        hidden_dim=args.Model.get('hidden_dim', 512),
        attention_dim=args.Model.get('attention_dim', 128),
        num_classes=num_classes,
        dropout=args.Model.get('dropout', 0.25),
        act=args.Model.get('act', 'relu'),
        attention_bias=args.Model.get('attention_bias', False),
    )
    mil_model.to(device)
    print('Model Ready!')

    optimizer, base_lr = get_optimizer(args, mil_model)
    scheduler, warmup_scheduler = get_scheduler(args, optimizer, base_lr)
    criterion = get_criterion(args.Model.criterion)
    warmup_epoch = args.Model.scheduler.warmup

    epoch_info_log = init_epoch_info_log()
    best_model_metric = args.General.best_model_metric
    reverse = False
    best_val_metric = 0
    if best_model_metric == 'val_loss':
        reverse = True
        best_val_metric = 9999
    best_epoch = 1
    print('Start Process!')
    print('Using Process Pipeline:', process_pipeline)

    for epoch in tqdm(range(args.General.num_epochs), colour='GREEN'):
        if epoch + 1 <= warmup_epoch:
            now_scheduler = warmup_scheduler
        else:
            now_scheduler = scheduler
        train_loss, cost_time = _train_car_mil(
            device,
            mil_model,
            train_dataloader,
            criterion,
            optimizer,
            now_scheduler,
            args,
        )
        if process_pipeline == 'Train_Val_Test':
            val_loss, val_metrics = val_loop(
                device, num_classes, mil_model, val_dataloader, criterion
            )
            test_loss, test_metrics = val_loop(
                device, num_classes, mil_model, test_dataloader, criterion
            )
        elif process_pipeline == 'Train_Val':
            val_loss, val_metrics = val_loop(
                device, num_classes, mil_model, val_dataloader, criterion
            )
            test_loss, test_metrics = None, None
        elif process_pipeline == 'Train_Test':
            val_loss, val_metrics, test_loss, test_metrics = None, None, None, None
            if epoch + 1 == args.General.num_epochs:
                test_loss, test_metrics = val_loop(
                    device, num_classes, mil_model, test_dataloader, criterion
                )

        print(
            f'EPOCH:{epoch + 1}, Train_Loss:{train_loss}, '
            f'Val_Loss:{val_loss}, Test_Loss:{test_loss}, Cost_Time:{cost_time}'
        )
        print('Val_Metrics:', val_metrics)
        print('Test_Metrics:', test_metrics)
        add_epoch_info_log(
            epoch_info_log,
            epoch,
            train_loss,
            val_loss,
            test_loss,
            val_metrics,
            test_metrics,
        )
        metrics_for_selection = val_metrics
        if val_metrics is not None and best_model_metric == 'val_loss':
            metrics_for_selection = dict(val_metrics)
            metrics_for_selection['val_loss'] = val_loss
        best_val_metric, best_epoch = model_select(
            reverse,
            args,
            mil_model.state_dict(),
            metrics_for_selection,
            best_model_metric,
            best_val_metric,
            epoch,
            best_epoch,
        )
        if early_stop(
            args,
            epoch_info_log,
            process_pipeline,
            epoch,
            mil_model.state_dict(),
            best_epoch,
        ):
            break
        if epoch + 1 == args.General.num_epochs:
            save_last_model(args, mil_model.state_dict(), epoch + 1)
            save_log(args, epoch_info_log, best_epoch, process_pipeline)
