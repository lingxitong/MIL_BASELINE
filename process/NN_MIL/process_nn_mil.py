"""Classification training pipeline for nnMIL in MIL_BASELINE."""

from functools import partial

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.NN_MIL.nn_mil import NN_MIL
from utils.general_utils import add_epoch_info_log, early_stop, init_epoch_info_log, set_global_seed
from utils.model_utils import get_criterion, get_optimizer, get_scheduler, model_select, save_last_model, save_log
from utils.nnmil_utils import BalancedBatchSampler, fixed_bag_collate, nnmil_train_loop, nnmil_val_loop, resolve_fixed_bag_size
from utils.process_utils import get_process_pipeline
from utils.wsi_utils import WSI_Dataset


def process_NN_MIL(args):
    train_dataset = WSI_Dataset(args.Dataset.dataset_csv_path, "train")
    val_dataset = WSI_Dataset(args.Dataset.dataset_csv_path, "val")
    test_dataset = WSI_Dataset(args.Dataset.dataset_csv_path, "test")
    process_pipeline = get_process_pipeline(val_dataset, test_dataset)
    args.General.process_pipeline = process_pipeline
    set_global_seed(args.General.seed)

    model_config = args.Model
    fixed_bag_size, observed_lengths = resolve_fixed_bag_size(
        train_dataset, getattr(model_config, "fixed_bag_size", "auto"),
        getattr(model_config, "auto_bag_size_factor", 0.5),
    )
    if observed_lengths is not None:
        print(f"NN_MIL resolved fixed_bag_size={fixed_bag_size} from train median={torch.tensor(observed_lengths).median().item():.0f}")
    else:
        print(f"NN_MIL using configured fixed_bag_size={fixed_bag_size}")

    batch_size = int(getattr(model_config, "batch_size", 32))
    collate_fn = partial(fixed_bag_collate, bag_size=fixed_bag_size)
    balanced = bool(getattr(model_config, "task_aware_sampler", True))
    if balanced:
        sampler = BalancedBatchSampler(train_dataset.labels_list, batch_size, seed=args.General.seed)
        train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=args.General.num_workers, collate_fn=collate_fn)
        print("NN_MIL training with class-balanced mini-batches")
    else:
        generator = torch.Generator().manual_seed(args.General.seed)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator,
                                  num_workers=args.General.num_workers, collate_fn=collate_fn)
        print("NN_MIL training with randomly shuffled mini-batches")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.General.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.General.num_workers)

    device = torch.device(f"cuda:{args.General.device}" if torch.cuda.is_available() else "cpu")
    model = NN_MIL(
        in_dim=model_config.in_dim, hidden_dim=getattr(model_config, "hidden_dim", 256),
        num_classes=args.General.num_classes, dropout=getattr(model_config, "dropout", 0.25),
        activation=getattr(model_config, "activation", "softmax"),
        feature_select=getattr(model_config, "feature_select", True),
        eval_stride_divisor=getattr(model_config, "eval_stride_divisor", 4),
        cover_shuffle=getattr(model_config, "cover_shuffle", True),
        cover_seed=getattr(model_config, "cover_seed", 42),
    ).to(device)
    print(f"NN_MIL model ready ({sum(parameter.numel() for parameter in model.parameters()):,} parameters) on {device}")
    optimizer, base_lr = get_optimizer(args, model)
    scheduler, warmup_scheduler = get_scheduler(args, optimizer, base_lr)
    criterion = get_criterion(model_config.criterion)
    warmup_epoch = model_config.scheduler.warmup

    epoch_info_log = init_epoch_info_log()
    best_metric = 9999 if args.General.best_model_metric == "val_loss" else 0
    reverse = args.General.best_model_metric == "val_loss"
    best_epoch = 1
    for epoch in tqdm(range(args.General.num_epochs), colour="GREEN"):
        active_scheduler = warmup_scheduler if epoch + 1 <= warmup_epoch else scheduler
        train_loss = nnmil_train_loop(device, model, train_loader, criterion, optimizer, active_scheduler)
        val_loss = val_metrics = test_loss = test_metrics = None
        if process_pipeline in ["Train_Val_Test", "Train_Val"]:
            val_loss, val_metrics = nnmil_val_loop(device, args.General.num_classes, model, val_loader, criterion)
        if process_pipeline == "Train_Val_Test":
            test_loss, test_metrics = nnmil_val_loop(device, args.General.num_classes, model, test_loader, criterion)
        elif process_pipeline == "Train_Test" and epoch + 1 == args.General.num_epochs:
            test_loss, test_metrics = nnmil_val_loop(device, args.General.num_classes, model, test_loader, criterion)
        print(f"NN_MIL epoch {epoch + 1}: train_loss={train_loss:.6f}, val_loss={val_loss}, test_loss={test_loss}")
        add_epoch_info_log(epoch_info_log, epoch, train_loss, val_loss, test_loss, val_metrics, test_metrics)
        best_metric, best_epoch = model_select(reverse, args, model.state_dict(), val_metrics,
                                                args.General.best_model_metric, best_metric, epoch, best_epoch)
        if early_stop(args, epoch_info_log, process_pipeline, epoch, model.state_dict(), best_epoch):
            break
        if epoch + 1 == args.General.num_epochs:
            save_last_model(args, model.state_dict(), epoch + 1)
            save_log(args, epoch_info_log, best_epoch, process_pipeline)
