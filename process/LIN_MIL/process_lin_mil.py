import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.LIN_MIL.lin_mil import LIN_MIL
from utils.general_utils import (
    add_epoch_info_log,
    early_stop,
    init_epoch_info_log,
    set_global_seed,
)
from utils.loop_utils import train_loop, val_loop
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


def process_LIN_MIL(args):
    train_dataset = WSI_Dataset(args.Dataset.dataset_csv_path, "train")
    val_dataset = WSI_Dataset(args.Dataset.dataset_csv_path, "val")
    test_dataset = WSI_Dataset(args.Dataset.dataset_csv_path, "test")
    process_pipeline = get_process_pipeline(val_dataset, test_dataset)
    args.General.process_pipeline = process_pipeline

    generator = torch.Generator().manual_seed(args.General.seed)
    set_global_seed(args.General.seed)
    loader_kwargs = {"batch_size": 1, "num_workers": args.General.num_workers}
    if args.Dataset.balanced_sampler.use:
        sampler = train_dataset.get_balanced_sampler(
            replacement=args.Dataset.balanced_sampler.replacement
        )
        train_loader = DataLoader(
            train_dataset, sampler=sampler, generator=generator, **loader_kwargs
        )
    else:
        train_loader = DataLoader(
            train_dataset, shuffle=True, generator=generator, **loader_kwargs
        )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    print("DataLoader Ready!")

    device = torch.device(f"cuda:{args.General.device}")
    model = LIN_MIL(
        in_dim=args.Model.in_dim,
        num_classes=args.General.num_classes,
        latent_dim=args.Model.get('latent_dim', 512),
        transformer_depth=args.Model.get('transformer_depth', 4),
        dropout=args.Model.get('dropout', 0.0),
        emb_dropout=args.Model.get('emb_dropout', 0.1),
        act=args.Model.get('act', 'ReLU'),
        pooling=args.Model.get('pooling', 'cls'),
        num_heads=args.Model.get('num_heads', 8),
        max_instances=args.Model.get('max_instances', None),
    ).to(device)
    optimizer, base_lr = get_optimizer(args, model)
    scheduler, warmup_scheduler = get_scheduler(args, optimizer, base_lr)
    criterion = get_criterion(args.Model.criterion)
    print("Model Ready!")

    epoch_info_log = init_epoch_info_log()
    best_metric = args.General.best_model_metric
    reverse = best_metric == "val_loss"
    best_value = float("inf") if reverse else float("-inf")
    best_epoch = 1
    print("Start Process!")
    print("Using Process Pipeline:", process_pipeline)
    for epoch in tqdm(range(args.General.num_epochs), colour="GREEN"):
        now_scheduler = (
            warmup_scheduler
            if epoch + 1 <= args.Model.scheduler.warmup
            else scheduler
        )
        train_loss, cost_time = train_loop(
            device,
            model,
            train_loader,
            criterion,
            optimizer,
            now_scheduler,
        )
        val_loss = val_metrics = test_loss = test_metrics = None
        if process_pipeline in ("Train_Val_Test", "Train_Val"):
            val_loss, val_metrics = val_loop(
                device,
                args.General.num_classes,
                model,
                val_loader,
                criterion,
            )
        if process_pipeline == "Train_Val_Test" or (
            process_pipeline == "Train_Test"
            and epoch + 1 == args.General.num_epochs
        ):
            test_loss, test_metrics = val_loop(
                device,
                args.General.num_classes,
                model,
                test_loader,
                criterion,
            )

        print(
            f"EPOCH:{epoch + 1}, Train_Loss:{train_loss}, "
            f"Val_Loss:{val_loss}, Test_Loss:{test_loss}, Cost_Time:{cost_time}"
        )
        print("Val_Metrics:", val_metrics)
        print("Test_Metrics:", test_metrics)
        add_epoch_info_log(
            epoch_info_log,
            epoch,
            train_loss,
            val_loss,
            test_loss,
            val_metrics,
            test_metrics,
        )
        selection_metrics = None if val_metrics is None else dict(val_metrics)
        if selection_metrics is not None:
            selection_metrics["val_loss"] = val_loss
        best_value, best_epoch = model_select(
            reverse,
            args,
            model.state_dict(),
            selection_metrics,
            best_metric,
            best_value,
            epoch,
            best_epoch,
        )
        if early_stop(
            args,
            epoch_info_log,
            process_pipeline,
            epoch,
            model.state_dict(),
            best_epoch,
        ):
            break
        if epoch + 1 == args.General.num_epochs:
            save_last_model(args, model.state_dict(), epoch + 1)
            save_log(args, epoch_info_log, best_epoch, process_pipeline)
