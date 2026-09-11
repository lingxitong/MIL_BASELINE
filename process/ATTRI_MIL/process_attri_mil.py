import time

import h5py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.ATTRI_MIL.attri_mil import (
    ATTRI_MIL,
    AttributeMemory,
    rank_constraint,
    spatial_constraint,
)
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


def _load_feature_file(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def _nearest_from_coords(coords, patch_size):
    coords = torch.as_tensor(coords).long()
    coordinate_to_index = {
        (int(coord[0]), int(coord[1])): index
        for index, coord in enumerate(coords)
    }
    offsets = (
        (0, 0),
        (0, -patch_size),
        (0, patch_size),
        (-patch_size, 0),
        (patch_size, 0),
        (-patch_size, -patch_size),
        (patch_size, -patch_size),
        (-patch_size, patch_size),
        (patch_size, patch_size),
    )
    nearest = []
    for index, coord in enumerate(coords):
        x, y = int(coord[0]), int(coord[1])
        nearest.append(
            [coordinate_to_index.get((x + dx, y + dy), index) for dx, dy in offsets]
        )
    return torch.tensor(nearest, dtype=torch.long)


class AttriWSIDataset(WSI_Dataset):
    def __init__(self, dataset_info_csv_path, group, patch_size=256):
        super().__init__(dataset_info_csv_path, group)
        self.patch_size = patch_size

    def __getitem__(self, idx):
        path = self.slide_path_list[idx]
        label = torch.tensor(int(self.labels_list[idx]))
        coords = nearest = None
        if path.endswith(".h5"):
            with h5py.File(path, "r") as h5_file:
                features = torch.from_numpy(h5_file["features"][:])
                if "nearest" in h5_file:
                    nearest = torch.from_numpy(h5_file["nearest"][:]).long()
                elif "coords" in h5_file:
                    coords = h5_file["coords"][:]
        else:
            loaded = _load_feature_file(path)
            if isinstance(loaded, dict):
                features = loaded.get("feats", loaded.get("features"))
                if features is None:
                    raise ValueError(f"Unknown feature dict keys in {path}")
                coords = loaded.get("coords")
                nearest = loaded.get("nearest")
            else:
                features = loaded
        features = torch.as_tensor(features)
        if features.dim() == 3:
            features = features.squeeze(0)
        if nearest is None and coords is not None:
            nearest = _nearest_from_coords(coords, self.patch_size)
        if nearest is None:
            nearest = torch.empty((0, 0), dtype=torch.long)
        return features, label, torch.as_tensor(nearest).long()


def _train_attri_mil(
    device,
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    num_classes,
    spatial_weight,
    rank_weight,
    queue_size,
):
    start = time.time()
    model.train()
    loss_sum = 0.0
    memory = AttributeMemory(num_classes, queue_size=queue_size)
    for bag, label, nearest in loader:
        bag = bag.to(device).float()
        label = label.to(device).long()
        nearest = nearest.to(device)
        if spatial_weight > 0 and nearest.numel() == 0:
            raise ValueError(
                "AttriMIL spatial_weight > 0 requires coords/nearest in every feature file"
            )
        optimizer.zero_grad()
        output = model(bag)
        bag_loss = criterion(output["logits"], label)
        spatial_loss = spatial_constraint(
            output["attribute_scores"], nearest, num_classes
        )
        rank_loss = rank_constraint(
            bag,
            label,
            model,
            output["attribute_scores"],
            memory,
            num_classes,
        )
        loss = bag_loss + spatial_weight * spatial_loss + rank_weight * rank_loss
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    if scheduler is not None:
        scheduler.step()
    return loss_sum / len(loader), time.time() - start


def process_ATTRI_MIL(args):
    patch_size = args.Model.get("patch_size", 256)
    train_dataset = AttriWSIDataset(
        args.Dataset.dataset_csv_path, "train", patch_size
    )
    val_dataset = AttriWSIDataset(
        args.Dataset.dataset_csv_path, "val", patch_size
    )
    test_dataset = AttriWSIDataset(
        args.Dataset.dataset_csv_path, "test", patch_size
    )
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
    model = ATTRI_MIL(
        in_dim=args.Model.in_dim,
        num_classes=args.General.num_classes,
        attention_dim=args.Model.get('attention_dim', None),
        dropout=args.Model.get('dropout', 0.0),
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
        train_loss, cost_time = _train_attri_mil(
            device,
            model,
            train_loader,
            criterion,
            optimizer,
            now_scheduler,
            args.General.num_classes,
            args.Model.spatial_weight,
            args.Model.rank_weight,
            args.Model.queue_size,
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
