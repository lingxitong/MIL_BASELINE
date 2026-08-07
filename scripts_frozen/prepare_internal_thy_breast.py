#!/usr/bin/env python3
"""Build MIL_BASELINE CSVs + 5:2:3 splits for internal thyroid/breast (full patches only)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/home/oymx/work/frozen/project/MIL_BASELINE")
TASK = Path("/mnt/sdb/cyx/FROZEN/SYSFL/2024_3_4_2_class_task_split_5_2_3")
FEAT_DIR = Path("/mnt/net_sda/oymx/frozen_distill/eval_uni2/pt_files")
OUT_DS = Path("/mnt/net_sda/oymx/frozen_distill/mil_uni2/datasets")
OUT_SPLIT = Path("/mnt/net_sda/oymx/frozen_distill/mil_uni2/splits")
LABEL = {"cancer": 0, "no_cancer": 1}  # binary


def build_raw_csv(organ: str) -> Path:
    obj = json.loads((TASK / organ / "dino_lt_988w.json").read_text())
    rows = []
    for split in ("train", "valid", "test"):
        for it in obj[split]:
            sid = Path(it["patch_dir"]).name
            rows.append(
                {
                    "slide_path": str(FEAT_DIR / f"{sid}.pt"),
                    "label": LABEL[it["label"]],
                    "slide_id": sid,
                    "patch_dir": it["patch_dir"],
                    "paper_split": split,
                    "raw_label": it["label"],
                }
            )
    OUT_DS.mkdir(parents=True, exist_ok=True)
    # MIL_BASELINE raw csv only needs slide_path,label
    raw = pd.DataFrame(rows)
    mil_csv = OUT_DS / f"internal_{organ}_uni2.csv"
    raw[["slide_path", "label"]].to_csv(mil_csv, index=False)
    meta_csv = OUT_DS / f"internal_{organ}_uni2_meta.csv"
    raw.to_csv(meta_csv, index=False)
    print(f"[raw] {organ}: n={len(raw)} -> {mil_csv}")
    print(raw["label"].value_counts().to_dict(), "paper_split", raw["paper_split"].value_counts().to_dict())
    return mil_csv


def run_split(raw_csv: Path, organ: str, seed: int = 42) -> Path:
    OUT_SPLIT.mkdir(parents=True, exist_ok=True)
    save = OUT_SPLIT / f"internal_{organ}_uni2_split_5_2_3_seed{seed}.csv"
    cmd = [
        sys.executable,
        str(ROOT / "split_scripts/split_datasets_user_define_train_val_test.py"),
        "--seed",
        str(seed),
        "--csv_path",
        str(raw_csv),
        "--save_path",
        str(save),
        "--train_ratio",
        "0.5",
        "--val_ratio",
        "0.2",
        "--test_ratio",
        "0.3",
    ]
    print("[split]", " ".join(cmd))
    subprocess.check_call(cmd)
    df = pd.read_csv(save)
    print(
        f"[split] {organ}: train={df['train_slide_path'].dropna().shape[0]} "
        f"val={df['val_slide_path'].dropna().shape[0]} "
        f"test={df['test_slide_path'].dropna().shape[0]} -> {save}"
    )
    return save


def write_yaml(organ: str, split_csv: Path, device: int = 4) -> Path:
    cfg_dir = Path("/mnt/net_sda/oymx/frozen_distill/mil_uni2/configs")
    log_dir = Path("/mnt/net_sda/oymx/frozen_distill/mil_uni2/logs")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    # breast is highly imbalanced -> balanced sampler on
    balanced = organ == "breast"
    text = f"""General:
    MODEL_NAME: AB_MIL
    seed: 42
    num_classes: 2
    num_epochs: 40
    device: {device}
    num_workers: 4
    best_model_metric: macro_auc
    earlystop:
        use: True
        patience: 10
        metric: macro_auc

Dataset:
    DATASET_NAME: internal_{organ}_uni2
    dataset_csv_path: {split_csv}
    balanced_sampler:
        use: {str(balanced)}
        replacement: True

Logs:
    log_root_dir: {log_dir}

Model:
    in_dim: 1536
    L: 512
    D: 128
    dropout: 0.1
    act: relu
    optimizer:
        which: adamw
        adam_config:
            lr: 0.0002
            weight_decay: 0.00001
        adamw_config:
            lr: 0.0002
            weight_decay: 0.00001
    criterion: ce
    scheduler:
        warmup: 2
        which: cosine
        step_config:
            step_size: 10
            gamma: 0.9
        multi_step_config:
            milestones: [20, 30]
            gamma: 0.9
        exponential_config:
            gamma: 0.9
        cosine_config:
            T_max: 20
            eta_min: 0.00001
"""
    out = cfg_dir / f"AB_MIL_internal_{organ}_uni2.yaml"
    out.write_text(text)
    print(f"[yaml] {out}")
    return out


def main():
    for organ in ("thyroid", "breast"):
        raw = build_raw_csv(organ)
        split = run_split(raw, organ, seed=42)
        write_yaml(organ, split, device=0)  # CUDA_VISIBLE_DEVICES will remap


if __name__ == "__main__":
    main()
