#!/usr/bin/env python3
"""Paper-style linear probe / ABMIL head on frozen UNI2 bag features."""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset


SEEDS = [1, 406, 528, 2026, 3407]
LABEL_MAP = {"cancer": 0, "no_cancer": 1}


class ABMIL(nn.Module):
    def __init__(self, dim_in=1536, L=512, D=128, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(dim_in, L), nn.ReLU())
        self.attention = nn.Sequential(
            nn.Linear(L, D), nn.Tanh(), nn.Linear(D, 1)
        )
        self.classifier = nn.Linear(L, 2)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):  # x: [N, C]
        h = self.drop(self.fc1(x))
        a = torch.softmax(self.attention(h), dim=0)  # [N,1]
        m = torch.sum(a * h, dim=0, keepdim=True)  # [1,L]
        logits = self.classifier(m)  # [1,2]
        return logits.squeeze(0)


class BagDS(Dataset):
    def __init__(self, items, feat_dir, max_patches=4096, seed=0):
        self.items = items
        self.feat_dir = feat_dir
        self.max_patches = max_patches
        self.seed = seed

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        path = os.path.join(self.feat_dir, f"{it['slide_id']}.pt")
        feats = torch.load(path, map_location="cpu", weights_only=True)
        if feats.dtype != torch.float32:
            feats = feats.float()
        if self.max_patches and feats.shape[0] > self.max_patches:
            g = torch.Generator().manual_seed(self.seed + idx)
            sel = torch.randperm(feats.shape[0], generator=g)[: self.max_patches]
            feats = feats[sel]
        y = LABEL_MAP[it["label"]]
        return feats, y, it["slide_id"]


def collate(batch):
    # batch_size=1
    return batch[0]


def run_one(train_items, test_items, feat_dir, seed, device, epochs=15, lr=1e-3, wd=1e-4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_ds = BagDS(train_items, feat_dir, seed=seed)
    test_ds = BagDS(test_items, feat_dir, seed=seed)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate)

    model = ABMIL().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # class weights from train
    counts = np.zeros(2, dtype=np.float64)
    for it in train_items:
        counts[LABEL_MAP[it["label"]]] += 1
    w = counts.sum() / (2 * np.maximum(counts, 1))
    weight = torch.tensor(w, dtype=torch.float32, device=device)

    best_state = None
    best_val = -1.0
    # use 20% of train as val by seed
    idx = list(range(len(train_items)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = max(1, int(0.2 * len(idx)))
    val_ids = set(idx[:n_val])
    tr_ids = [i for i in idx if i not in val_ids]
    # rebuild loaders with subset — simpler: train all epochs, pick last; or track train balacc
    # Paper uses separate valid; we use held-out from train_items
    tr_items = [train_items[i] for i in tr_ids]
    va_items = [train_items[i] for i in val_ids]
    tr_loader = DataLoader(BagDS(tr_items, feat_dir, seed=seed), batch_size=1, shuffle=True, collate_fn=collate)
    va_loader = DataLoader(BagDS(va_items, feat_dir, seed=seed), batch_size=1, shuffle=False, collate_fn=collate)

    for ep in range(epochs):
        model.train()
        for feats, y, _ in tr_loader:
            feats = feats.to(device)
            y = torch.tensor(y, device=device)
            opt.zero_grad()
            logits = model(feats)
            loss = F.cross_entropy(logits.unsqueeze(0), y.unsqueeze(0), weight=weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # val
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for feats, y, _ in va_loader:
                logits = model(feats.to(device))
                pred = int(logits.argmax().item())
                ys.append(y)
                ps.append(pred)
        if ys:
            bacc = balanced_accuracy_score(ys, ps)
            if bacc >= best_val:
                best_val = bacc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for feats, y, _ in test_loader:
            logits = model(feats.to(device))
            pred = int(logits.argmax().item())
            ys.append(y)
            ps.append(pred)
    bacc = float(balanced_accuracy_score(ys, ps))
    wf1 = float(f1_score(ys, ps, average="weighted"))
    return {"balanced_accuracy": bacc, "weighted_f1": wf1, "n_test": len(ys), "best_val_bacc": best_val}


def load_manifest_groups(manifest_path, feat_dir):
    man = json.loads(Path(manifest_path).read_text())
    # only keep slides with features present
    ok = []
    for s in man["slides"]:
        pt = os.path.join(feat_dir, f"{s['slide_id']}.pt")
        if os.path.isfile(pt) and os.path.getsize(pt) > 0:
            ok.append(s)
    by = defaultdict(list)
    for s in ok:
        by[s["dataset"]].append(s)
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/mnt/net_sda/oymx/frozen_distill/eval_uni2/manifests/uni2_extract_manifest.json")
    ap.add_argument(
        "--feat_dir",
        default="/mnt/net_sda/oymx/frozen_distill/eval_uni2/pt_files",
    )
    ap.add_argument("--out_dir", default="/mnt/net_sda/oymx/frozen_distill/eval_uni2/results")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    by = load_manifest_groups(args.manifest, args.feat_dir)
    print("available datasets:", {k: len(v) for k, v in by.items()}, flush=True)

    tasks = []
    # 1) internal brain test: train on brain tv, eval on brain test
    if by.get("internal_brain_tv") and by.get("internal_brain_test"):
        tasks.append(
            (
                "internal_brain_test",
                by["internal_brain_tv"],
                by["internal_brain_test"],
                {"FrozenPath_paper_site_bacc": 0.6266},
            )
        )
    # 2/3) external breast: train on internal breast tv
    if by.get("internal_breast_tv"):
        if by.get("gyey_breast"):
            tasks.append(
                (
                    "gyey_breast",
                    by["internal_breast_tv"],
                    by["gyey_breast"],
                    {"FrozenPath_paper_site_bacc": 0.6683},
                )
            )
        if by.get("gfph_breast"):
            tasks.append(
                (
                    "gfph_breast",
                    by["internal_breast_tv"],
                    by["gfph_breast"],
                    {"FrozenPath_paper_site_bacc": 0.5883},
                )
            )

    all_results = {}
    for name, tr, te, base in tasks:
        print(f"\n=== {name} train={len(tr)} test={len(te)} ===", flush=True)
        seed_metrics = []
        for seed in SEEDS:
            m = run_one(tr, te, args.feat_dir, seed, device, epochs=args.epochs)
            m["seed"] = seed
            seed_metrics.append(m)
            print(f"  seed={seed} bacc={m['balanced_accuracy']:.4f} wf1={m['weighted_f1']:.4f}", flush=True)
        baccs = [m["balanced_accuracy"] for m in seed_metrics]
        wf1s = [m["weighted_f1"] for m in seed_metrics]
        summary = {
            "task": name,
            "n_train": len(tr),
            "n_test": len(te),
            "balanced_accuracy_mean": float(np.mean(baccs)),
            "balanced_accuracy_std": float(np.std(baccs, ddof=1)) if len(baccs) > 1 else 0.0,
            "weighted_f1_mean": float(np.mean(wf1s)),
            "weighted_f1_std": float(np.std(wf1s, ddof=1)) if len(wf1s) > 1 else 0.0,
            "seeds": seed_metrics,
            "baselines": base,
            "protocol": "ABMIL linear-probe head, freeze UNI2 bags, AdamW lr=1e-3, 15ep, 5 seeds, class-weighted CE",
        }
        all_results[name] = summary
        out_p = Path(args.out_dir) / f"{name}_uni2_abmil.json"
        out_p.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(
            f">> {name}: bacc={summary['balanced_accuracy_mean']:.4f}±{summary['balanced_accuracy_std']:.4f} "
            f"wf1={summary['weighted_f1_mean']:.4f}±{summary['weighted_f1_std']:.4f} "
            f"(FP paper site {base.get('FrozenPath_paper_site_bacc')})",
            flush=True,
        )

    Path(args.out_dir).joinpath("summary_uni2.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False)
    )
    # also mirror into workspace eval_ready
    mirror = Path("/home/oymx/work/frozen/data/eval_ready/uni2_results")
    mirror.mkdir(parents=True, exist_ok=True)
    (mirror / "summary_uni2.json").write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print("wrote", args.out_dir, "and", mirror, flush=True)


if __name__ == "__main__":
    main()
