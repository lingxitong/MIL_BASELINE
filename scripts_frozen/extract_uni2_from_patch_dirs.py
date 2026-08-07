#!/usr/bin/env python3
"""Extract UNI2-h features from existing patch image directories (MIL_BASELINE uni_v2 recipe)."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def load_uni_v2(weights_dir: str, device: torch.device):
    """Same recipe as MIL_BASELINE feature_extractor/.../model_utils.py uni_v2."""
    uni_v2_config = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    model = timm.create_model(
        model_name="vit_giant_patch14_224", pretrained=False, **uni_v2_config
    )
    ckpt = os.path.join(weights_dir, "pytorch_model.bin")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return model, transform


class PatchDirDataset(Dataset):
    def __init__(self, patch_dir: str, transform, max_patches: int | None, seed: int):
        paths = [
            os.path.join(patch_dir, n)
            for n in sorted(os.listdir(patch_dir))
            if Path(n).suffix in IMG_EXTS
        ]
        if max_patches is not None and len(paths) > max_patches:
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(len(paths), generator=g)[:max_patches].tolist()
            paths = [paths[i] for i in sorted(idx)]
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        p = self.paths[i]
        with Image.open(p) as im:
            im = im.convert("RGB")
            if im.size != (224, 224):
                im = im.resize((224, 224), Image.BILINEAR)
            x = self.transform(im)
        return x


def extract_one(model, transform, patch_dir, device, batch_size, num_workers, max_patches, seed):
    ds = PatchDirDataset(patch_dir, transform, max_patches=max_patches, seed=seed)
    if len(ds) == 0:
        return None
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    feats = []
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            f = model(batch)
            if f.ndim == 3:
                f = f.squeeze(0)
            feats.append(f.float().cpu())
    return torch.cat(feats, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default="/mnt/net_sda/oymx/frozen_distill/eval_uni2/manifests/uni2_extract_manifest.json",
    )
    ap.add_argument(
        "--feat_dir",
        default="/mnt/net_sda/oymx/frozen_distill/eval_uni2/pt_files",
    )
    ap.add_argument(
        "--pretrained_weights_dir",
        default="/mnt/sdb/chenwm/PFM_Segmentation/weight/uni_v2",
    )
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--max_patches", type=int, default=0, help="0 = all patches")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--roles", nargs="*", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.feat_dir, exist_ok=True)
    Path(args.feat_dir).parent.joinpath("logs").mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(args.manifest).read_text())
    slides = manifest["slides"]
    if args.datasets:
        slides = [s for s in slides if s["dataset"] in args.datasets]
    if args.roles:
        slides = [s for s in slides if s["role"] in args.roles]

    print(f"device={device} n_slides={len(slides)} batch={args.batch_size}", flush=True)
    print("loading uni_v2 ...", flush=True)
    model, transform = load_uni_v2(args.pretrained_weights_dir, device)

    max_patches = args.max_patches if args.max_patches > 0 else None
    done = skipped = failed = 0
    t_all = time.time()
    patch_total = 0

    for i, s in enumerate(slides):
        sid = s["slide_id"]
        out_pt = os.path.join(args.feat_dir, f"{sid}.pt")
        if os.path.isfile(out_pt) and os.path.getsize(out_pt) > 0:
            skipped += 1
            if i % 20 == 0:
                print(f"[{i+1}/{len(slides)}] skip {sid}", flush=True)
            continue
        t0 = time.time()
        try:
            feats = extract_one(
                model,
                transform,
                s["patch_dir"],
                device,
                args.batch_size,
                args.num_workers,
                max_patches,
                args.seed + i,
            )
            if feats is None or feats.numel() == 0:
                failed += 1
                print(f"[{i+1}/{len(slides)}] EMPTY {sid}", flush=True)
                continue
            torch.save(feats.half().contiguous(), out_pt)
            meta = {
                **{
                    k: s[k]
                    for k in (
                        "slide_id",
                        "label",
                        "dataset",
                        "cohort",
                        "organ",
                        "split",
                        "role",
                    )
                },
                "feat_shape": list(feats.shape),
                "feat_path": out_pt,
                "patch_dir": s["patch_dir"],
            }
            with open(out_pt.replace(".pt", ".meta.json"), "w") as f:
                json.dump(meta, f, ensure_ascii=False)
            done += 1
            patch_total += int(feats.shape[0])
            dt = time.time() - t0
            rate = feats.shape[0] / max(dt, 1e-6)
            print(
                f"[{i+1}/{len(slides)}] OK {sid} shape={tuple(feats.shape)} "
                f"{dt:.1f}s {rate:.1f} img/s dataset={s['dataset']}",
                flush=True,
            )
        except Exception as e:
            failed += 1
            print(f"[{i+1}/{len(slides)}] FAIL {sid}: {e}", flush=True)

    elapsed = time.time() - t_all
    summary = {
        "done": done,
        "skipped": skipped,
        "failed": failed,
        "patch_total_new": patch_total,
        "elapsed_sec": elapsed,
        "feat_dir": args.feat_dir,
    }
    print("SUMMARY", json.dumps(summary), flush=True)
    Path(args.feat_dir).parent.joinpath("logs", "extract_summary.json").write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()
