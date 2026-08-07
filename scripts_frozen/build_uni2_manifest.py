#!/usr/bin/env python3
"""Build slide manifest for UNI2 feature extraction + linear-probe eval."""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

EVAL_READY = Path("/home/oymx/work/frozen/data/eval_ready")
INTERNAL_ROOT = Path("/mnt/sdb/cyx/FROZEN/SYSFL/2024_3_4_2_class_task_split_5_2_3")
OUT = Path("/mnt/net_sda/oymx/frozen_distill/eval_uni2/manifests")


def _slide_id(item: dict) -> str:
    pd = item.get("patch_dir") or ""
    return Path(pd).name


def from_eval_json(name: str, role: str) -> list[dict]:
    obj = json.loads((EVAL_READY / f"{name}.json").read_text())
    rows = []
    for it in obj["test"]:
        if not it.get("patch_dir_exists", True):
            continue
        if (it.get("n_jpg") or 0) <= 0:
            continue
        rows.append(
            {
                "slide_id": it["slide_id"],
                "label": it["label"],
                "patch_dir": it["patch_dir"],
                "dataset": name,
                "cohort": obj["cohort"],
                "organ": obj["organ"],
                "split": "test",
                "role": role,
                "n_jpg": it.get("n_jpg"),
            }
        )
    return rows


def from_internal(organ: str, splits: list[str], dataset: str, role: str) -> list[dict]:
    obj = json.loads((INTERNAL_ROOT / organ / "dino_lt_988w.json").read_text())
    rows = []
    for split in splits:
        for it in obj[split]:
            pd = it["patch_dir"]
            if not os.path.isdir(pd):
                continue
            # cheap non-empty check
            try:
                names = os.listdir(pd)
            except OSError:
                continue
            if not any(n.lower().endswith((".jpg", ".jpeg", ".png")) for n in names[:5]) and not any(
                n.lower().endswith((".jpg", ".jpeg", ".png")) for n in names
            ):
                continue
            rows.append(
                {
                    "slide_id": _slide_id(it),
                    "label": it["label"],
                    "patch_dir": pd,
                    "dataset": dataset,
                    "cohort": "FAHSYSU_internal",
                    "organ": organ,
                    "split": split,
                    "role": role,
                    "n_jpg": None,
                }
            )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    # Priority eval sets (features we must have)
    rows += from_eval_json("internal_brain_test", role="eval")
    rows += from_eval_json("gyey_breast", role="eval")
    rows += from_eval_json("gfph_breast", role="eval")
    # Internal train/valid for paper-style linear probe heads
    rows += from_internal("brain", ["train", "valid"], dataset="internal_brain_tv", role="train_lp")
    rows += from_internal("breast", ["train", "valid"], dataset="internal_breast_tv", role="train_lp")

    # de-dup by patch_dir (brain test already in eval; tv must not duplicate)
    seen = set()
    uniq = []
    for r in rows:
        key = r["patch_dir"]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)

    manifest = {
        "n_slides": len(uniq),
        "by_dataset": dict(Counter(r["dataset"] for r in uniq)),
        "by_role": dict(Counter(r["role"] for r in uniq)),
        "by_label": dict(Counter(r["label"] for r in uniq)),
        "slides": uniq,
        "notes": {
            "backbone": "uni_v2",
            "weights": "/mnt/sdb/chenwm/PFM_Segmentation/weight/uni_v2/pytorch_model.bin",
            "patch_size": "256x256 jpg -> resize 224 for UNI2",
            "feat_dim": 1536,
        },
    }
    out_path = OUT / "uni2_extract_manifest.json"
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"wrote {out_path}")
    print("n_slides", manifest["n_slides"])
    print("by_dataset", manifest["by_dataset"])
    print("by_role", manifest["by_role"])


if __name__ == "__main__":
    main()
