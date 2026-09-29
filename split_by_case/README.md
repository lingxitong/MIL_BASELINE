# split_by_case: Case-Level (case_id) Splitting Scripts

## Differences from `split_scripts/`

The scripts under `split_scripts/` perform stratified sampling by CSV **row (slide)**. When one case contains multiple slides, slides of the same case may end up on both the train and test sides, causing **case-level leakage** and inflated evaluation metrics.

The scripts in this directory switch the splitting unit to the **case**:

- All slides of the same `case_id` always go into the same subset (train / val / test are mutually exclusive)
- Class proportions are preserved (case-level stratification) and random seeds are reproducible
- Before writing, each script self-checks that no `case_id` appears in more than one subset, and raises an error immediately on any overlap
- The output format is exactly the same as `split_scripts/` and can be fed directly into `train_mil.py`

## Input Format

The CSV must contain the following three columns (all other columns are ignored):

| Column       | Description                                  |
| ------------ | -------------------------------------------- |
| `case_id`    | Case ID (grouping key)                       |
| `slide_path` | Path to the slide feature file               |
| `label`      | Class label                                  |

Validation rules (the script exits with an error when violated):

- None of the three columns may contain missing values
- One `case_id` may correspond to only one `label`

Reference example: `datasets/example_Dataset_by_case.csv`.

## Script List (One-to-One with `split_scripts/`)

### (A) split_datasets_k_fold_train_val_by_case.py

Standard K-fold (train/val, no separate test set). Suitable for local cross-validation and hyper-parameter tuning.

```shell
python split_datasets_k_fold_train_val_by_case.py --seed 2026 \
    --csv_path datasets/example_Dataset_by_case.csv --save_dir datasets/splits/ \
    --dataset_name TCGA-xxxx --k 5
```

### (B) split_datasets_k_fold_train_val_then_test_by_case.py

First draws a fixed test set at the case level (shared by all folds), then performs K-fold train/val on the remaining cases.

```shell
python split_datasets_k_fold_train_val_then_test_by_case.py --seed 2026 \
    --csv_path datasets/example_Dataset_by_case.csv --save_dir datasets/splits/ \
    --dataset_name TCGA-xxxx --k 5 --test_ratio 0.2
```

### (C) split_datasets_k_fold_train_val_test_by_case.py

First K-fold splits all cases into dev (k-1 folds) and test (1 fold, rotating across folds), then draws val from the dev cases.

```shell
python split_datasets_k_fold_train_val_test_by_case.py --seed 2026 \
    --csv_path datasets/example_Dataset_by_case.csv --save_dir datasets/splits/ \
    --dataset_name TCGA-xxxx --k 5 --val_ratio 0.2
```

### (D) split_datasets_user_define_train_val_test_by_case.py

User-defined train/val/test ratios (single-file output).

```shell
python split_datasets_user_define_train_val_test_by_case.py --seed 2026 \
    --csv_path datasets/example_Dataset_by_case.csv --save_path datasets/splits/TCGA-xxxx_split.csv \
    --dataset_name TCGA-xxxx --train_ratio 0.6 --val_ratio 0.2 --test_ratio 0.2
```

### (E) splits_datasets_user_define_train_test_by_case.py

User-defined train/test ratios (no validation set; train for a fixed number of epochs, then test directly).

```shell
python splits_datasets_user_define_train_test_by_case.py --seed 2026 \
    --csv_path datasets/example_Dataset_by_case.csv --save_path datasets/splits/TCGA-xxxx_train_test.csv \
    --dataset_name TCGA-xxxx --train_ratio 0.7 --test_ratio 0.3
```

### (F) splits_datasets_user_define_train_val_by_case.py

User-defined train/val ratios (no test set; select the best epoch by val and report it).

```shell
python splits_datasets_user_define_train_val_by_case.py --seed 2026 \
    --csv_path datasets/example_Dataset_by_case.csv --save_path datasets/splits/TCGA-xxxx_train_val.csv \
    --dataset_name TCGA-xxxx --train_ratio 0.7 --val_ratio 0.3
```

## Output and Training Integration

- K-fold scripts (A)(B)(C): outputs go to `{save_dir}/{dataset_name}/Total_{k}-fold_{dataset_name}_{i}fold.csv`.
  Set `Dataset.dataset_root_dir` in the model config.yaml to that folder; `train_mil.py` will train fold by fold in filename order.
- Single-split scripts (D)(E)(F): output goes to the single CSV specified by `--save_path`.
  Set `Dataset.dataset_csv_path` in config.yaml to that file; training runs only once.
- The six output columns are exactly the same as `split_scripts/`: `train_slide_path, train_label, val_slide_path, val_label, test_slide_path, test_label`.
  Segments of unequal length are padded with empty values (read back as NaN) and filtered per column with `dropna()` when read; labels are written verbatim (0/1 is not converted to 0.0/1.0).

## Splitting Algorithm

- K-fold (A/B/C): `sklearn.model_selection.StratifiedGroupKFold` (`groups=case_id`, stratified by `label`).
- Single splits and the non-K-fold parts of the two-stage scripts (B/C/D/E/F): cases are first drawn on the **case table** (one row per case, `drop_duplicates('case_id')`) with
  `train_test_split(stratify=label)`, then mapped back to slide rows.
- Ratio parameters apply to the **number of cases**: e.g. `--test_ratio 0.2` means roughly 20% of cases go to test, independent of the number of slides.

## Dependencies

- Python 3.8+, pandas, numpy, scikit-learn >= 1.1 (`StratifiedGroupKFold`)
