import argparse
import os

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def load_and_validate(csv_path):
    df = pd.read_csv(csv_path)
    required = ['case_id', 'slide_path', 'label']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'CSV is missing required columns: {missing}')
    if df[required].isna().any().any():
        raise ValueError('case_id / slide_path / label contains missing values, please clean the data first')
    nunique = df.groupby('case_id')['label'].nunique()
    bad = nunique[nunique > 1]
    if len(bad) > 0:
        raise ValueError(f'{len(bad)} case_id(s) have multiple different labels (examples: {list(bad.index[:5])}), cannot perform case-level stratified splitting')
    return df


def check_case_disjoint(**frames):
    sets = [(name, set(frame['case_id'])) for name, frame in frames.items() if frame is not None]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i][1] & sets[j][1]
            if overlap:
                raise ValueError(f'case-level leakage: {sets[i][0]} and {sets[j][0]} overlap on {len(overlap)} case(s)')


def write_split_csv(save_path, train_df=None, val_df=None, test_df=None):
    def col(frame, name):
        return frame[name].tolist() if frame is not None else []
    data = {
        'train_slide_path': col(train_df, 'slide_path'),
        'train_label': col(train_df, 'label'),
        'val_slide_path': col(val_df, 'slide_path'),
        'val_label': col(val_df, 'label'),
        'test_slide_path': col(test_df, 'slide_path'),
        'test_label': col(test_df, 'label'),
    }
    max_len = max((len(v) for v in data.values()), default=0)
    columns = {k: pd.Series(v + [None] * (max_len - len(v)), dtype='object') for k, v in data.items()}
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(columns).to_csv(save_path, index=False)


def print_split_summary(parts):
    for name, frame in parts:
        if frame is None or len(frame) == 0:
            print(f'  {name}: 0 cases / 0 slides')
            continue
        dist = frame.drop_duplicates('case_id')['label'].value_counts().sort_index().to_dict()
        print(f'  {name}: {frame["case_id"].nunique()} cases / {len(frame)} slides, case-level label distribution {dist}')


def Balanced_k_fold_train_val_then_test_By_Case(args):
    df = load_and_validate(args.csv_path)
    cases = df.drop_duplicates('case_id')[['case_id', 'label']].reset_index(drop=True)
    dev_cases, test_cases = train_test_split(
        cases, test_size=args.test_ratio, stratify=cases['label'], random_state=args.seed, shuffle=True)
    test_df = df[df['case_id'].isin(set(test_cases['case_id']))].reset_index(drop=True)
    dev_df = df[df['case_id'].isin(set(dev_cases['case_id']))].reset_index(drop=True)

    k = args.k
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=args.seed)
    save_dir = os.path.join(args.save_dir, args.dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    for fold, (train_index, val_index) in enumerate(skf.split(dev_df, y=dev_df['label'], groups=dev_df['case_id'])):
        train_df, val_df = dev_df.iloc[train_index], dev_df.iloc[val_index]
        check_case_disjoint(train=train_df, val=val_df, test=test_df)
        save_path = os.path.join(save_dir, f'Total_{k}-fold_{args.dataset_name}_{fold + 1}fold.csv')
        write_split_csv(save_path, train_df=train_df, val_df=val_df, test_df=test_df)
        print(f'[fold {fold + 1}/{k}] -> {save_path}')
        print_split_summary([('train', train_df), ('val', val_df), ('test', test_df)])


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--csv_path', type=str, default='/path/to/your/dataset-csv-file.csv')
    argparser.add_argument('--dataset_name', type=str, default='your_dataset_name')
    argparser.add_argument('--test_ratio', type=float, default=0.2)
    argparser.add_argument('--k', type=int, default=5)
    argparser.add_argument('--save_dir', type=str, default='/dir/to/save/dataset/csvs')
    args = argparser.parse_args()
    Balanced_k_fold_train_val_then_test_By_Case(args)
