import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split


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


def Balanced_Train_Val_Test_By_Case(args):
    df = load_and_validate(args.csv_path)
    cases = df.drop_duplicates('case_id')[['case_id', 'label']].reset_index(drop=True)

    train_cases, temp_cases = train_test_split(
        cases, train_size=args.train_ratio, stratify=cases['label'], random_state=args.seed, shuffle=True)
    val_cases, test_cases = train_test_split(
        temp_cases, test_size=args.test_ratio / (args.test_ratio + args.val_ratio),
        stratify=temp_cases['label'], random_state=args.seed, shuffle=True)

    train_df = df[df['case_id'].isin(set(train_cases['case_id']))]
    val_df = df[df['case_id'].isin(set(val_cases['case_id']))]
    test_df = df[df['case_id'].isin(set(test_cases['case_id']))]
    check_case_disjoint(train=train_df, val=val_df, test=test_df)

    write_split_csv(args.save_path, train_df=train_df, val_df=val_df, test_df=test_df)
    print(f'saved -> {args.save_path}')
    print_split_summary([('train', train_df), ('val', val_df), ('test', test_df)])


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--csv_path', type=str, default='/path/to/your/dataset-csv-file.csv')
    argparser.add_argument('--save_path', type=str, default='/path/to/your/save-path.csv')
    argparser.add_argument('--dataset_name', type=str, default='your_dataset_name')
    argparser.add_argument('--train_ratio', type=float, default=0.6)
    argparser.add_argument('--val_ratio', type=float, default=0.2)
    argparser.add_argument('--test_ratio', type=float, default=0.2)
    args = argparser.parse_args()
    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1) < 1e-8, \
        'train_ratio + val_ratio + test_ratio must be equal to 1'
    Balanced_Train_Val_Test_By_Case(args)
