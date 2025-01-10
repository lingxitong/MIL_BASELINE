import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import argparse
import os

def Balanced_k_fold_train_val_test(args):
    csv_path = args.csv_path
    df = pd.read_csv(csv_path)
    val_ratio = args.val_ratio
    save_dir = args.save_dir
    dataset_name = args.dataset_name
    K=args.k
    skf = StratifiedKFold(n_splits=K, random_state=args.seed, shuffle=True)


    for fold, (TRAIN_index, test_index) in enumerate(skf.split(df, df['label'])):
        TRAIN_fold_df = df.iloc[TRAIN_index]
        test_fold_df = df.iloc[test_index]
        train_fold_df, val_fold_df = train_test_split(TRAIN_fold_df, test_size=val_ratio, stratify=TRAIN_fold_df['label'], random_state=args.seed, shuffle=True)
        combined_df = pd.concat([
            train_fold_df.rename(columns={'slide_path': 'train_slide_path', 'label': 'train_label'}),
            val_fold_df.rename(columns={'slide_path': 'val_slide_path', 'label': 'val_label'}),
            test_fold_df.rename(columns={'slide_path': 'test_slide_path', 'label': 'test_label'})
        ], axis=1)
        os.makedirs(f'{args.save_dir}/{args.dataset_name}', exist_ok=True)
        combined_df.to_csv(f'{save_dir}/{dataset_name}/Total_{K}-fold_{dataset_name}_{fold+1}fold.csv', index=False)
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--csv_path', type=str, default='/path/to/your/dataset-csv-file.csv')
    argparser.add_argument('--dataset_name', type=str, default='your_dataset_name')
    argparser.add_argument('--k', type=int, default=5) # split the total of data into k folds (dev-set and test-set)
    argparser.add_argument('--val_ratio', type=float, default=0.2) # then select the val ratio in dev-set
    argparser.add_argument('--save_dir', type=str, default='/dir/to/save/dataset/csvs')
    args = argparser.parse_args()
    Balanced_k_fold_train_val_test(args)