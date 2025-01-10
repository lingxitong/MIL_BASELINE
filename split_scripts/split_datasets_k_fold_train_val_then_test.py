import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import argparse
import os

def Balanced_k_fold_train_val_then_test(args):
    csv_path = args.csv_path
    df = pd.read_csv(csv_path)
    test_ratio = args.test_ratio
    save_dir = args.save_dir
    dataset_name = args.dataset_name
    train_df, test_df = train_test_split(df, test_size=test_ratio, stratify=df['label'], random_state=args.seed, shuffle=True)
    K=args.k
    skf = StratifiedKFold(n_splits=K, random_state=args.seed, shuffle=True)


    for fold, (train_index, val_index) in enumerate(skf.split(train_df, train_df['label'])):
        train_fold_df = train_df.iloc[train_index]
        val_fold_df = train_df.iloc[val_index]
        
        combined_df = pd.concat([
            train_fold_df.rename(columns={'slide_path': 'train_slide_path', 'label': 'train_label'}),
            val_fold_df.rename(columns={'slide_path': 'val_slide_path', 'label': 'val_label'}),
            test_df.rename(columns={'slide_path': 'test_slide_path', 'label': 'test_label'})
        ], axis=1)
        os.makedirs(f'{args.save_dir}/{args.dataset_name}', exist_ok=True)
        combined_df.to_csv(f'{save_dir}/{dataset_name}/Total_{K}-fold_{dataset_name}_{fold+1}fold.csv', index=False)
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--csv_path', type=str, default='/path/to/your/dataset-csv-file.csv')
    argparser.add_argument('--dataset_name', type=str, default='your_dataset_name')
    argparser.add_argument('--test_ratio', type=float, default=0.2) # first select test_ratio of data as test data
    argparser.add_argument('--k', type=int, default=5) # then split the rest of data into k folds
    argparser.add_argument('--save_dir', type=str, default='/dir/to/save/dataset/csvs')
    args = argparser.parse_args()
    Balanced_k_fold_train_val_then_test(args)