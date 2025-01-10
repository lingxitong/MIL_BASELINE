from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import os
import argparse
def Balanced_K_fold_Train_Val(args):
    csv_path = args.csv_path
    k = args.k
    df = pd.read_csv(csv_path)
    X = df['slide_path']
    y = df['label']
    save_dir = args.save_dir
    dataset_name = args.dataset_name
    skf = StratifiedKFold(n_splits=k, random_state=args.seed,shuffle=True)
    for k_idx,(train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index].values.tolist(), X[test_index].values.tolist()
        y_train, y_test = y[train_index].values.tolist(), y[test_index].values.tolist()
        
        max_len = max(len(X_train), len(X_test))
        X_train += [np.nan] * (max_len - len(X_train))
        y_train += [np.nan] * (max_len - len(y_train))
        X_test += [np.nan] * (max_len - len(X_test))
        y_test += [np.nan] * (max_len - len(y_test))
        
        one_fold = pd.DataFrame({
            'train_slide_path': X_train, 
            'train_label': y_train,
            'val_slide_path': X_test, 
            'val_label': y_test,
            'test_slide_path':  [np.nan] * max_len,
            'test_label': [np.nan] * max_len
        })
        
        os.makedirs(f'{args.save_dir}/{args.dataset_name}', exist_ok=True)
        one_fold.to_csv(f'{save_dir}/{args.dataset_name}/Total_{k}-fold_{dataset_name}_{k_idx+1}fold.csv', index=False)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=42)
    argparser.add_argument('--csv_path', type=str, default='/path/to/your/dataset-csv-file.csv')
    argparser.add_argument('--dataset_name', type=str, default='your_dataset_name')
    argparser.add_argument('--k', type=int, default=3)
    argparser.add_argument('--save_dir', type=str, default='/dir/to/save/dataset/csvs')
    args = argparser.parse_args()
    Balanced_K_fold_Train_Val(args)