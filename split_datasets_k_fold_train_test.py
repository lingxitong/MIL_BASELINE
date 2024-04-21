from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import os
import argparse
def Balanced_K_fold_Train_Test(args):
    csv_path = args.csv_path
    k = args.k
    df = pd.read_csv(csv_path)
    X = df['slide_path']
    y = df['label']
    
    skf = StratifiedKFold(n_splits=5)
    for k_idx,(train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index].values.tolist(), X[test_index].values.tolist()
        y_train, y_test = y[train_index].values.tolist(), y[test_index].values.tolist()
        
        # 如果训练数据和测试数据的长度不一样，用NaN填充缺失的部分
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
            'test_slide_path': X_test, 
            'test_label': y_test
        })
        
        os.makedirs(f'{args.save_dir}/{args.dataset}', exist_ok=True)
        one_fold.to_csv(f'/data_sdd/lxt/GEM_MIL/MIL_BASELINE/datasets_csv/{args.dataset}/Total_{k}-fold_GEM-2cls_{k_idx+1}fold.csv', index=False)







if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--csv_path', type=str, default='/data_sdd/lxt/GEM_MIL/MIL_BASELINE/datasets/GEM-2cls.csv')
    argparser.add_argument('--dataset_name', type=str, default='GEM-2cls')
    argparser.add_argument('--k', type=int, default=5)
    argparser.add_argument('--save_dir', type=str, default='/data_sdd/lxt/GEM_MIL/MIL_BASELINE/datasets_csv/')
    args = argparser.parse_args()
    
    
    Balanced_K_fold_Train_Test(args)