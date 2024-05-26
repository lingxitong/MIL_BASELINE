import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold


def Balanced_k_fold_train_val_test(args)
    csv_path = args.csv_path
    df = pd.read_csv(csv_path)
    test_ratio = args.test_ratio
    dataset_name = args.dataset_name
    train_df, test_df = train_test_split(df, test_size=test_ratio, stratify=df['label'])
    K=args.k
    skf = StratifiedKFold(n_splits=K)


    for fold, (train_index, val_index) in enumerate(skf.split(train_df, train_df['label'])):
        train_fold_df = train_df.iloc[train_index]
        val_fold_df = train_df.iloc[val_index]
        
        combined_df = pd.concat([
            train_fold_df.rename(columns={'slide_path': 'train_slide_path', 'label': 'train_label'}),
            val_fold_df.rename(columns={'slide_path': 'val_slide_path', 'label': 'val_label'}),
            test_df.rename(columns={'slide_path': 'test_slide_path', 'label': 'test_label'})
        ], axis=1)
        combined_df.to_csv(f'{save_dir}/{dataset_name}/Total_{k}-fold_{dataset_name}_{fold+1}fold.csv', index=False)
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--csv_path', type=str, default='/data_sdd/lxt/GEM_MIL/MIL_BASELINE/datasets/GEM-2cls.csv')
    argparser.add_argument('--dataset_name', type=str, default='GEM-2cls')
    argparser.add_argument('--test_ratio', type=float, default=0.2)
    argparser.add_argument('--k', type=int, default=5)
    argparser.add_argument('--save_dir', type=str, default='/data_sdd/lxt/GEM_MIL/MIL_BASELINE/datasets_csv/')
    args = argparser.parse_args()
    Balanced_K_fold_Train_Val_Test(args)