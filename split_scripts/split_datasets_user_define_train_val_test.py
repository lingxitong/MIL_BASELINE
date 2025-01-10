from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import os
import argparse
    
from sklearn.model_selection import train_test_split

def Balanced_Train_Val_Test(args):
    df = pd.read_csv(args.csv_path)

    X = df.drop('label', axis=1)  
    y = df['label']

    train_size = args.train_ratio
    test_size = args.test_ratio  
    val_size = args.val_ratio  

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, stratify=y, random_state=args.seed)

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size/(test_size + val_size), stratify=y_temp, random_state=args.seed, shuffle=True)

    result = pd.DataFrame({
        'train_slide_path': pd.Series(X_train.values.flatten()),
        'train_label': pd.Series(y_train.values),
        'val_slide_path': pd.Series(X_val.values.flatten()),
        'val_label': pd.Series(y_val.values),
        'test_slide_path': pd.Series(X_test.values.flatten()),
        'test_label': pd.Series(y_test.values)
    })

    result.to_csv(args.save_path, index=False)


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
    assert args.train_ratio + args.val_ratio + args.test_ratio == 1 , print('train_ratio + val_ratio + test_ratio must be equal to 1')
    Balanced_Train_Val_Test(args)

