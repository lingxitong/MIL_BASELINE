from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import os
import argparse
    
from sklearn.model_selection import train_test_split

def Balanced_Train_Val_Test(args):
    # 加载数据
    df = pd.read_csv(args.csv_path)

    # 定义特征和目标变量
    X = df.drop('label', axis=1)  # 假设你的目标列名为'label'
    y = df['label']

    # 计算训练集和测试集的大小
    train_size = args.train_ratio
    test_size = args.test_ratio  # 测试集的大小
    val_size = args.val_ratio  # 验证集的大小

    # 首先，划分训练集和其余部分
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, stratify=y, random_state=42)

    # 然后，从其余部分中划分验证集和测试集
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size/(test_size + val_size), stratify=y_temp, random_state=42)

    # 创建一个新的DataFrame来保存结果
    result = pd.DataFrame({
        'train_slide_path': pd.Series(X_train.values.flatten()),
        'train_label': pd.Series(y_train.values),
        'val_slide_path': pd.Series(X_val.values.flatten()),
        'val_label': pd.Series(y_val.values),
        'test_slide_path': pd.Series(X_test.values.flatten()),
        'test_label': pd.Series(y_test.values)
    })

    # 保存结果到CSV文件
    result.to_csv(args.save_path, index=False)





if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--csv_path', type=str, default='/data_sdd/lxt/GEM_MIL/MIL_BASELINE/datasets/GEM-2cls.csv')
    argparser.add_argument('--save_path', type=str, default='/data_sdd/lxt/GEM-2cls-train_val_test.csv')
    argparser.add_argument('--dataset_name', type=str, default='GEM-2cls')
    argparser.add_argument('--train_ratio', type=float, default=0.6)
    argparser.add_argument('--val_ratio', type=float, default=0.2)
    argparser.add_argument('--test_ratio', type=float, default=0.2)
    args = argparser.parse_args()
    assert args.train_ratio + args.val_ratio + args.test_ratio == 1 , print('train_ratio + val_ratio + test_ratio must be equal to 1')
    Balanced_Train_Val_Test(args)

