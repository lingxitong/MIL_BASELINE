## :satellite: **Different Dataset Split Methods**
### **(A) split_datasets_k_fold_train_val.py**
This script performs standard k-fold cross-validation splitting without distinguishing between validation and test sets. This kind of partitioning is suitable for scenarios like local Cross Validation and hyperparameter tuning in competitions such as **Kaggle**. Sometimes this kind of partitioning can also be seen in algorithmic papers, but it is important to note that **fair comparisons** need to be made, meaning the same partitioning should be used for all models.

```shell
python split_datasets_k_fold_train_val.py --seed 2024 --csv_path /datasets/example_Dataset.csv --save_dir /your/save/dir --dataset_name Your_Dataset_Name --k 5 
```
### **(B) split_datasets_k_fold_train_val_then_test.py**
This script first splits the test set according to the user-specified ratio, and then performs k-fold cross-validation on the remaining dataset. 

```shell
python split_datasets_k_fold_train_val_then_test.py --seed 2024 --csv_path /datasets/example_Dataset.csv --save_dir /your/save/dir --dataset_name Your_Dataset_Name --k 5 --test_ratio 0.2 
```
### **(C) split_datasets_k_fold_train_val_test.py**
This script first uses standard k-fold cross-validation to split the data into a `TRAIN` set and a test set, and then splits the `TRAIN` set into training and validation sets according to the user-specified ratio. 

```shell
python split_datasets_k_fold_train_val_test.py --seed 2024 --csv_path /datasets/example_Dataset.csv --save_dir /your/save/dir --dataset_name Your_Dataset_Name --k 5 --val_ratio 0.2 
```
### **(D) split_datasets_user_define_train_val_test.py**
This script allows users to customize the ratio of the training, validation, and test sets.

```shell
python splits_datasets_user_define_train_val_test.py --seed 2024 --csv_path /datasets/example_Dataset.csv --save_path /your/save/csv --dataset_name Your_Dataset_Name --train_ratio 0.6 --val_ratio 0.2 --test_ratio 0.2
```
### **(E) split_datasets_user_define_train_test.py**
This script allows users to customize the ratio of the training and test sets. There is no validation set, so the process will not conclude model selection, which means the process will train `num_epochs` and test once.

```shell
python splits_datasets_user_define_train_test.py --seed 2024 --csv_path /datasets/example_Dataset.csv --save_path /your/save/csv --dataset_name Your_Dataset_Name --train_ratio 0.7 --test_ratio 0.3
```
### **(F) split_datasets_user_define_train_val.py**
This script allows users to customize the ratio of the training and validation sets. There is no test set, so the process will conclude model selection, which means the process will train one epoch and evaluation once and then report the best epoch metrics. 

```shell
python splits_datasets_user_define_train_val.py --seed 2024 --csv_path /datasets/example_Dataset.csv --save_path /your/save/csv --dataset_name Your_Dataset_Name --train_ratio 0.7 --val_ratio 0.3
```

### Dataset Settings in Config.yaml
The following fields are included in all configuration files.
```bash
Dataset:
    DATASET_NAME: your_dataset_name
    # to use None-fold split, open dataset_csv_dir
    # to use k-fold split, open dataset_root_dir
    
    dataset_csv_path: /path/to/your/dataset.csv
    # dataset_root_dir: /dir/to/your/dataset_dir/
```
For the method without k-fold splitting, you only need to provide the CSV file path, so you should open the `dataset_csv_path` field.   For the method with k-fold splitting, your split results are stored in a folder containing k separate CSV files, so you need to open the `dataset_root_dir` field. The `DATASET_NAME` field is used to label the dataset name When saving logs.