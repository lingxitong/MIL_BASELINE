## **Different Dataset Split Methods**
### **(A) split_datasets_k_fold_train_val.py**
This script performs standard k-fold cross-validation splitting without distinguishing between validation and test sets.
### **(B) split_datasets_user_define_train_val_test.py**
This script allows users to customize the ratio of the training, validation, and test sets.
### **(C) split_datasets_k_fold_train_val_then_test.py**
This script first splits the test set according to the user-specified ratio, and then performs k-fold cross-validation on the remaining dataset. This method distinguishes between the validation set and the test set.
### **(D) split_datasets_k_fold_train_val_test.py**
This script first uses standard k-fold cross-validation to split the data into a development set and a test set, and then splits the development set into training and validation sets according to the user-specified ratio. The script distinguishes between the validation set and the test set.



## **Pay Attention to the following tips**
1.Regardless of the method used, the split results will always include both the validation and test sets. However, when there is no distinction between the validation and test sets, the validation and test sets will be the same. The `VIST (Val is Test)` parameter can be set in the configuration file to avoid redundant calculations.
