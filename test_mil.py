import argparse
from utils.yaml_utils import read_yaml
from torch.utils.data import DataLoader
from utils.loop_utils import val_loop,clam_val_loop
import warnings
from utils.wsi_utils import WSI_Dataset
import torch
import shutil
import os
from utils.model_utils import get_model,get_criterion
warnings.filterwarnings('ignore')

def test(args):
    yaml_path = args.yaml_path
    
    
    print(f"MIL-model-yaml path: {yaml_path}")
    yaml_args = read_yaml(yaml_path)
    mil_model = get_model(yaml_args)
    model_name = yaml_args.General.MODEL_NAME
    num_classes = yaml_args.General.num_classes
    
    test_dataset_csv = args.test_dataset_csv
    print(f"Dataset csv path: {test_dataset_csv}")
    test_ds = WSI_Dataset(test_dataset_csv,'test')
    test_dataloader = DataLoader(test_ds,batch_size=1,shuffle=False)
    
    device = torch.device(f'cuda:{yaml_args.General.device}')
    mil_model = mil_model.to(device)
    
    model_weight_path = args.model_weight_path
    print(f"Model weight path: {model_weight_path}")
    mil_model.load_state_dict(torch.load(model_weight_path))
    criterion = yaml_args.Model.criterion
    criterion = get_criterion(criterion)
    
    # CLAM_SB_MIL and CLAM_MB_MIL models have different val loop pipeline (has instance loss)
    if yaml_args.General.MODEL_NAME == 'CLAM_MB_MIL':
        test_loss,test_metrics = clam_val_loop(device,num_classes,mil_model,test_dataloader,criterion)
    elif yaml_args.General.MODEL_NAME == 'CLAM_SB_MIL':
        test_loss,test_metrics = clam_val_loop(device,num_classes,mil_model,test_dataloader,criterion)
    else:
        test_loss,test_metrics =  val_loop(device,num_classes,mil_model,test_dataloader,criterion)
    
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    print('----------------INFO----------------\n')
    print(f'{FAIL}Test_Loss:{ENDC}{test_loss}\n')
    print(f'{FAIL}Test_Metrics:  {ENDC}{test_metrics}\n')
    
    test_log_dir = args.test_log_dir
    os.makedirs(test_log_dir,exist_ok=True)
    new_yaml_path = os.path.join(test_log_dir,f'Test_{model_name}.yaml')
    shutil.copyfile(yaml_path,new_yaml_path)
    new_test_dataset_csv_path = os.path.join(test_log_dir,f'Test_dataset_{yaml_args.Dataset.DATASET_NAME}.csv')
    shutil.copyfile(test_dataset_csv,new_test_dataset_csv_path)
    

    test_log_path = os.path.join(test_log_dir,f'Test_Log_{model_name}.txt')
    log_to_save = {'test_loss':test_loss,'test_metrics':test_metrics}
    with open(test_log_path,'w') as f:
        f.write(str(log_to_save))

    
    print(f"Test log saved at: {test_log_path}")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path',type=str,default='/path/to/your/config-yaml',help='path to MIL-model-yaml file')
    parser.add_argument('--test_dataset_csv',type=str,default='/path/to/your/ds-csv-path',help='path to dataset csv')
    parser.add_argument('--model_weight_path',type=str,default='/path/to/your/model-weight',help='path to model weights ')
    parser.add_argument('--test_log_dir',type=str,default='/path/to/your/test-log-dir',help='path to test log dir')
    args = parser.parse_args()
    test(args)
    