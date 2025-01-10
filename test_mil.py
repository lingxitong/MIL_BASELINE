import argparse
from utils.yaml_utils import read_yaml
from torch.utils.data import DataLoader
from utils.loop_utils import val_loop,clam_val_loop,ds_val_loop,dtfd_val_loop
import warnings
from utils.wsi_utils import WSI_Dataset,CDP_MIL_WSI_Dataset,LONG_MIL_WSI_Dataset
import torch
import shutil
import os
from utils.model_utils import get_model_from_yaml,get_criterion
warnings.filterwarnings('ignore')

def test(args):
    yaml_path = args.yaml_path
    print(f"MIL-model-yaml path: {yaml_path}")
    yaml_args = read_yaml(yaml_path)
    model_name = yaml_args.General.MODEL_NAME
    num_classes = yaml_args.General.num_classes
    test_dataset_csv = args.test_dataset_csv
    print(f"Dataset csv path: {test_dataset_csv}")
    # CDP_MIL and LONG_MIL models have different dataset pipeline
    if model_name == 'CDP_MIL':
        test_ds = CDP_MIL_WSI_Dataset(test_dataset_csv,yaml_args.Dataset.BeyesGuassian_pt_dir,'test')
    elif model_name == 'LONG_MIL':
        LONG_MIL_WSI_Dataset(test_dataset_csv,yaml_args.Dataset.h5_csv_path,'test')
    test_ds = WSI_Dataset(test_dataset_csv,'test')
    test_dataloader = DataLoader(test_ds,batch_size=1,shuffle=False)
    model_weight_path = args.model_weight_path
    print(f"Model weight path: {model_weight_path}")
    device = torch.device(f'cuda:{yaml_args.General.device}')
    criterion = get_criterion(yaml_args.Model.criterion)
    if yaml_args.General.MODEL_NAME == 'DTFD_MIL':
        classifier,attention,dimReduction,attCls = get_model_from_yaml(yaml_args)
        state_dict = torch.load(model_weight_path,weights_only=True)
        classifier.load_state_dict(state_dict['classifier'])
        attention.load_state_dict(state_dict['attention'])
        dimReduction.load_state_dict(state_dict['dimReduction'])
        attCls.load_state_dict(state_dict['attCls'])
        model_list = [classifier,attention,dimReduction,attCls]
        model_list = [model.to(device).eval() for model in model_list]
    else:
        mil_model = get_model_from_yaml(yaml_args)
        mil_model = mil_model.to(device)
        mil_model.load_state_dict(torch.load(model_weight_path,weights_only=True))

    
    # CLAM_SB_MIL and CLAM_MB_MIL models have different val loop pipeline (has instance loss)
    if yaml_args.General.MODEL_NAME == 'CLAM_MB_MIL':
        bag_weight = yaml_args.Model.bag_weight
        test_loss,test_metrics = clam_val_loop(device,num_classes,mil_model,test_dataloader,criterion,bag_weight)
    elif yaml_args.General.MODEL_NAME == 'CLAM_SB_MIL':
        bag_weight = yaml_args.Model.bag_weight
        test_loss,test_metrics = clam_val_loop(device,num_classes,mil_model,test_dataloader,criterion,bag_weight)
    elif yaml_args.General.MODEL_NAME == 'DS_MIL':
        test_loss,test_metrics =  ds_val_loop(device,num_classes,mil_model,test_dataloader,criterion)
    elif yaml_args.General.MODEL_NAME == 'DTFD_MIL':
        test_loss,test_metrics =  dtfd_val_loop(device,num_classes,model_list,test_dataloader,criterion,yaml_args.Model.num_Group,yaml_args.Model.grad_clipping,yaml_args.Model.distill,yaml_args.Model.total_instance)
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

