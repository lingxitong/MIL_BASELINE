import argparse
from utils.yaml_utils import read_yaml
from process.process_all import process
import warnings
import os
from utils.general_utils import get_time
warnings.filterwarnings('ignore')

def main(arg):
    yaml_path = arg.yaml_path
    print(f"MIL-yaml path: {yaml_path}")
    args = read_yaml(yaml_path)
    

    if args.Dataset.dataset_root_dir == {} and args.Dataset.dataset_csv_path != None:
        '''
        train-val-test split
        '''
        log_root_dir = args.Logs.log_root_dir
        os.makedirs(log_root_dir,exist_ok=True)
        sub_dir = os.path.join(log_root_dir,args.Dataset.DATASET_NAME,args.General.MODEL_NAME)
        os.makedirs(sub_dir,exist_ok=True)
        args.Logs.now_log_dir = os.path.join(sub_dir,f'time_{get_time()}_{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}_seed_{args.General.seed}')
        process(args,yaml_path)
        

    else:
        '''
        train-test with k-fold split or train-val-test with k-fold split
        '''
        dataset_root_dir = args.Dataset.dataset_root_dir
        k = len(os.listdir(dataset_root_dir)) 
        k_fold_csv_paths = [os.path.join(dataset_root_dir,path) for path in os.listdir(dataset_root_dir)]
        process_time = get_time()
        for k_idx,k_fold_csv_path in enumerate(k_fold_csv_paths):
            args.Dataset.dataset_csv_path = k_fold_csv_path
            now_fold = k_idx+1
            args.Dataset.now_fold = now_fold
            log_root_dir = args.Logs.log_root_dir
            os.makedirs(log_root_dir,exist_ok=True)
            sub_dir = os.path.join(log_root_dir,args.Dataset.DATASET_NAME,args.General.MODEL_NAME)
            os.makedirs(sub_dir,exist_ok=True)
            if now_fold != None:
                fold_dir = f'fold_{now_fold}'
                args.Logs.now_log_dir = os.path.join(sub_dir,f'time_{process_time}_{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}_seed_{args.General.seed}/{fold_dir}')
            os.makedirs(args.Logs.now_log_dir,exist_ok=True)
            process(args,yaml_path,k_idx+1)
            print(f'K-Fold:{k_idx+1} Done!')
        
        
    
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path',type=str,default='/data_sdd/lxt/GEM_MIL/MIL_BASELINE/configs/CLAM_MB_MIL.yaml',help='path to MIL-yaml file')
    arg = parser.parse_args()
    main(arg)
    