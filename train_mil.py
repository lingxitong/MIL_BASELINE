import argparse
from utils.yaml_utils import read_yaml
from process.process_all import process
import warnings
warnings.filterwarnings('ignore')

def main(arg):
    yaml_path = arg.yaml_path
    print(f"MIL-yaml path: {yaml_path}")
    args = read_yaml(yaml_path)
    process(args,yaml_path)
    
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path',type=str,default='/data_sda/lxt/CAMELYON_BENCHMARK/configs/CLAM_MB_MIL.yaml',help='path to MIL-yaml file')
    arg = parser.parse_args()
    main(arg)
    