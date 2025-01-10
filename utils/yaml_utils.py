import yaml
from addict import Dict
from ruamel.yaml import YAML
import ruamel.yaml as ryaml
import shutil
import os
def read_yaml(fpath=None):
    with open(fpath, mode="r") as file:
        yml = yaml.load(file, Loader=yaml.Loader)
        return Dict(yml)
    
def update_config_from_options(config, options):
    for option in options:
        key, value = option.split('=')
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = type(d[keys[-1]])(value)
    return config


from ruamel.yaml import YAML

def change_yaml_by_options(yaml_path, options):
    Yaml = YAML(typ='rt') 
    with open(yaml_path, 'r', encoding='utf-8') as file:
        config = Yaml.load(file)  

    for option in options:
        key, value = option.split('=')
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = type(d[keys[-1]])(value)  

    with open(yaml_path, 'w', encoding='utf-8') as file:
        Yaml.dump(config, file) 



def save_yaml(args,yaml_path,options):
    dst_path = os.path.join(args.Logs.now_log_dir,os.path.basename(yaml_path))
    shutil.copyfile(yaml_path,dst_path)
    if options != None:
        change_yaml_by_options(dst_path,options)