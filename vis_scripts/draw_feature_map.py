from sklearn.manifold import TSNE
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(grandparent_dir)
import numpy as np
import argparse
from utils.model_utils import get_model_from_yaml
import matplotlib.pyplot as plt
import argparse
from utils.yaml_utils import read_yaml
from torch.utils.data import DataLoader
from utils.loop_utils import val_loop,clam_val_loop,ds_val_loop,dtfd_val_loop
import warnings
from utils.wsi_utils import WSI_Dataset,CDP_MIL_WSI_Dataset,LONG_MIL_WSI_Dataset
import torch
from utils.model_utils import get_model_from_yaml,get_criterion
import ast
warnings.filterwarnings('ignore')

def draw_tsne(feature_tensor, label_tensor, id2class, save_path, fig_size = (10, 8), seed = 42):
    """
    Draws a TSNE plot for the given feature tensor and labels, and saves it to the specified path.

    Parameters:
    feature_tensor (numpy.ndarray): An N x D tensor of features.
    label_tensor (numpy.ndarray): An N x 1 tensor of labels.
    id2class (str): str type dictionary mapping label ids to class names.
    save_path (str): The path where the plot will be saved.
    fig_size (tuple, optional): The size of the figure, default is (10, 8).

    Returns: 
    None
    """
    id2class = ast.literal_eval(id2class)
    perplexity = min(30, feature_tensor.shape[0] - 1)  # 默认 30，且保证小于样本数
    tsne = TSNE(perplexity=perplexity, n_components=2, random_state=seed)
    tsne_result = tsne.fit_transform(feature_tensor)

    plt.figure(figsize=fig_size)
    for label_id, class_name in id2class.items():
        indices = np.where(label_tensor == int(label_id))[0]
        plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], label=class_name)
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    
def main(args):
    yaml_path = args.yaml_path
    yaml_args = read_yaml(yaml_path)
    model_name = yaml_args.General.MODEL_NAME
    mil_model = get_model_from_yaml(yaml_args)
    model_name = yaml_args.General.MODEL_NAME
    print(f"Model name: {model_name}")
    num_classes = yaml_args.General.num_classes
    test_dataset_csv = args.test_dataset_csv
    print(f"Dataset csv path: {test_dataset_csv}")
    # CDP_MIL and LONG_MIL models have different dataset pipeline
    if model_name == 'CDP_MIL':
        raise NotImplementedError("CDP_MIL model is not supported for feature map visualization now.")
        test_ds = CDP_MIL_WSI_Dataset(test_dataset_csv,yaml_args.Dataset.BeyesGuassian_pt_dir,'test')
    elif model_name == 'LONG_MIL':
        LONG_MIL_WSI_Dataset(test_dataset_csv,yaml_args.Dataset.h5_csv_path,'test')
    test_ds = WSI_Dataset(test_dataset_csv,'test')
    test_dataloader = DataLoader(test_ds,batch_size=1,shuffle=False)
    model_weight_path = args.ckpt_path
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
        mil_model = mil_model.to(device)
        mil_model.load_state_dict(torch.load(model_weight_path,weights_only=True))

    if yaml_args.General.MODEL_NAME == 'CLAM_MB_MIL' or yaml_args.General.MODEL_NAME == 'CLAM_SB_MIL':
        bag_weight = yaml_args.Model.bag_weight
        WSI_features = clam_val_loop(device,num_classes,mil_model,test_dataloader,criterion,bag_weight,retrun_WSI_feature=True)
    elif yaml_args.General.MODEL_NAME == 'DS_MIL':
        WSI_features =  ds_val_loop(device,num_classes,mil_model,test_dataloader,criterion,retrun_WSI_feature=True)
    elif yaml_args.General.MODEL_NAME == 'DTFD_MIL':
        WSI_features =  dtfd_val_loop(device,num_classes,model_list,test_dataloader,criterion,yaml_args.Model.num_Group,yaml_args.Model.grad_clipping,yaml_args.Model.distill,yaml_args.Model.total_instance,retrun_WSI_feature=True)
    else:
        WSI_features =  val_loop(device,num_classes,mil_model,test_dataloader,criterion,retrun_WSI_feature=True)

    WSI_labels = np.array(test_ds.labels_list)
    WSI_labels = WSI_labels.astype(int)
    draw_tsne(WSI_features, WSI_labels, args.id2class, args.save_path, (10,8) ,args.seed)
    print(f"TSNE plot saved at {args.save_path}")

if __name__ == "__main__":    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--yaml_path', type=str,default = '',help='path to yaml file')
    argparser.add_argument('--seed', type=int,default = 42, help='seed for reproducibility')
    argparser.add_argument('--ckpt_path', type=str,default='',help='path to pretrained weights')
    argparser.add_argument('--save_path', type=str, default='',help='path to save the model')
    argparser.add_argument('--id2class', type=str,default='',help='str type dictionary mapping label ids to class names')
    argparser.add_argument('--test_dataset_csv', type=str,default='',help='path to dataset csv file')
    args = argparser.parse_args()
    main(args)
