import pandas as pd
import math
import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

class WSI_Dataset(Dataset):
    def __init__(self,dataset_info_csv_path,group):
        assert group in ['train','val','test'], 'group must be in [train,val,test]'
        self.dataset_info_csv_path = dataset_info_csv_path
        self.dataset_df = pd.read_csv(self.dataset_info_csv_path)
        self.slide_path_list = self.dataset_df[group+'_slide_path'].dropna().to_list()
        self.labels_list = self.dataset_df[group+'_label'].dropna().to_list()

    def __len__(self):
        return len(self.slide_path_list)
    
    def __getitem__(self, idx):

        slide_path = self.slide_path_list[idx]
        label = int(self.labels_list[idx])
        label = torch.tensor(label)
        feat = torch.load(slide_path)
        if len(feat.shape) == 3:
            feat = feat.squeeze(0)
        return feat,label

    def is_None_Dataset(self):
        return (self.__len__() == 0)    
    
    def is_with_labels(self):
        return (len(self.labels_list) != 0)
    

    
class CDP_MIL_WSI_Dataset(WSI_Dataset):
    def __init__(self,dataset_info_csv_path,BeyesGuassian_pt_dir,group):
        super(CDP_MIL_WSI_Dataset,self).__init__(dataset_info_csv_path,group)
        self.slide_path_list = [os.path.join(BeyesGuassian_pt_dir,os.path.basename(slide_path).replace('.pt', '_bayesian_gaussian.pt')) for slide_path in self.slide_path_list]
        

    
class LONG_MIL_WSI_Dataset(WSI_Dataset):
    def __init__(self,dataset_info_csv_path,h5_csv_path,group):
        super(LONG_MIL_WSI_Dataset,self).__init__(dataset_info_csv_path,group)
        self.h5_path_list = pd.read_csv(h5_csv_path)['h5_path'].dropna().values

    def __getitem__(self, idx):
        slide_path = self.slide_path_list[idx]
        slide_name = os.path.basename(slide_path).replace('.pt','')
        h5_path = self._find_h5_path_by_slide_name(slide_name, self.h5_path_list)
        print(h5_path)
        h5_file = h5py.File(h5_path, 'r')
        coords = torch.from_numpy(np.array(h5_file['coords']))
        label = int(self.labels_list[idx])
        label = torch.tensor(label)
        feat = torch.load(slide_path) 
        if len(feat.shape) == 3:
            feat = feat.squeeze(0) # (N,D)
        if len(coords.shape) == 3:
            coords = coords.squeeze(0) # (N,2)
        feat_with_coords = torch.cat([feat, coords], dim=-1) # (N,D+2) 
        return feat_with_coords,label 
    
    def _find_h5_path_by_slide_name(self, slide_name, h5_paths_list):
        h5_dict = {os.path.basename(h5_path).replace('.h5', ''): h5_path for h5_path in h5_paths_list}
        return h5_dict.get(slide_name, None)
