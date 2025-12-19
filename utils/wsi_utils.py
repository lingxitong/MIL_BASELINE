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

        # adapting to different feature file types(https://github.com/mahmoodlab/TRIDENT)
        if slide_path.endswith('.h5'):
            with h5py.File(slide_path, 'r') as h5_file:
                feat = h5_file['features'][:]
                feat = torch.from_numpy(feat)
        else:
            feat = torch.load(slide_path)
            # Handle dictionary format (e.g., {'feats': tensor, 'coords': tensor})
            if isinstance(feat, dict):
                if 'feats' in feat:
                    feat = feat['feats']
                elif 'features' in feat:
                    feat = feat['features']
                else:
                    raise ValueError(f"Unknown dict format in {slide_path}, keys: {list(feat.keys())}")
        if len(feat.shape) == 3:
            feat = feat.squeeze(0)
        return feat,label

    def is_None_Dataset(self):
        return (self.__len__() == 0)    
    
    def is_with_labels(self):
        return (len(self.labels_list) != 0)
    
    def get_balanced_sampler(self, replacement=True):
        from collections import Counter
        from torch.utils.data import WeightedRandomSampler

        label_counts = Counter(self.labels_list)
        weights = [1.0 / label_counts[label] for label in self.labels_list]
        num_samples = len(self.labels_list)

        sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples, replacement=replacement)
        return sampler

    
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

class SC_MIL_WSI_Dataset(WSI_Dataset):
    """
    Dataset for SC_MIL that can work without h5 files
    If h5_csv_path is provided, uses coords from h5 files (like LONG_MIL)
    If h5_csv_path is None, generates dummy coords based on patch indices
    """
    def __init__(self, dataset_info_csv_path, h5_csv_path=None, group='train', use_dummy_coords=True):
        super(SC_MIL_WSI_Dataset, self).__init__(dataset_info_csv_path, group)
        self.use_dummy_coords = use_dummy_coords
        
        if h5_csv_path is not None and os.path.exists(h5_csv_path):
            # Use h5 files if available
            self.h5_path_list = pd.read_csv(h5_csv_path)['h5_path'].dropna().values
            self.use_dummy_coords = False
        else:
            # Use dummy coords
            self.h5_path_list = None
            self.use_dummy_coords = True
            if h5_csv_path is not None:
                print(f"⚠️  Warning: h5_csv_path '{h5_csv_path}' not found. Using dummy coords for SC_MIL.")
    
    def _generate_dummy_coords(self, num_patches):
        """
        Generate dummy coordinates based on patch indices
        Assumes patches are arranged in a grid-like structure
        """
        # Estimate grid size (assume roughly square)
        grid_size = int(np.ceil(np.sqrt(num_patches)))
        
        # Generate 2D grid coordinates
        coords = []
        for i in range(num_patches):
            row = i // grid_size
            col = i % grid_size
            coords.append([col, row])  # (x, y) format
        
        return np.array(coords, dtype=np.float32)
    
    def __getitem__(self, idx):
        slide_path = self.slide_path_list[idx]
        slide_name = os.path.basename(slide_path).replace('.pt', '')
        label = int(self.labels_list[idx])
        label = torch.tensor(label)
        
        # Load features
        feat = torch.load(slide_path)
        if len(feat.shape) == 3:
            feat = feat.squeeze(0)  # (N, D)
        
        num_patches = feat.shape[0]
        
        # Get coords
        if self.use_dummy_coords:
            # Generate dummy coords
            coords_np = self._generate_dummy_coords(num_patches)
            coords = torch.from_numpy(coords_np)  # (N, 2)
        else:
            # Load from h5 file
            h5_path = self._find_h5_path_by_slide_name(slide_name, self.h5_path_list)
            if h5_path is None:
                # Fallback to dummy coords if h5 not found
                coords_np = self._generate_dummy_coords(num_patches)
                coords = torch.from_numpy(coords_np)
            else:
                h5_file = h5py.File(h5_path, 'r')
                coords = torch.from_numpy(np.array(h5_file['coords']))
                h5_file.close()
                if len(coords.shape) == 3:
                    coords = coords.squeeze(0)  # (N, 2)
        
        # Concatenate features and coords
        feat_with_coords = torch.cat([feat, coords], dim=-1)  # (N, D+2)
        return feat_with_coords, label
    
    def _find_h5_path_by_slide_name(self, slide_name, h5_paths_list):
        if h5_paths_list is None:
            return None
        h5_dict = {os.path.basename(h5_path).replace('.h5', ''): h5_path for h5_path in h5_paths_list}
        return h5_dict.get(slide_name, None)
