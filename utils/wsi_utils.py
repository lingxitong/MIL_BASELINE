import openslide
import numpy as np
import cv2
import pandas as pd
import math
import torch
from torch.utils.data import Dataset




class WSI_Dataset(Dataset):
    def __init__(self,dataset_info_csv_path,group):
        assert group in ['train','val','test'], 'group must be in [train,val,test]'
        self.dataset_info_csv_path = dataset_info_csv_path
        self.dataset_df = pd.read_csv(self.dataset_info_csv_path)
        self.slide_path_list = self.dataset_df[group+'_slide_path'].values
        self.labels_list = self.dataset_df[group+'_label'].values
        self.slide_path_list = [x for x in self.slide_path_list if isinstance(x,str)]
        self.labels_list = [x for x in self.labels_list if not math.isnan(x)]

    def __len__(self):
        return len(self.slide_path_list)
    def __getitem__(self, idx):
        slide_path = self.slide_path_list[idx]
        label = int(self.labels_list[idx])
        label = torch.tensor(label)
        feat = torch.load(slide_path)
        return feat,label