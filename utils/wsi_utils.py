import openslide
import numpy as np
import cv2
import pandas as pd
import math
import torch
from torch.utils.data import Dataset


def display_camlyon_info(wsi_path):
    # 打开 WSI 文件
    slide = openslide.open_slide(wsi_path)

    # 获取 WSI 的尺寸
    width, height = slide.dimensions
    print("WSI尺寸：{}x{}".format(width, height))

    # 获取 WSI 的 level 数量
    num_levels = slide.level_count
    dimensions = slide.level_dimensions
    print("WSI Level数量：", num_levels)
    print("WSI Level明细：", dimensions)
    
    #获取WSI元数据，读取相关信息
    if 'aperio.AppMag' in slide.properties.keys():
        level_0_magnification = int(slide.properties['aperio.AppMag'])
    elif 'openslide.mpp-x' in slide.properties.keys():
        level_0_magnification = 40 if int(np.floor(float(slide.properties['openslide.mpp-x']) * 10)) == 2 else 20
    else:
        level_0_magnification = 40
        
    #输出level0对应的放大倍数
    print("level_0对应的放大倍数为:",level_0_magnification)



def save_thumbnail(wsi_path,save_path,level=-1):
    # 打开 WSI 文件
    slide = openslide.open_slide(wsi_path)

    # 获取 WSI 的尺寸
    width, height = slide.dimensions

    # 获取 WSI 的 level 数量
    num_levels = slide.level_count

    # 选择一个 level，这里选择 level 5

    thumbnail = slide.get_thumbnail(slide.level_dimensions[level])
    thumbnail = np.array(thumbnail)
    thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, thumbnail)
    print("缩略图已保存为{}".format(save_path))


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