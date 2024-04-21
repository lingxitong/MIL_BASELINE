'''
检查C16和C17是否有相同的WSI
'''
import os
import glob
import openslide
from tqdm import tqdm
import pandas as pd
def get_dataset_paths(root_dir_list,suffix='.tif'):
    dataset_paths = []
    for root_dir in root_dir_list:
        dataset_paths += glob.glob(os.path.join(root_dir,'*'+suffix ))
    return dataset_paths
    
def check_2_wsi(wsi_1_path,wsi_2_path):
    wsi_1 = openslide.OpenSlide(wsi_1_path)
    wsi_2 = openslide.OpenSlide(wsi_2_path)
    wsi_1_dim = wsi_1.level_dimensions[0]
    wsi_2_dim = wsi_2.level_dimensions[0]
    if wsi_1_dim == wsi_2_dim:
        wsi_1_dim_1 = wsi_1.level_dimensions[1]
        wsi_2_dim_1 = wsi_2.level_dimensions[1]
        if wsi_1_dim_1 == wsi_2_dim_1:
            print(f'{wsi_1_path} and {wsi_2_path} may be the same WSI')
            return (wsi_1_path,wsi_2_path)
        return None
    return None
    wsi_1.close()
    wsi_2.close()
    



if __name__ == '__main__':
    # C16_root_dir = '/data_sdb/CAMELYON16'
    C16_train_tumor = '/data_sdb/CAMELYON16/training/tumor'
    C16_train_normal = '/data_sdb/CAMELYON16/training/normal'
    C16_test = '/data_sdb/CAMELYON16/testing/images'
    C16_tif_paths = get_dataset_paths([C16_train_tumor,C16_train_normal,C16_test],'tif')

    # C17_root_dir = '/data_sdf/CAMELYON17_TIF'
    C17_train = '/data_sdf/CAMELYON17_TIF/training'
    C17_test = '/data_sdf/CAMELYON17_TIF/testing'
    C17_tif_paths = get_dataset_paths([C17_train,C17_test],'tif')
    C16_same = []
    C17_same = []
    for C16_path in tqdm(C16_tif_paths):
        for C17_path in C17_tif_paths:
            same_path = check_2_wsi(C16_path,C17_path)
            if same_path:
                C16_same.append(same_path[0])
                C17_same.append(same_path[1])
    same_df = pd.DataFrame({'C16_same': C16_same, 'C17_same': C17_same})
    same_df.to_csv('/data_sda/lxt/CAMELYON_BENCHMARK/utils/same_wsi.csv',index=False)