# CAMELYON17 提取h5
python create_patches_fp.py --source /data_sdd/lxt/GEM_WSI/postive --save_dir /data_sdd/lxt/GEM_PT_20X_PLIP/postive  --patch_level 1 --patch_size 256 --seg --patch

python create_patches_fp.py --source /data_sdf/CAMELYON17_TIF/testing --save_dir /data_sdf/CAMELYON_NEW/testing --preset /data_sda/lxt/CAMELYON_BENCHMARK/feature_extractor/CLAM/presets/bwh_biopsy.csv --patch_level 1 --patch_size 256 --seg --patch

20X下256的h5 preset是bwh_biopsy.csv


# CLAM提h5文件
patchsize是patch_level下的，最终h5会保存映射到level0的大小
比如patch_size=256,patch_level=1,如果是2倍下采样，那么h5中就是512大小的patch

# plip提取GEM特征
CUDA_VISIBLE_DEVICES=1 python extract_features_fp.py --data_h5_dir /data_sdd/lxt/GEM_PT_20X_PLIP/postive_patch --data_slide_dir /data_sdd/lxt/GEM_WSI/postive --csv_path /data_sdd/lxt/GEM_PT_20X_PLIP/postive_patch/process_list_autogen.csv --feat_dir /data_sdd/lxt/GEM_PT_20X_PLIP/postive_feat --batch_size 256 --slide_ext .ndpi --backbone plip --target_patch_size 224

CUDA_VISIBLE_DEVICES=1 python extract_features_fp.py --data_h5_dir /data_sdf/CAMELYON_NEW/CANMELYON17/H5/training --data_slide_dir /data_sdf/CAMELYON17_TIF/training --csv_path /data_sdf/CAMELYON_NEW/CANMELYON17/H5/training/process_list_autogen.csv --feat_dir /data_sdf/CAMELYON_NEW/training_feature_r50 --batch_size 512 --slide_ext .tif --backbone resnet50