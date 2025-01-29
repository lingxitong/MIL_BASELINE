## :ledger: **Feature Extractor Pipeline**
### :key: **Firstly you should get the .h5 files which save the patch coords information**
You can generate `.h5` files by this command:
``` shell
python create_h5_patches.py --source /dir/to/your/slides --ext_list your_ext_list  --level_or_magnification_control level/magnification --patch_level your_patch_level --magnification your_magnification --patch_size your_patch_size --step_size your_step_size --save_dir /dir/to/your/save_dir --preset your_preset_csv_path --patch --seg --stitch
```
The meanings of the important parameters are as follows:

- `--source` : The root directory of the `WSIs` to be processed, which can contain subdirectories. When it contains subdirs, you should ensure there is no 
WSIs with the same name. If you have same-named WSI in different subdirs, you should use each subdir as the `--source` to perform the process.
- `--ext_list` : All possible WSI file extensions list, for example `['.svs','.sdpc']`
- `--level_or_magnification_control` : use which to control patching process, `level` or `magnification`

- `--patch_level` : patch level for patching process. You had better ensure the patch_level you set is lower than or equal to the max level of the WSIs. If not, the patching process will skip the WSIs that does not meet the conditions. It only works when `--level_or_magnification_control` is set to `level`.

- `--magnification` : magnification for patching process. You had better ensure the magnification you set is lower than or equal to the original magnification of the WSIs. If not, the patching process will skip the WSIs that does not meet the conditions. It only works when `--level_or_magnification_control` is set to `magnification`

- `--patch_size` : patch size for patching process in the level or magnification you set, for example, you set `--patch_size 256`,`--patch_level 2` and `--level_or_magnification_control level`,the patch_size in the original level will be `256 * downsample_factor^2`. `downsample_factor` means the ratio of the length between adjacent WSI levels.
`--step_size` : step size for patching process.
`--save_dir` : h5 and other files save dir.
`--preset` : select the preset from `/feature_extractor/presets`, which contains some preset parameters for patching process.

Example of `--source` 
```bash
source/
	├── slide_1.svs
	├── slide_2.svs
	└── /sub_dir1
        ├── slide_3.sdpc
        ├── slide_4.sdpc
        ├── /sub_dir2 
            ├── slide_5.svs
            ├── ...

```
Results of `--save_dir` 
```bash
save_dir/
	├── masks
    		├── slide_1.png
    		├── slide_2.png
    		└── ...
	├── patches
    		├── slide_1.h5
    		├── slide_2.h5
    		└── ...
	├── stitches
    		├── slide_1.png
    		├── slide_2.png
    		└── ...
	└── process_list_autogen.csv
```


### :lock: **Then you should get the .pt files**
You can generate `.pt` files by this command:
``` shell
python create_pt_features.py --data_h5_dir your_data_h5_dir --feat_dir your_save_feat_dir --batch_size batch_size --target_patch_size your_target_patch_size --backbone your_backbone --pretrained_weights_dir /dir/to/backbone_weights
```
The meanings of the important parameters are as follows:

`--data_h5_dir` : `save_dir` in `create_h5_patches.py`.

`--feat_dir` : dir to save `.pt`.
`--batch_size` : batch_size for feature extracting process.

`--target_patch_size` : size for backbone model input, for example, you set `--patch_size 256` when you run the `create_h5_patches.py`, but you want to or the backbone model needs 224 as input_size, you can set `--target_patch_size 224`.

`--backbone` : use which backbone model to process feature extracting process. `['vit_s_imagenet','resnet50_imagenet','plip','conch','uni','ctranspath','gigapath','virchow','virchow_v2','conch_v1_5']` are support now.
`--pretrained_weights_dir` : pretrained weights dir for the backbone model you use. for example:
```bash
resnet50_imagenet_weights_dir/
	├──resnet50-19c8e357.pth
vit_s_imagenet_weightes_dir/
	├──pytorch_model.bin
ctranspath_weights_dir/
	├──ctranspath.pth 
plip_weightes_dir/
	├──pytorch_model.bin
conch_weights_dir/
	├──pytorch_model.bin 
conch_v1_5_weights_dir/
	├──conch_v1_5_pytorch_model.bin
uni_weightes_dir/
	├──pytorch_model.bin
gigapath_weights_dir/
	├──pytorch_model.bin 
virchow_weightes_dir/
	├──pytorch_model.bin
virchow_v2_weightes_dir/
	├──pytorch_model.bin

```
Model Download Link and timm-version:
- [resnet50_imagenet](https://download.pytorch.org/models/resnet50-19c8e357.pth)
- [vit_s_imagenet](https://huggingface.co/timm/vit_small_patch16_224.augreg_in21k_ft_in1k/tree/main)
- [ctranspath](https://drive.google.com/file/d/1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX/view?usp=sharing)/[chief_ctranspath](https://drive.google.com/drive/folders/1uRv9A1HuTW5m_pJoyMzdN31bE1i-tDaV) `custom-timm==0.5.4` [download-custom-timm==0.5.4](https://drive.google.com/file/d/1JV7aj9rKqGedXY1TdDfi3dP07022hcgZ/view)
- [plip](https://huggingface.co/vinid/plip/tree/main) `timm>=0.9.16`
- [conch](https://huggingface.co/MahmoodLab/CONCH) `timm>=0.9.16`
- [uni](https://huggingface.co/MahmoodLab/UNI) `timm>=0.9.16`
- [gigapath](https://huggingface.co/prov-gigapath/prov-gigapath/tree/main) `timm>=1.0.3`
- [virchow](https://huggingface.co/paige-ai/Virchow) `timm>=0.9.11`
- [virchow_v2](https://huggingface.co/paige-ai/Virchow2/tree/main) `timm>=0.9.11`
- [conch_v1_5](https://huggingface.co/MahmoodLab/TITAN) `timm==1.0.3,transformers==4.46.0,einops==0.6.1,einops-exts==0.0.4`

Results of `--feat_dir`
```bash
feat_dir/
    ├── h5_files
            ├── slide_1.h5
            ├── slide_2.h5
            └── ...
    └── pt_files
            ├── slide_1.pt
            ├── slide_2.pt
            └── ...
```
