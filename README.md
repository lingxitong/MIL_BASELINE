# MIL_BASELINE
A library that integrates different MIL methods into a unified framework

## Library Introduction
* A library that integrates different MIL methods into a unified framework
* A library that integrates different Datasets into a unified Interface
* A library that provides different Datasets-Split-Methods which commonly used
* A library that easily extend by following a uniform definition

## Dataset Uniform Interface
* User only need to provide the following csvs whether Public/Private Dataset
  
## Supported Dataset-Split-Method
* User-difined Train-Val-Test split
* Train-Test split with K-fold
* Train-Val-Test split with K-fold

## Feature Encoder
* R50
* VIT-S
* PLIP
* UNI
  
## Implementated NetWork
* MEAN_MIL
* MAX_MIL
* AB_MIL [Attention-based Deep Multiple Instance Learning](https://arxiv.org/abs/1802.04712) (ICML 2018) 
* TRANS_MIL [Transformer based Correlated Multiple Instance Learning for WSI Classification](https://arxiv.org/abs/2106.00908) (NeurIPS 2021)
* DS_MIL [Dual-stream MIL Network for WSI Classification with Self-supervised Contrastive Learning](https://arxiv.org/abs/2011.08939) (CVPR 2021)
* CLAM_MIL [Data Efficient and Weakly Supervised Computational Pathology on WSI](https://arxiv.org/abs/2004.09666) (NAT BIOMED ENG 2021)
* WENO_MIL [Bi-directional Weakly Supervised Knowledge Distillation for WSI Classification](https://arxiv.org/abs/2210.03664) (NeurIPS 2022)
* DTFD_MIL [Double-Tier Feature Distillation MIL for Histopathology WSI Classification](https://arxiv.org/abs/2203.12081) (CVPR 2022)
* MHIM_MIL [MIL Framework with Masked Hard Instance Mining for WSI Classification](https://arxiv.org/abs/2307.15254) (ICCV 2023)
* IB_MIL [Interventional Bag Multi-Instance Learning On Whole-Slide Pathological Images](https://arxiv.org/abs/2303.06873) (CVPR 2023)
* RRT_MIL [Towards Foundation Model-Level Performance in Computational Pathology](https://arxiv.org/abs/2402.17228) (CVPR 2024)
* WIKG_MIL [Dynamic Graph Representation with Knowledge-aware Attention for WSI Analysis](https://arxiv.org/abs/2403.07719) (CVPR 2024)
* UPDATING...

## :orange_book: Let's Begin Now
### 🔨 **Code Framework**
- `/configs:` MIL_BASELINE defines MIL models through a YAML configuration file.
- `/modules:` Defined the network architectures of different MIL models.
- `/process:` Defined the training frameworks for different MIL models.
- `/feature_extracter:` Supports different feature extractors.
- `/datasets:` User-Datasets path information.
- `/utils:` Framework's utility scripts.
- `/train_mil.py:` Entry function of the framework.

### 📁 **Dataset Pre-Process**
#### **Feature Extracter**
- OpenSlide supported formats and SDPC formats
- R50 and VIT-S are supported directly.
- PLIP and UNI are supported by push the model-weights in `/feature_extracter/PLIP` and `/feature_extracter/UNI`.
- To permform feature extracter </br>
  - First, you should get h5 file </br>
`python /feature_extracter/CLAM/create_patches_fp.py --source /path/to/your/slide_dir --save_dir /path/to/your/save_dr --preset /feature_extractor/CLAM/presets/bwh_biopsy.csv --patch_level your_path_level --patch_size your_patch_size --seg --patch` 
  - Second, you should get pt file </br>
`CUDA_VISIBLE_DEVICES=your_cuda_id python /feature_extractor/CLAM/extract_features_fp.py --data_h5_dir /path/to/your/h5_save_dir --data_slide_dir /path/to/your/slide_dir --csv_path /path/to/your/h5_save_dir/process_list_autogen.csv --feat_dir path/to/your/pt_save_dir --batch_size your_bach_size --slide_ext your_ext --backbone your_backbone --target_patch_size your_target_patch_size`
      - backbone : resnet50_imagenet/vit_s_imagenet/plip/uni
      - ext : .svs/.tif/.ndpi/......./.sdpc
      - target_patch_size : for example, vit_s need 224 as input

#### **Dataset-Csv Construction**
- You should construct a csv-file like the format of `/datasets/example.csv`
