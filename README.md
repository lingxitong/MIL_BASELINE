<p align="center">
    <br>
    <img src="https://github.com/lingxitong/MIL_BASELINE/blob/main/lo.png" width="400"/>
    <br>
<!-- <p>
<div align="center">
  <img alt="docs" src="https://github.com/salesforce/LAVIS/actions/workflows/docs.yaml/badge.svg"/>
  <a href="https://opensource.org/licenses/BSD-3-Clause">
  <img alt="license" src="https://img.shields.io/badge/License-BSD_3--Clause-blue.svg"/>
  </a> 
</div> -->

# MIL_BASELINE 
A library that integrates different MIL methods into a unified framework



## :white_check_mark: **UPDATES INFO**
**2024-07-20**
- fix bug of all MIL-models expect DTFD-MIL

**2024-07-21**
- fix bug of DTFD-MIL
- fix bug of test_mil.py

**2024-07-27**
- fix bug of plip-transforms

## :memo: **Overall Introduction**
### :bookmark: Library Introduction
* A library that integrates different MIL methods into a unified framework
* A library that integrates different Datasets into a unified Interface
* A library that provides different Datasets-Split-Methods which commonly used
* A library that easily extend by following a uniform definition

### :bulb: Dataset Uniform Interface
* User only need to provide the following csvs whether Public/Private Dataset<br/>
  `/datasets/example_Dataset.csv`
  
### :closed_umbrella: Supported Dataset-Split-Method
* User-difined Train-Val-Test split
* Train-Val split with K-fold
* Train-Val-Test split with K-fold

### :triangular_ruler: Feature Encoder
* R50 [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) (CVPR 2016)
* VIT-S [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929) (ICLR 2021)
* CTRANSPATH [Transformer-Based Self-supervised Learning for Histopathological Image Classification](https://link.springer.com/chapter/10.1007/978-3-030-87237-3_18) (MIA 2023)
* PLIP [A visual–language foundation model for pathology image analysis using medical Twitter](https://www.nature.com/articles/s41591-023-02504-3) (NAT MED 2023)
* CONCH [A visual-language foundation model for computational pathology](https://www.nature.com/articles/s41591-024-02856-4) (NAT MED 2024)
* UNI [Towards a general-purpose foundation model for computational pathology](https://www.nature.com/articles/s41591-024-02857-3) (NAT MED 2024)
* GIG [A whole-slide foundation model for digital pathology from real-world data](https://www.nature.com/articles/s41586-024-07441-w) (NAT 2024)
  
###  :gem: Implementated NetWork
* MEAN_MIL
* MAX_MIL
* AB_MIL [Attention-based Deep Multiple Instance Learning](https://arxiv.org/abs/1802.04712) (ICML 2018)
* TRANS_MIL [Transformer based Correlated Multiple Instance Learning for WSI Classification](https://arxiv.org/abs/2106.00908) (NeurIPS 2021)
* DS_MIL [Dual-stream MIL Network for WSI Classification with Self-supervised Contrastive Learning](https://arxiv.org/abs/2011.08939) (CVPR 2021)
* CLAM_MIL [Data Efficient and Weakly Supervised Computational Pathology on WSI](https://arxiv.org/abs/2004.09666) (NAT BIOMED ENG 2021)
* DTFD_MIL [Double-Tier Feature Distillation MIL for Histopathology WSI Classification](https://arxiv.org/abs/2203.12081) (CVPR 2022)
* RRT_MIL [Towards Foundation Model-Level Performance in Computational Pathology](https://arxiv.org/abs/2402.17228) (CVPR 2024)
* WIKG_MIL [Dynamic Graph Representation with Knowledge-aware Attention for WSI Analysis](https://arxiv.org/abs/2403.07719) (CVPR 2024)
* SC_MIL [Sparse Context-aware MIL for Predicting Cancer Survival Probability](https://arxiv.org/abs/2407.00664) (MICCAI 2024) UD...
* RET_MIL [Retentive Multiple Instance Learning for Histopathological WSI Classification](https://arxiv.org/abs/2403.10858) (MICCAI 2024) UD...
* UPDATING...

### ☑️  Implementated Metrics
* `AUC`: macro,micro,weighed (same when 2-classes)
* `F1,PRE,RECALL`: macro,micro,weighed
* `ACC,BACC`: BACC is macro-RECALL
* `KAPPLE`: linear,quadratic
* `Confusion_Mat`
* `FROC`
* `Patch-level-metrics`


## :orange_book: Let's Begin Now
### 🔨 **Code Framework**
- `/configs:` MIL_BASELINE defines MIL models through a YAML configuration file.
- `/modules:` Defined the network architectures of different MIL models.
- `/process:` Defined the training frameworks for different MIL models.
- `/feature_extracter:` Supports different feature extractors.
- `/datasets:` User-Datasets path information.
- `/utils:` Framework's utility scripts.
- `/train_mil.py:` Train Entry function of the framework.
- `/test_mil.py:` Test Entry function of the framework.

### 📁 **Dataset Pre-Process**
####  :egg: **Feature Extracter**
- OpenSlide supported formats and SDPC formats
- R50 and VIT-S are supported directly.
- PLIP/UNI/COUCH/TRANSPATH/GIG are supported by push the model-weights in `/feature_extracter/PLIP` `/feature_extracter/UNI` `/feature_extracter/COUCH ` `/feature_extracter/CTRANSPATH` `/feature_extracter/GIG`.
- To permform feature extracter </br>
  - First, you should get h5 file </br>
  `python /feature_extracter/CLAM/create_patches_fp.py --source /path/to/your/slide_dir --save_dir /path/to/your/save_dr --preset /feature_extractor/CLAM/presets/bwh_biopsy.csv --patch_level your_path_level --patch_size your_patch_size --seg --patch` 
  - Second, you should get pt file </br>
    `CUDA_VISIBLE_DEVICES=your_cuda_id python /feature_extractor/CLAM/extract_features_fp.py --data_h5_dir /path/to/your/h5_save_dir --data_slide_dir /path/to/your/slide_dir --csv_path /path/to/your/h5_save_dir/process_list_autogen.csv --feat_dir path/to/your/pt_save_dir --batch_size your_bach_size --slide_ext your_ext --backbone your_backbone --target_patch_size your_target_patch_size --model_dir your_backbone_dir` 
    - backbone : resnet50_imagenet/vit_s_imagenet/plip/uni
    - ext : .svs/.tif/.ndpi/......./.sdpc
    - target_patch_size : for example, vit_s need 224 as input
    - ctranspath needs timm== timm-0.5.4 / gig needs timm=>1.0.3 / uni,conch... need timm==0.9.16 

#### :custard: **Dataset-Csv Construction**
- You should construct a csv-file like the format of `/datasets/example_Dataset.csv`

#### :cookie: **Dataset-Csv Construction**
- You can use the dataset-split-scripts to perform different data-split.


### :fire: **Train/Test MIL**
#### :8ball: **Yaml Config**
- you can config the yaml-file in `/configs`
- for example, `/configs/AB_MIL.yaml`, A detailed explanation has been written in  `/configs/AB_MIL.yaml`
- Then, `/train_mil.py` will help you !
- `/test_mil.py` will help you test pretrained model !


### :sparkles: **Git Pull**
- Personal experience is limited, and code submissions are welcome !!
