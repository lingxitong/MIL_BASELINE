# MIL_BASELINE <p align="center">
  <a href='https://scholar.google.com/citations?user=5lNlpagAAAAJ&hl=en'>
  <img src='https://img.shields.io/badge/Arxiv-2404.19759-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a> 
  <a href='https://github.com/lingxitong/MIL_BASELINE'>
  <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'></a> 
</p>

<img src="https://github.com/lingxitong/MIL_BASELINE/blob/main/Logo.png"  width="290px" align="right" />
With the rapid advancement of computational power and artificial intelligence technologies, computational pathology has gradually been regarded as the most promising and transformative auxiliary diagnostic paradigm in the field of pathology diagnosis. However, due to the fact that pathological images often consist of hundreds of billions of pixels, traditional natural image analysis methods face significant computational and technical limitations. Multiple Instance Learning (MIL) is one of the most successful and widely adopted paradigms for addressing computational pathology analysis. However, current MIL methods often adopt different frameworks and structures, which poses challenges for subsequent research and reproducibility. We have developed the MIL_BASELINE library with the aim of providing a fundamental and simple template for Multiple Instance Learning applications.


<details>
<summary>News of MIL-Baseline</summary>

**2025-1-10**
fix bug of MIL_BASELINE, update visualization tools, add new MIL methods, add new dataset split methods

**2024-11-24**
update `mil-finetuning` (gate_ab_mil,ab_mil) for rrt_mil

**2024-10-12**
fix bug of `Ctranspath` feature encoder
  
**2024-10-02**
add `FR_MIL` Implement

**2024-08-20**
fix bug of early-stop
  
**2024-07-27**
fix bug of plip-transforms
  
**2024-07-21**
fix bug of DTFD-MIL
fix bug of test_mil.py

**2024-07-20**
fix bug of all MIL-models expect DTFD-MIL
</details>
  
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
* User-difined Train-Val split
* User-difined Train-Test split
* Train-Val split with K-fold
* Train-Val-Test split with K-fold
* Train-Val with K-fold then test
* The difference between the different splits is in   `/split_scripts/README.md`

### :triangular_ruler: Feature Encoder
* R50 [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) (CVPR 2016)
* VIT-S [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929) (ICLR 2021)
* CTRANSPATH [Transformer-Based Self-supervised Learning for Histopathological Image Classification](https://link.springer.com/chapter/10.1007/978-3-030-87237-3_18) (MIA 2023)
* PLIP [A visual–language foundation model for pathology image analysis using medical Twitter](https://www.nature.com/articles/s41591-023-02504-3) (NAT MED 2023)
* CONCH [A visual-language foundation model for computational pathology](https://www.nature.com/articles/s41591-024-02856-4) (NAT MED 2024)
* UNI [Towards a general-purpose foundation model for computational pathology](https://www.nature.com/articles/s41591-024-02857-3) (NAT MED 2024)
* GIGAPATH [A whole-slide foundation model for digital pathology from real-world data](https://www.nature.com/articles/s41586-024-07441-w) (NAT 2024)
* VIRCHOW [A foundation model for clinical-grade computational pathology and rare cancers detection](https://www.nature.com/articles/s41591-024-03141-0) (NAT 2024)
* VIRCHOW-V2 [Virchow2: Scaling Self-Supervised Mixed Magnification Models in Pathology](https://arxiv.org/pdf/2408.00738) (arxiv 2024)
* CONCH-V1.5 [Multimodal Whole Slide Foundation Model for Pathology](https://arxiv.org/abs/2411.19666) (arxiv 2024)
* UPDATING...

###  :gem: Implementated NetWork
* MEAN_MIL
* MAX_MIL
* AB_MIL [Attention-based Deep Multiple Instance Learning](https://arxiv.org/abs/1802.04712) (ICML 2018)
* TRANS_MIL [Transformer based Correlated Multiple Instance Learning for WSI Classification](https://arxiv.org/abs/2106.00908) (NeurIPS 2021)
* DS_MIL [Dual-stream MIL Network for WSI Classification with Self-supervised Contrastive Learning](https://arxiv.org/abs/2011.08939) (CVPR 2021)
* CLAM_MIL [Data Efficient and Weakly Supervised Computational Pathology on WSI](https://arxiv.org/abs/2004.09666) (NAT BIOMED ENG 2021)
* DTFD_MIL [Double-Tier Feature Distillation MIL for Histopathology WSI Classification](https://arxiv.org/abs/2203.12081) (CVPR 2022)
* ADD_MIL [Additive MIL: Intrinsically Interpretable MIL for Pathology](https://arxiv.org/pdf/2206.01794) (NeurIPS 2022)
* ILRA_MIL [Exploring Low-rank Property in MIL for Whole Slide Image classification](https://openreview.net/pdf?id=01KmhBsEPFO) (ICLR 2023)
* WIKG_MIL [Dynamic Graph Representation with Knowledge-aware Attention for WSI Analysis](https://arxiv.org/abs/2403.07719) (CVPR 2024)
* AMD_MIL [Agent Aggregator with Mask Denoise Mechanism for Histopathology WSI Analysis](https://dl.acm.org/doi/10.1145/3664647.3681425) (MM 2024)
* FR-MIL [Distribution Re-calibration based MIL with Transformer for WSI Classification](https://ieeexplore.ieee.org/abstract/document/10640165) (TMI 2024)
* LONG_MIL [Scaling Long Contextual MIL for Histopathology WSI Analysis](https://arxiv.org/abs/2311.12885) (NeurIPS 2024) 
* DGR_MIL [Exploring Diverse Global Representation in MIL for WSI Classification](https://arxiv.org/abs/2407.03575) (ECCV 2024) 
* CDP_MIL [cDP-MIL: Robust Multiple Instance Learning via Cascaded Dirichlet Process](https://arxiv.org/abs/2407.11448) (ECCV 2024)
* CA_MIL [Context-Aware Multiple Instance Learning for WSI Classification](https://arxiv.org/pdf/2305.05314) (ICLR 2024)
* AC_MIL [Attention-Challenging Multiple Instance Learning for WSI Classification](https://arxiv.org/pdf/2311.07125) (ECCV 2024)
* UPDATING...

### ☑️  Implementated Metrics
* `AUC`: macro,micro,weighed (same when 2-classes)
* `F1,PRE,RECALL`: macro,micro,weighed
* `ACC,BACC`: BACC is macro-RECALL
* `KAPPA`: linear,quadratic
* `Confusion_Mat`


## :orange_book: Let's Begin Now
### 🔨 **Code Framework**
`MIL_BASELINE` is constructed by the following parts：
- `/configs:` MIL_BASELINE defines MIL models through a YAML configuration file.
- `/modules:` Defined the network architectures of different MIL models.
- `/process:` Defined the training frameworks for different MIL models.
- `/feature_extractor:` Supports different feature extractors.
- `/split_scripts:` Supports different dataset split methods.
- `/vis_scripts:` Visualization scripts for TSNE and Attention.
- `/datasets:` User-Datasets path information.
- `/utils:` Framework's utility scripts.
- `train_mil.py:` Train Entry function of the framework.
- `test_mil.py:` Test Entry function of the framework.


### 📁 **Dataset Pre-Process**
#### **Feature Extracter**
Supported formats include `OpenSlide` and `SDPC` formats. The following backbones are supported: `R50, VIT-S, CTRANSPATH, PLIP, CONCH, UNI, GIGAPATH, VIRCHOW, VIRCHOW-V2 and CONCH-V1.5`. Detailed usage instructions can be found in `/feature_extractor/README.md`.

#### **Dataset-Csv Construction**
You should construct a csv-file like the format of `/datasets/example_Dataset.csv`

#### **Dataset-Split Construction**
You can use the dataset-split-scripts to perform different dataset-split, the detailed split method descriptions are in `/split_scripts/README.md`.


### :fire: **Train/Test MIL**
#### **Yaml Config**
You can config the yaml-file in `/configs`. For example, `/configs/AB_MIL.yaml`, A detailed explanation has been written in  `/configs/AB_MIL.yaml`. 
#### **Train & Test**
Then, `/train_mil.py` will help you like this:
``` shell
python train_mil.py --yaml_path /configs/AB_MIL.yaml 
```
We also support dynamic parameter passing, and you can pass any parameters that exist in the `/configs/AB_MIL.yaml` file, for example:
``` shell
python train_mil.py --yaml_path /configs/AB_MIL.yaml --options General.seed=2024 General.num_epochs=20 Model.in_dim=768
```
The `/test_mil.py` will help you test pretrained MIL models like this:
``` shell
python test_mil.py --yaml_path /configs/AB_MIL.yaml --test_dataset_csv /your/test_csv/path --model_weight_path /your/model_weights/path --test_log_dir /your/test/log/dir
```
You should ensure the `--test_dataset_csv` contains the column of `test_slide_path` which contains the `/path/to/your_pt.pt`. If `--test_dataset_csv` also contains the 'test_slide_label' column, the metrics will be calculated and written to logs.


### :fountain: **Visualization**
You can easily visualize the dimensionality reduction map of the features from the trained MIL model and the distribution of attention scores (or importance scores) by `/vis_scripts/draw_feature_map.py` and `/vis_scripts/draw_attention_map.py`. We have implemented standardized global feature and attention score output interfaces for most models, making the above visualization scripts compatible with most MIL model in the library. The detailed usage instructions are in `/vis_scripts/README.md`.


### :card_file_box: **Tips**
You can use `MIL_BASELINE` as a package, but you should rename the folder `MIL_BASELINE-main` to `MIL_BASELINE`.

### :beers: **Acknowledgement**
Thanks to the following repositories for inspiring this repository
  - https://github.com/mahmoodlab/CLAM
  - https://github.com/WonderLandxD/opensdpc
  - https://github.com/RenaoYan/Build-Patch-for-Sdpc
  - https://github.com/DearCaat/MHIM-MIL

### :sparkles: **Git Pull**
Personal experience is limited, and code submissions are welcome.
