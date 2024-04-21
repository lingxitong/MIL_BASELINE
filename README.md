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

## Feature Encoder
* R50
* PLIP
  
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
### **Code Framework**
- MIL_BASELINE defines MIL models through a YAML configuration file.
- Since different MIL models may have different training logic, MIL_BASELINE defines a separate training framework for each MIL model, thus achieving decoupling.
- In addition, it provides scripts for WSI feature extraction and data partitioning, supporting feature extractors like R50 and PLIP, making expansion very convenient.
- Dataset partitioning supports the most common TVT split and Train-Test with k-fold division.
- MIL_BASELINE provides a highly convenient and easily extendable framework for research on improvements to MIL models. Thank you for your 🥇 star.


  
