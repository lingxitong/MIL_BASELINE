import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(grandparent_dir)
import pandas as pd
from feature_extractor.clam_base_utils.utils.utils import *
from PIL import Image
from math import floor
import matplotlib.pyplot as plt
from feature_extractor.clam_base_utils.clam_datasets.wsi_dataset import Wsi_Region
from feature_extractor.clam_base_utils.utils.transform_utils import get_eval_transforms
import h5py
from feature_extractor.clam_base_utils.wsi_core.WholeSlideImage import WholeSlideImage
from scipy.stats import percentileofscore
import math
from feature_extractor.clam_base_utils.utils.file_utils import save_hdf5
from feature_extractor.clam_base_utils.utils.model_utils import feature_extractor_adapter
from scipy.stats import percentileofscore
from feature_extractor.clam_base_utils.utils.constants import MODEL2CONSTANTS
from tqdm import tqdm
from utils.loop_utils import get_cam_1d

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def dtfd_infer_single_slide(model_list,features, **kwargs):
    total_instance = kwargs['total_instance']
    num_Group = kwargs['num_Group']
    grad_clipping,distill = kwargs['grad_clipping'],kwargs['distill']
    instance_per_group = total_instance // num_Group
    classifier,attention,dimReduction,attCls = model_list
    classifier.eval()
    attention.eval()
    dimReduction.eval()
    attCls.eval()
    bag = features
    bag = bag.to(device)
    slide_sub_preds = []
    slide_sub_labels = []
    slide_pseudo_feat = []
    inputs_pseudo_bags=torch.chunk(bag.squeeze(0), num_Group,dim=0)
    for subFeat_tensor in inputs_pseudo_bags:
        subFeat_tensor=subFeat_tensor.to(device)
        with torch.no_grad():
            tmidFeat = dimReduction(subFeat_tensor)
            tAA = attention(tmidFeat).squeeze(0)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  # n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0, keepdim=True)  # 1 x fs
            tPredict = classifier(tattFeat_tensor)  # 1 x 2
        tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
        tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
        patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
        patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
        patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

        _, sort_idx = torch.sort(patch_pred_softmax[:,-1], descending=True)
        topk_idx_max = sort_idx[:instance_per_group].long()
        topk_idx_min = sort_idx[-instance_per_group:].long()
        topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
        MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)   
        max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
        af_inst_feat = tattFeat_tensor

        if distill == 'MaxMinS':
            slide_pseudo_feat.append(MaxMin_inst_feat)
        elif distill == 'MaxS':
            slide_pseudo_feat.append(max_inst_feat)
        elif distill == 'AFS':
            slide_pseudo_feat.append(af_inst_feat)
        slide_sub_preds.append(tPredict)

    slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
    # gSlidePred = torch.softmax(attCls(slide_pseudo_feat), dim=1)
    forward_return = attCls(slide_pseudo_feat, return_WSI_attn = True)  
    A = forward_return['WSI_attn']
    A = A.view(-1, 1).detach().cpu().numpy()
    return A

def infer_single_slide(model, model_name, features, attention_kwargs = {}):
    if model_name == 'DTFD_MIL':
        return dtfd_infer_single_slide(model, features,**attention_kwargs)
    features = features.to(device)
    model.eval()
    with torch.inference_mode():
        A = model(features,return_WSI_attn=True)['WSI_attn']
        A = A.view(-1, 1).cpu().numpy()
    return  A

# def score2percentile(score, ref):
#     percentile = percentileofscore(ref, score)
#     return percentile

def score2percentile(score, ref):
    percentile = percentileofscore(ref.squeeze(), score.squeeze())
    return percentile

def drawHeatmap(scores, coords, slide_path=None, wsi_object=None, vis_level = -1, **kwargs):
    if wsi_object is None:
        wsi_object = WholeSlideImage(slide_path)
        print(wsi_object.name)
    
    wsi = wsi_object.getOpenSlide()
    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)
    
    heatmap = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, **kwargs)
    return heatmap

def initialize_wsi(wsi_path, seg_mask_path=None, seg_params=None, filter_params=None):
    wsi_object = WholeSlideImage(wsi_path)
    if seg_params['seg_level'] < 0:
        best_level = wsi_object.wsi.get_best_level_for_downsample(32)
        seg_params['seg_level'] = best_level

    wsi_object.segmentTissue(**seg_params, filter_params=filter_params)
    wsi_object.saveSegmentation(seg_mask_path)
    return wsi_object

def compute_from_patches(wsi_object, feature_extractor_name=None, feature_extractor_transforms=None, feature_extractor=None,mil_model=None, mil_model_name = None, batch_size=512,  
    attn_save_path=None, ref_scores=None, feat_save_path=None,attention_kwargs = {}, **wsi_kwargs):    
    top_left = wsi_kwargs['top_left']
    bot_right = wsi_kwargs['bot_right']
    patch_size = wsi_kwargs['patch_size'] 
    
    roi_dataset = Wsi_Region(wsi_object, t=feature_extractor_transforms, **wsi_kwargs)
    roi_loader = get_simple_loader(roi_dataset, batch_size=batch_size, num_workers=8)
    print('total number of patches to process: ', len(roi_dataset))
    num_batches = len(roi_loader)
    print('number of batches: ', num_batches)
    mode = "w"
    for idx, (roi, coords) in enumerate(tqdm(roi_loader)):
        roi = roi.to(device)
        coords = coords.numpy()
        
        with torch.inference_mode():
            features = feature_extractor_adapter(feature_extractor, roi, feature_extractor_name)t
            if len(features.shape) == 3:
                features = features.squeeze(0)

            if attn_save_path is not None:
                # A = mil_model(features, return_WSI_attn=True)
                # A = A.view(-1, 1).cpu().numpy()
                A = infer_single_slide(mil_model, mil_model_name, features, attention_kwargs)

                if ref_scores is not None:
                    for score_idx in range(len(A)):
                        A[score_idx] = score2percentile(A[score_idx], ref_scores)

                asset_dict = {'attention_scores': A, 'coords': coords}
                save_path = save_hdf5(attn_save_path, asset_dict, mode=mode)
    
        if feat_save_path is not None:
            asset_dict = {'features': features.cpu().numpy(), 'coords': coords}
            save_hdf5(feat_save_path, asset_dict, mode=mode)

        mode = "a"
    return attn_save_path, feat_save_path, wsi_object
