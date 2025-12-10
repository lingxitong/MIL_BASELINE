"""
Mix augmentation utilities for MIL
Includes PseMix, InsMix, Mixup, RankMix, ReMix methods
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from scipy.spatial.distance import cdist


def mixup_data(features1, label1, features2, label2, alpha=1.0):
    """Standard Mixup method"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    # Handle bags of different sizes
    n1, n2 = features1.size(0), features2.size(0)
    max_n = max(n1, n2)
    
    # Resample to the same size
    if n1 < max_n:
        indices = torch.randint(0, n1, (max_n,))
        features1 = features1[indices]
    if n2 < max_n:
        indices = torch.randint(0, n2, (max_n,))
        features2 = features2[indices]
    
    mixed_features = lam * features1 + (1 - lam) * features2
    mixed_label = lam * label1 + (1 - lam) * label2
    
    return mixed_features, mixed_label, lam


def insmix_data(features1, label1, features2, label2, alpha=1.0):
    """Instance-level Mixup (InsMix)"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    n1, n2 = features1.size(0), features2.size(0)
    
    # Randomly select instances for mixing
    num_select1 = int(n1 * lam)
    num_select2 = n2 - num_select1
    
    if num_select1 > 0 and num_select1 < n1:
        indices1 = torch.randperm(n1)[:num_select1]
        selected1 = features1[indices1]
    elif num_select1 >= n1:
        selected1 = features1
    else:
        selected1 = torch.empty(0, features1.size(1), device=features1.device)
    
    if num_select2 > 0 and num_select2 < n2:
        indices2 = torch.randperm(n2)[:num_select2]
        selected2 = features2[indices2]
    elif num_select2 >= n2:
        selected2 = features2
    else:
        selected2 = torch.empty(0, features2.size(1), device=features2.device)
    
    if selected1.size(0) > 0 and selected2.size(0) > 0:
        mixed_features = torch.cat([selected1, selected2], dim=0)
        area_ratio = selected1.size(0) / mixed_features.size(0)
    elif selected1.size(0) > 0:
        mixed_features = selected1
        area_ratio = 1.0
    else:
        mixed_features = selected2
        area_ratio = 0.0
    
    mixed_label = area_ratio * label1 + (1 - area_ratio) * label2
    
    return mixed_features, mixed_label, area_ratio


def psebmix_data(features1, label1, features2, label2, n_pseudo_bags=30, alpha=1.0):
    """Pseudo-bag Mixup (PseMix)"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    device = features1.device
    
    # Simplified pseudo-bag division: based on k-means clustering
    def simple_pseudo_bag_divide(features, n_bags):
        n_instances = features.size(0)
        if n_instances < n_bags:
            n_bags = n_instances
        
        # Simple uniform division
        bag_assignments = torch.randint(0, n_bags, (n_instances,), device=features.device)
        return bag_assignments
    
    # Divide into pseudo-bags
    pseb_ind1 = simple_pseudo_bag_divide(features1, n_pseudo_bags)
    pseb_ind2 = simple_pseudo_bag_divide(features2, n_pseudo_bags)
    
    # Select number of pseudo-bags to mix
    n_select = int(n_pseudo_bags * lam)
    n_select = max(1, min(n_select, n_pseudo_bags-1))
    
    # Select pseudo-bags from both bags
    selected_psebs1 = torch.randperm(n_pseudo_bags)[:n_select]
    selected_psebs2 = torch.randperm(n_pseudo_bags)[:(n_pseudo_bags - n_select)]
    
    # Combine selected pseudo-bags
    mask1 = torch.zeros(features1.size(0), dtype=torch.bool, device=features1.device)
    for pseb_id in selected_psebs1:
        mask1 |= (pseb_ind1 == pseb_id)
    
    mask2 = torch.zeros(features2.size(0), dtype=torch.bool, device=features2.device)
    for pseb_id in selected_psebs2:
        mask2 |= (pseb_ind2 == pseb_id)
    
    selected1 = features1[mask1] if mask1.sum() > 0 else torch.empty(0, features1.size(1), device=features1.device)
    selected2 = features2[mask2] if mask2.sum() > 0 else torch.empty(0, features2.size(1), device=features2.device)
    
    if selected1.size(0) > 0 and selected2.size(0) > 0:
        mixed_features = torch.cat([selected1, selected2], dim=0)
        content_ratio = n_select / n_pseudo_bags
    elif selected1.size(0) > 0:
        mixed_features = selected1
        content_ratio = 1.0
    else:
        mixed_features = selected2
        content_ratio = 0.0
    
    mixed_label = content_ratio * label1 + (1 - content_ratio) * label2
    
    return mixed_features, mixed_label, content_ratio


def rankmix_data(features1, label1, features2, label2, model, alpha=1.0, strategy='rank'):
    """RankMix: attention-based mixing"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    device = features1.device
    
    # Get attention scores
    with torch.no_grad():
        out1 = model(features1.unsqueeze(0), return_WSI_attn=True)
        out2 = model(features2.unsqueeze(0), return_WSI_attn=True)
        attn1 = out1['WSI_attn'].squeeze() if 'WSI_attn' in out1 else torch.ones(features1.size(0), device=device)
        attn2 = out2['WSI_attn'].squeeze() if 'WSI_attn' in out2 else torch.ones(features2.size(0), device=device)
    
    n1, n2 = features1.size(0), features2.size(0)
    
    if strategy == 'rank':
        # Selection based on attention ranking
        _, sorted_idx1 = torch.sort(attn1, descending=True)
        _, sorted_idx2 = torch.sort(attn2, descending=True)
        
        num_select1 = int(n1 * lam)
        num_select2 = int(n2 * (1 - lam))
        
        selected1 = features1[sorted_idx1[:num_select1]] if num_select1 > 0 else torch.empty(0, features1.size(1), device=device)
        selected2 = features2[sorted_idx2[:num_select2]] if num_select2 > 0 else torch.empty(0, features2.size(1), device=device)
    else:  # 'shrink' strategy
        min_n = min(n1, n2)
        max_n = max(n1, n2)
        
        if n1 > n2:
            _, sorted_idx = torch.sort(attn1, descending=True)
            selected1 = features1[sorted_idx[:min_n]]
            selected2 = features2
        else:
            _, sorted_idx = torch.sort(attn2, descending=True)
            selected1 = features1
            selected2 = features2[sorted_idx[:min_n]]
        
        selected1 = lam * selected1
        selected2 = (1 - lam) * selected2
    
    if selected1.size(0) > 0 and selected2.size(0) > 0:
        mixed_features = torch.cat([selected1, selected2], dim=0)
        mix_ratio = selected1.size(0) / (selected1.size(0) + selected2.size(0))
    elif selected1.size(0) > 0:
        mixed_features = selected1
        mix_ratio = 1.0
    else:
        mixed_features = selected2
        mix_ratio = 0.0
    
    mixed_label = mix_ratio * label1 + (1 - mix_ratio) * label2
    
    return mixed_features, mixed_label, mix_ratio


def remix_data(features1, label1, features2, label2, mode='replace', rate=0.3, strength=0.5):
    """ReMix: feature-level mixing"""
    features1_np = features1.cpu().numpy()
    features2_np = features2.cpu().numpy()
    
    mixed_features = [f.copy() for f in features1_np]
    
    # Find nearest neighbors
    if features1_np.shape[0] > 0 and features2_np.shape[0] > 0:
        closest_idxs = np.argmin(cdist(features1_np, features2_np), axis=1)
        
        for i in range(len(features1_np)):
            if np.random.rand() <= rate:
                if mode == 'replace':
                    mixed_features[i] = features2_np[closest_idxs[i]]
                elif mode == 'append':
                    mixed_features.append(features2_np[closest_idxs[i]])
                elif mode == 'interpolate':
                    generated = (1 - strength) * mixed_features[i] + strength * features2_np[closest_idxs[i]]
                    mixed_features.append(generated)
    
    mixed_features = torch.from_numpy(np.array(mixed_features)).to(features1.device).float()
    
    # Label mixing (simplified to uniform mixing)
    mixed_label = 0.5 * label1 + 0.5 * label2
    
    return mixed_features, mixed_label, 0.5


def apply_mixup(features, labels, batch_idx, method='none', model=None, **kwargs):
    """
    Unified mixup interface
    Args:
        features: list of feature tensors
        labels: list of label tensors
        batch_idx: current batch indices
        method: 'none', 'mixup', 'insmix', 'psebmix', 'rankmix', 'remix'
        model: model (required for RankMix)
        **kwargs: additional parameters for each method
    Returns:
        mixed_features, mixed_labels, mix_ratios
    """
    if method == 'none' or len(features) < 2:
        return features, labels, [1.0] * len(features)
    
    # Random pairing for mixing
    n_samples = len(features)
    perm = torch.randperm(n_samples)
    
    mixed_features = []
    mixed_labels = []
    mix_ratios = []
    
    for i in range(n_samples):
        j = perm[i]
        if i == j:  # If paired with itself, skip mixup
            mixed_features.append(features[i])
            mixed_labels.append(labels[i])
            mix_ratios.append(1.0)
            continue
        
        feat1, label1 = features[i], labels[i]
        feat2, label2 = features[j], labels[j]
        
        alpha = kwargs.get('alpha', 1.0)
        
        if method == 'mixup':
            mixed_feat, mixed_label, ratio = mixup_data(feat1, label1, feat2, label2, alpha)
        elif method == 'insmix':
            mixed_feat, mixed_label, ratio = insmix_data(feat1, label1, feat2, label2, alpha)
        elif method == 'psebmix':
            n_pseudo_bags = kwargs.get('n_pseudo_bags', 30)
            mixed_feat, mixed_label, ratio = psebmix_data(feat1, label1, feat2, label2, n_pseudo_bags, alpha)
        elif method == 'rankmix':
            strategy = kwargs.get('strategy', 'rank')
            mixed_feat, mixed_label, ratio = rankmix_data(feat1, label1, feat2, label2, model, alpha, strategy)
        elif method == 'remix':
            mode = kwargs.get('mode', 'replace')
            rate = kwargs.get('rate', 0.3)
            strength = kwargs.get('strength', 0.5)
            mixed_feat, mixed_label, ratio = remix_data(feat1, label1, feat2, label2, mode, rate, strength)
        else:
            mixed_feat, mixed_label, ratio = feat1, label1, 1.0
        
        mixed_features.append(mixed_feat)
        mixed_labels.append(mixed_label)
        mix_ratios.append(ratio)
    
    return mixed_features, mixed_labels, mix_ratios

