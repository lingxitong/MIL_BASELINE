"""
Training loop utilities with Mixup support
"""
import torch
import torch.nn.functional as F
import time
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from utils.mix_utils import apply_mixup


def train_loop_with_mixup(device, model, dataloader, criterion, optimizer, scheduler, mixup_config=None):
    """Training loop with mixup augmentation"""
    model.train()
    train_loss = 0.
    start_time = time.time()
    
    # Collect batch data for mixup
    all_features = []
    all_labels = []
    all_indices = []
    
    # First collect all data
    for idx, batch in enumerate(dataloader):
        if len(batch) == 3:
            data_idx, data, label = batch
        else:  # len(batch) == 2
            data, label = batch
            data_idx = idx
        all_features.append(data.squeeze(0).to(device))
        all_labels.append(label.to(device))
        all_indices.append(data_idx)
    
    # Apply mixup augmentation
    if mixup_config is not None and mixup_config.get('method', 'none') != 'none':
        method = mixup_config['method']
        prob = mixup_config.get('prob', 0.5)
        
        # Randomly decide whether to apply mixup
        if np.random.rand() < prob:
            kwargs = {
                'alpha': mixup_config.get('alpha', 1.0),
                'n_pseudo_bags': mixup_config.get('n_pseudo_bags', 30),
                'strategy': mixup_config.get('strategy', 'rank'),
                'mode': mixup_config.get('mode', 'replace'),
                'rate': mixup_config.get('rate', 0.3),
                'strength': mixup_config.get('strength', 0.5),
            }
            all_features, all_labels, mix_ratios = apply_mixup(
                all_features, all_labels, all_indices,
                method=method, model=model, **kwargs
            )
    
    # Training loop
    for features, label in zip(all_features, all_labels):
        optimizer.zero_grad()
        
        forward_return = model(features.unsqueeze(0))
        logits = forward_return['logits']
        
        # Handle both hard and soft labels
        if label.dim() == 0 or label.size(0) == 1:
            # Hard label
            if criterion.__class__.__name__ == 'BCEWithLogitsLoss':
                label_for_loss = F.one_hot(label.long(), num_classes=logits.size(1)).float()
            else:
                label_for_loss = label.long()
        else:
            # Soft label (from mixup)
            label_for_loss = label.float()
        
        if criterion.__class__.__name__ == 'BCEWithLogitsLoss':
            loss = criterion(logits, label_for_loss)
        else:
            loss = criterion(logits, label_for_loss.argmax(dim=-1) if label_for_loss.dim() > 1 else label_for_loss)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    if scheduler is not None:
        scheduler.step()
    
    train_loss /= len(all_features)
    cost_time = time.time() - start_time
    
    return train_loss, cost_time

