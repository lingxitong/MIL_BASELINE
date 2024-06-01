import torch
import time
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,roc_curve,precision_recall_fscore_support,balanced_accuracy_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, cohen_kappa_score, confusion_matrix
import random
import time
import os
import random
import glob
import pickle
import torch
def _map_list(lst, i):
    return [1 if x == i else 0 for x in lst]
def _optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def cal_scores(logits, labels, num_classes):       # logits:[batch_size, num_classes]   labels:[batch_size, ]
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    predicted_classes = torch.argmax(logits, dim=1)
    accuracy = accuracy_score(labels.numpy(), predicted_classes.numpy())
    probs = F.softmax(logits, dim=1)
    if num_classes > 2:
        macro_auc = roc_auc_score(y_true=labels.numpy(), y_score=probs.numpy(), average='macro', multi_class='ovr')
        micro_auc = roc_auc_score(y_true=labels.numpy(), y_score=probs.numpy(), average='micro', multi_class='ovr')
    else:
        macro_auc = roc_auc_score(y_true=labels.numpy(), y_score=probs[:,1].numpy())
    weighted_f1 = f1_score(labels.numpy(), predicted_classes.numpy(), average='weighted')
    weighted_recall = recall_score(labels.numpy(), predicted_classes.numpy(), average='weighted')
    weighted_precision = precision_score(labels.numpy(), predicted_classes.numpy(), average='weighted')
    macro_f1 = f1_score(labels.numpy(), predicted_classes.numpy(), average='macro')
    macro_recall = recall_score(labels.numpy(), predicted_classes.numpy(), average='macro')
    macro_precision = precision_score(labels.numpy(), predicted_classes.numpy(), average='macro')
    micro_f1 = f1_score(labels.numpy(), predicted_classes.numpy(), average='micro')
    micro_recall = recall_score(labels.numpy(), predicted_classes.numpy(), average='micro')
    micro_precision = precision_score(labels.numpy(), predicted_classes.numpy(), average='micro') 
    baccuracy = balanced_accuracy_score(labels.numpy(), predicted_classes.numpy())
    kappa = cohen_kappa_score(labels.numpy(), predicted_classes.numpy(), weights='quadratic')
    specificity_list = []
    for class_idx in range(num_classes):
        true_positive = np.sum((labels.numpy() == class_idx) & (predicted_classes.numpy() == class_idx))
        true_negative = np.sum((labels.numpy() != class_idx) & (predicted_classes.numpy() != class_idx))
        false_positive = np.sum((labels.numpy() != class_idx) & (predicted_classes.numpy() == class_idx))
        false_negative = np.sum((labels.numpy() == class_idx) & (predicted_classes.numpy() != class_idx))
        specificity = true_negative / (true_negative + false_positive)
        specificity_list.append(specificity)
    macro_specificity = np.mean(specificity_list)
    confusion_mat = confusion_matrix(labels.numpy(), predicted_classes.numpy())
    metrics = {'accuracy': accuracy, 'macro_auc': macro_auc, 'micro_auc': micro_auc, 'weighted_f1': weighted_f1, 'weighted_recall': weighted_recall, 'weighted_precision': weighted_precision, 'macro_f1': macro_f1, 'macro_recall': macro_recall, 'macro_precision': macro_precision, 'micro_f1': micro_f1, 'micro_recall': micro_recall, 'micro_precision': micro_precision, 'baccuracy': baccuracy, 'kappa': kappa, 'macro_specificity': macro_specificity, 'confusion_mat': confusion_mat}
    return baccuracy, accuracy, macro_auc, macro_precision, macro_recall, macro_f1

def train_loop(args,model,loader,criterion,optimizer,scheduler):
    
    start = time.time()
    model.train()
    train_loss_log = 0
    device = torch.device(f'cuda:{args.General.device}')
    print(f'device:{device}')
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        label = data[1].long().to(device)
        bag = data[0].to(device).float()
        train_logits = model(bag)
        train_loss = criterion(train_logits, label)
        train_loss_log += train_loss.item()
        train_loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    train_loss_log /= len(loader)
    end = time.time()
    total_time = end - start
    return train_loss_log,total_time


def val_loop(args,model,loader,criterion):
    model.eval()
    val_loss_log = 0
    labels = []
    bag_predictions_after_normal = []
    device = torch.device(f'cuda:{args.General.device}')
    model = model.to(device)
    with torch.no_grad():
        for i, data in enumerate(loader):
            label = data[1].to(device).long()
            labels.append(label.cpu().numpy())
            bag = data[0].to(device).float()
            val_logits = model(bag)
            val_logits = val_logits.squeeze(0)
            bag_predictions_after_normal.append(torch.softmax(val_logits,0).cpu().numpy())
            val_logits = val_logits.unsqueeze(0)
            val_loss = criterion(val_logits,label)
            val_loss_log += val_loss.item()

    baccuracy,accuracy, auc_value, precision, recall, f1score = cal_scores(bag_predictions_after_normal,labels,args.General.num_classes)
    val_metrics = {'bacc':baccuracy,'acc':accuracy,'auc':auc_value,'pre':precision,'recall':recall,'f1':f1score}
    val_loss_log /= len(loader)
    return val_loss_log,val_metrics


def test_loop(args,model,loader,criterion):
    model.eval()
    test_loss_log = 0
    labels = []
    bag_predictions_after_normal = []
    device = torch.device(f'cuda:{args.General.device}')
    model = model.to(device)
    with torch.no_grad():
        for i, data in enumerate(loader):
            label = data[1].to(device).long()
            labels.append(label.cpu().numpy())
            bag = data[0].to(device).float()
            test_logits = model(bag)
            test_logits = test_logits.squeeze(0)
            bag_predictions_after_normal.append(torch.softmax(test_logits,0).cpu().numpy())
            test_logits = test_logits.unsqueeze(0)
            test_loss = criterion(test_logits,label)
            test_loss_log += test_loss.item()

    baccuracy,accuracy, auc_value, precision, recall, f1score = cal_scores(bag_predictions_after_normal,labels,args.General.num_classes)
    test_metrics = {'bacc':baccuracy,'acc':accuracy,'auc':auc_value,'pre':precision,'recall':recall,'f1':f1score}
    test_loss_log /= len(loader)
    return test_loss_log,test_metrics
