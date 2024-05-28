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

def cal_six_scores(logits, labels, num_classes):       # logits:[batch_size, num_classes]   labels:[batch_size, ]
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    predicted_classes = torch.argmax(logits, dim=1)
    accuracy = accuracy_score(labels.numpy(), predicted_classes.numpy())
    probs = F.softmax(logits, dim=1)
    if num_classes > 2:
        auc = roc_auc_score(y_true=labels.numpy(), y_score=probs.numpy(), average='macro', multi_class='ovr')
    else:
        auc = roc_auc_score(y_true=labels.numpy(), y_score=probs[:,1].numpy())
    f1 = f1_score(labels.numpy(), predicted_classes.numpy(), average='weighted')
    recall = recall_score(labels.numpy(), predicted_classes.numpy(), average='weighted')
    precision = precision_score(labels.numpy(), predicted_classes.numpy(), average='weighted')
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
    return baccuracy,accuracy, auc, precision, recall, f1

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

    baccuracy,accuracy, auc_value, precision, recall, f1score = cal_six_scores(bag_predictions_after_normal,labels,args.General.num_classes)
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

    baccuracy,accuracy, auc_value, precision, recall, f1score = cal_six_scores(bag_predictions_after_normal,labels,args.General.num_classes)
    test_metrics = {'bacc':baccuracy,'acc':accuracy,'auc':auc_testue,'pre':precision,'recall':recall,'f1':f1score}
    test_loss_log /= len(loader)
    return test_loss_log,test_metrics
