import torch
import time
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,roc_curve,precision_recall_fscore_support,balanced_accuracy_score
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

def _cal_six_scores_optimal_thresh_2class(args,bag_labels,bag_predictions):

    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = _optimal_thresh(fpr, tpr, threshold)
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    this_class_label = np.array(bag_predictions)
    this_class_label[this_class_label>=threshold_optimal] = 1
    this_class_label[this_class_label<threshold_optimal] = 0
    bag_predictions = this_class_label
    precision, recall, f1score, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average='macro',zero_division=0)
    accuracy = accuracy_score(bag_labels, bag_predictions)
    baccuracy = balanced_accuracy_score(bag_labels, bag_predictions)
    return baccuracy,accuracy, auc_value, precision, recall, f1score


def cal_six_scores_optimal_thresh(args,bag_labels,model_logist_after_normal):
    if args.General.num_classes == 2:
        bag_predictions = [model_logist_after_normal_idx[1] for model_logist_after_normal_idx in model_logist_after_normal]
        bag_predictions = np.array(bag_predictions)
        baccuracy,accuracy, auc_value, precision, recall, f1score = _cal_six_scores_optimal_thresh_2class(args,bag_labels,bag_predictions)
    else:
        baccuracies = []
        accuracies = []
        auc_values = []
        precisions = []
        recalls = []
        f1scores = []
        for i in range(args.General.num_classes):
            label_i = _map_list(bag_labels, i) 
            bag_predictions_i = [model_logist_after_normal_idx[i] for model_logist_after_normal_idx in model_logist_after_normal]
            bag_predictions_i = np.array(bag_predictions_i)
            baccuracy_i,accuracy_i, auc_value_i, precision_i, recall_i, f1score_i = _cal_six_scores_optimal_thresh_2class(args,label_i,bag_predictions_i)
            baccuracies.append(baccuracy_i)
            accuracies.append(accuracy_i)
            auc_values.append(auc_value_i)
            precisions.append(precision_i)
            recalls.append(recall_i)
            f1scores.append(f1score_i)
        baccuracy = np.mean(baccuracies)
        accuracy = np.mean(accuracies)
        auc_value = np.mean(auc_values)
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1score = np.mean(f1scores)
    return baccuracy,accuracy, auc_value, precision, recall, f1score
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

    baccuracy,accuracy, auc_value, precision, recall, f1score = cal_six_scores_optimal_thresh(args,labels,bag_predictions_after_normal)
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

    baccuracy,accuracy, auc_testue, precision, recall, f1score = cal_six_scores_optimal_thresh(args,labels,bag_predictions_after_normal)
    test_metrics = {'bacc':baccuracy,'acc':accuracy,'auc':auc_testue,'pre':precision,'recall':recall,'f1':f1score}
    test_loss_log /= len(loader)
    return test_loss_log,test_metrics
