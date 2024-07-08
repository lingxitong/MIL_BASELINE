import torch
import time
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,roc_curve,precision_recall_fscore_support,balanced_accuracy_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, cohen_kappa_score, confusion_matrix
import time
import torch

def cal_scores(logits, labels, num_classes):       # logits:[batch_size, num_classes]   labels:[batch_size, ]
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    predicted_classes = torch.argmax(logits, dim=1)
    accuracy = accuracy_score(labels.numpy(), predicted_classes.numpy())
    probs = F.softmax(logits, dim=1)
    if num_classes > 2:
        macro_auc = roc_auc_score(y_true=labels.numpy(), y_score=probs.numpy(), average='macro', multi_class='ovr')
        micro_auc = roc_auc_score(y_true=labels.numpy(), y_score=probs.numpy(), average='micro', multi_class='ovr')
        weighted_auc = roc_auc_score(y_true=labels.numpy(), y_score=probs.numpy(), average='weighted', multi_class='ovr')
    else:
        macro_auc = roc_auc_score(y_true=labels.numpy(), y_score=probs[:,1].numpy())
        weighted_auc = micro_auc = macro_auc
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
    quadratic_kappa = cohen_kappa_score(labels.numpy(), predicted_classes.numpy(), weights='quadratic')
    linear_kappa = cohen_kappa_score(labels.numpy(), predicted_classes.numpy(), weights='linear')
    confusion_mat = confusion_matrix(labels.numpy(), predicted_classes.numpy())
    metrics = {'acc': accuracy,  'bacc': baccuracy, 
               'macro_auc': macro_auc, 'micro_auc': micro_auc, 'weighted_auc':weighted_auc,
                'macro_f1': macro_f1, 'micro_f1': micro_f1, 'weighted_f1': weighted_f1, 
                 'macro_recall': macro_recall, 'micro_recall': micro_recall,'weighted_recall': weighted_recall, 
                 'macro_pre': macro_precision, 'micro_pre': micro_precision,'weighted_pre': weighted_precision,
                 'quadratic_kappa': quadratic_kappa,'linear_kappa':linear_kappa,  
                 'confusion_mat': confusion_mat}
    return metrics

def train_loop(device,model,loader,criterion,optimizer,scheduler):
    
    start = time.time()
    model.train()
    train_loss_log = 0
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


def val_loop(device,num_classes,model,loader,criterion):
    model.eval()
    val_loss_log = 0
    labels = []
    bag_predictions_after_normal = []
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

    val_metrics= cal_scores(bag_predictions_after_normal,labels,num_classes)
    val_loss_log /= len(loader)
    return val_loss_log,val_metrics


# clam has instance-loss defferent from other mil models
def clam_train_loop(device,model,loader,criterion,optimizer,scheduler,bag_weight):
    
    start = time.time()
    model.train()
    train_loss_log = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        label = data[1].long().to(device)
        bag = data[0].to(device).float()
        train_logits, Y_prob,Y_hat, A_raw, results_dict = model(bag,label=label)
        instance_loss = results_dict['instance_loss']
        train_loss = criterion(train_logits, label)
        total_loss = train_loss * bag_weight + instance_loss * (1-bag_weight)
        train_loss_log += total_loss.item()
        total_loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    train_loss_log /= len(loader)
    end = time.time()
    total_time = end - start
    return train_loss_log,total_time


def clam_val_loop(device,num_classes,model,loader,criterion,bag_weight):
    model.eval()
    val_loss_log = 0
    labels = []
    bag_predictions_after_normal = []
    model = model.to(device)
    with torch.no_grad():
        for i, data in enumerate(loader):
            label = data[1].to(device).long()
            labels.append(label.cpu().numpy())
            bag = data[0].to(device).float()
            val_logits, Y_paro,Y_hat, A_raw, results_dict = model(bag,label=label)
            instance_loss = results_dict['instance_loss']
            val_logits = val_logits.squeeze(0)
            bag_predictions_after_normal.append(torch.softmax(val_logits,0).cpu().numpy())
            val_logits = val_logits.unsqueeze(0)
            val_loss = criterion(val_logits,label)
            total_loss = val_loss * bag_weight + instance_loss * (1-bag_weight)
            val_loss_log += total_loss.item()

    val_metrics= cal_scores(bag_predictions_after_normal,labels,num_classes)
    val_loss_log /= len(loader)
    return val_loss_log,val_metrics

def ds_train_loop(device,model,loader,criterion,optimizer,scheduler):
    
    start = time.time()
    model.train()
    train_loss_log = 0
    model = model.to(device)
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        label = data[1].long().to(device)
        bag = data[0].to(device).float()

        max_prediction, train_logits = model(bag)
        max_prediction, _ = torch.max(max_prediction, 0)
        max_prediction = max_prediction.unsqueeze(0)
        loss_bag = criterion(train_logits, label)
        loss_max = criterion(max_prediction, label)
        train_loss = 0.5*loss_bag + 0.5*loss_max
        train_loss_log += train_loss.item()

        train_loss.backward()

        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    train_loss_log /= len(loader)
    end = time.time()
    total_time = end - start
    return train_loss_log,total_time

def ds_val_loop(device,model,loader,criterion):
    
    start = time.time()
    model.train()
    train_loss_log = 0
    model = model.to(device)
    with torch.autograd.set_detect_anomaly(True):
        for i, data in enumerate(loader):
            label = data[1].long().to(device)
            bag = data[0].to(device).float()

            max_prediction, train_logits = model(bag)
            max_prediction, _ = torch.max(max_prediction, 0)
            max_prediction = max_prediction.unsqueeze(0)
            loss_bag = criterion(train_logits, label)
            loss_max = criterion(max_prediction, label)
            val_loss = 0.5*loss_bag + 0.5*loss_max
            val_loss_log += val_loss.item()

    train_loss_log /= len(loader)
    end = time.time()
    total_time = end - start
    return val_loss_log,total_time