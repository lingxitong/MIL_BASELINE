import torch
import time
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,roc_curve,precision_recall_fscore_support,balanced_accuracy_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, cohen_kappa_score, confusion_matrix
import time
import torch.nn as nn

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
        train_logits = model(bag)['logits']
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


def val_loop(device,num_classes,model,loader,criterion,retrun_WSI_feature = False,return_WSI_attn=False):
    model.eval()
    val_loss_log = 0
    labels = []
    bag_predictions_after_normal = []
    model = model.to(device)
    WSI_features = []
    WSI_attns = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            label = data[1].to(device).long()
            labels.append(label.cpu().numpy())
            bag = data[0].to(device).float()
            if retrun_WSI_feature:
                WSI_feature = model(bag,return_WSI_feature=True)['WSI_feature']
                WSI_features.append(WSI_feature)
                continue
            if return_WSI_attn:
                WSI_attn = model(bag,return_WSI_attn=True)['WSI_attn']
                WSI_attns.append(WSI_attn)
                continue
            val_logits = model(bag)['logits']
            val_logits = val_logits.squeeze(0)
            bag_predictions_after_normal.append(torch.softmax(val_logits,0).cpu().numpy())
            val_logits = val_logits.unsqueeze(0)
            val_loss = criterion(val_logits,label)
            val_loss_log += val_loss.item()
    if retrun_WSI_feature:
        WSI_features = torch.cat(WSI_features, dim=0).cpu().numpy()
        return WSI_features
    if return_WSI_attn:
        return WSI_attns
    val_metrics= cal_scores(bag_predictions_after_normal,labels,num_classes)
    val_loss_log /= len(loader)
    return val_loss_log,val_metrics

def ac_train_loop(device,model,loader,criterion,optimizer,scheduler,n_token):
    start = time.time()
    model.train()
    train_loss_log = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        label = data[1].long().to(device)
        bag = data[0].to(device).float()
        forward_return = model(bag)
        train_logits = forward_return['logits']
        sub_preds = forward_return['sub_preds']
        attns = forward_return['attns']
        if n_token > 1:
            loss0 = criterion(sub_preds, label.repeat_interleave(n_token))
        else:
            loss0 = torch.tensor(0.)
        diff_loss = torch.tensor(0).to(device, dtype=torch.float)
        attns = torch.softmax(attns, dim=-1)

        for i in range(n_token):
            for j in range(i + 1, n_token): 
                diff_loss += torch.cosine_similarity(attns[:, i], attns[:, j], dim=-1).mean() / (
                            n_token * (n_token - 1) / 2)
        train_loss = criterion(train_logits, label)
        train_loss = diff_loss + loss0 + train_loss
        train_loss_log += train_loss.item()
        train_loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    train_loss_log /= len(loader)
    end = time.time()
    total_time = end - start
    return train_loss_log,total_time


def ac_val_loop(device,num_classes,model,loader,criterion,n_token,retrun_WSI_feature = False,return_WSI_attn=False):
    model.eval()
    val_loss_log = 0
    labels = []
    bag_predictions_after_normal = []
    model = model.to(device)
    WSI_features = []
    WSI_attns = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            label = data[1].long().to(device)
            labels.append(label.cpu().numpy())
            bag = data[0].to(device).float()
            forward_return = model(bag)
            val_logits = forward_return['logits']
            val_logits = val_logits.squeeze(0)
            bag_predictions_after_normal.append(torch.softmax(val_logits,0).cpu().numpy())
            val_logits = val_logits.unsqueeze(0)
            sub_preds = forward_return['sub_preds']
            attns = forward_return['attns']
            if n_token > 1:
                loss0 = criterion(sub_preds, label.repeat_interleave(n_token))
            else:
                loss0 = torch.tensor(0.)
            diff_loss = torch.tensor(0).to(device, dtype=torch.float)
            attns = torch.softmax(attns, dim=-1)
            for i in range(n_token):
                for j in range(i + 1, n_token): 
                    diff_loss += torch.cosine_similarity(attns[:, i], attns[:, j], dim=-1).mean() / (
                                n_token * (n_token - 1) / 2)
            val_loss = criterion(val_logits, label)
            val_loss = diff_loss + loss0 + val_loss
            val_loss_log += val_loss.item()
    if retrun_WSI_feature:
        WSI_features = torch.cat(WSI_features, dim=0).cpu().numpy()
        return WSI_features
    if return_WSI_attn:
        return WSI_attns
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
        forward_return = model(bag,label=label)
        instance_loss = forward_return['instance_loss']
        train_logits = forward_return['logits']
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


def tripleloss(golabal,p_center,nc_center):
    golabal = golabal.squeeze(0)
    n_globallesionrepresente, _ = golabal.shape
    p_center = p_center.repeat(n_globallesionrepresente, 1)
    nc_center = nc_center.repeat(n_globallesionrepresente, 1)
    triple_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y) ,margin=1)
    loss = triple_loss(golabal,p_center,nc_center)
    return loss

def dgr_train_loop(device,model,loader,criterion,optimizer,scheduler,now_epoch,epoch_des,n_lesion):
    
    start = time.time()
    model.train()
    train_loss_log = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        label = data[1].long().to(device)
        bag = data[0].to(device).float()
        if torch.argmax(label)==0:
            forward_return = model(bag,bag_mode='normal')
            train_logits, A,H,p_center,nc_center,lesion = forward_return['logits'],forward_return['A'],forward_return['H'],forward_return['postivecenter'],forward_return['normalcenter'],forward_return['lesion_enhacing']
        else:
            train_logits, A,H,p_center,nc_center,lesion= model(bag,bag_mode='abnormal')
        if now_epoch < epoch_des:
            train_loss = criterion(train_logits, label)
        else:
            train_loss = criterion(train_logits, label)
            lesion_norm = lesion.squeeze(0)
            lesion_norm = torch.nn.functional.normalize(lesion_norm)
            div_loss = -torch.logdet(lesion_norm@lesion_norm.T+1e-10*torch.eye(n_lesion).to(device))
            sim_loss = tripleloss(lesion,p_center,nc_center)
            train_loss = train_loss + 0.1*div_loss + 0.1*sim_loss 

        train_loss_log += train_loss.item()
        train_loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    train_loss_log /= len(loader)
    end = time.time()
    total_time = end - start
    return train_loss_log,total_time

def clam_val_loop(device,num_classes,model,loader,criterion,bag_weight,retrun_WSI_feature = False,return_WSI_attn=False):
    model.eval()
    val_loss_log = 0
    labels = []
    bag_predictions_after_normal = []
    model = model.to(device)
    WSI_features = []
    WSI_attns = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            label = data[1].to(device).long()
            labels.append(label.cpu().numpy())
            bag = data[0].to(device).float()
            if retrun_WSI_feature:
                WSI_feature = model(bag,label = label, return_WSI_feature=True)['WSI_feature']
                WSI_features.append(WSI_feature)
                continue
            if return_WSI_attn:
                WSI_attn = model(bag,label = label, return_WSI_attn=True)['WSI_attn']
                WSI_attns.append(WSI_attn)
                continue
            forward_return = model(bag,label=label)
            instance_loss = forward_return['instance_loss']
            val_logits = forward_return['logits']
            val_logits = val_logits.squeeze(0)
            bag_predictions_after_normal.append(torch.softmax(val_logits,0).cpu().numpy())
            val_logits = val_logits.unsqueeze(0)
            val_loss = criterion(val_logits,label)
            total_loss = val_loss * bag_weight + instance_loss * (1-bag_weight)
            val_loss_log += total_loss.item()
    if retrun_WSI_feature:
        WSI_features = torch.cat(WSI_features, dim=0).cpu().numpy()
        return WSI_features
    if return_WSI_attn:
        return WSI_attns
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
        forward_return = model(bag)
        max_prediction = forward_return['max_prediction']
        train_logits = forward_return['logits']
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

def ds_val_loop(device,num_classes,model,loader,criterion,retrun_WSI_feature = False,return_WSI_attn=False):
    WSI_features = []
    WSI_attns = []
    labels = []
    bag_predictions_after_normal = []
    val_loss_log = 0
    model.eval()
    model = model.to(device)
    with torch.autograd.set_detect_anomaly(True):
        for i, data in enumerate(loader):
            label = data[1].long().to(device)
            labels.append(label.cpu().numpy())
            bag = data[0].to(device).float()
            forward_return = model(bag)
            if retrun_WSI_feature:
                WSI_feature = model(bag,return_WSI_feature=True)['WSI_feature']
                WSI_features.append(WSI_feature)
                continue
            if return_WSI_attn:
                WSI_attn = model(bag,return_WSI_attn=True)['WSI_attn']
                WSI_attns.append(WSI_attn)
                continue
            max_prediction = forward_return['max_prediction']
            val_logits = forward_return['logits']
            bag_predictions_after_normal.append(torch.softmax(val_logits[0],0).cpu().detach().numpy())
            loss_bag = criterion(val_logits, label)
            loss_max = criterion(max_prediction, label)
            val_loss = 0.5*loss_bag + 0.5*loss_max
            val_loss_log += val_loss.item()
    if retrun_WSI_feature:
        WSI_features = torch.cat(WSI_features, dim=0).cpu().detach().numpy()
        return WSI_features
    if return_WSI_attn:
        return WSI_attns
    val_loss_log /= len(loader)
    val_metrics= cal_scores(bag_predictions_after_normal,labels,num_classes)
    return val_loss_log,val_metrics

def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps

def dtfd_train_loop(device, model_list, loader, criterion, optimizer_list, scheduler_list, num_Group, grad_clipping,distill,total_instance):
    train_loss_log = 0
    start = time.time()
    instance_per_group = total_instance // num_Group
    # Unpack model list
    classifier, attention, dimReduction, attCls = model_list
    classifier.train()
    attention.train()
    dimReduction.train()
    attCls.train()

    # Unpack optimizer and scheduler lists
    optimizer_A, optimizer_B = optimizer_list
    scheduler_A, scheduler_B = scheduler_list

    total_loss = 0
    for i, data in enumerate(loader):
        label = data[1].long().to(device)
        bag = data[0].to(device).float()

        slide_sub_preds = []
        slide_sub_labels = []
        slide_pseudo_feat = []

        # Split bag into chunks
        inputs_pseudo_bags = torch.chunk(bag.squeeze(0), num_Group, dim=0)

        for subFeat_tensor in inputs_pseudo_bags:
            slide_sub_labels.append(label)
            subFeat_tensor = subFeat_tensor.to(device)

            # Forward pass through models
            tmidFeat = dimReduction(subFeat_tensor)
            tAA = attention(tmidFeat).squeeze(0)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  # n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0, keepdim=True)  # 1 x fs
            tPredict = classifier(tattFeat_tensor)  # 1 x 2
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


        # Concatenate tensors
        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0)  # numGroup x fs
        slide_sub_labels = torch.cat(slide_sub_labels, dim=0)  # numGroup

        # Calculate and backpropagate loss for the first tier
        loss_A = criterion(slide_sub_preds, slide_sub_labels)
        optimizer_A.zero_grad()
        loss_A.backward(retain_graph=True)
        total_loss += loss_A.item()

        # Clip gradients and update weights
        torch.nn.utils.clip_grad_norm_(dimReduction.parameters(), grad_clipping)
        torch.nn.utils.clip_grad_norm_(attention.parameters(), grad_clipping)
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), grad_clipping)


        # Second tier optimization
        gSlidePred = attCls(slide_pseudo_feat)['logits']
        loss_B = criterion(gSlidePred, label).mean()
        optimizer_B.zero_grad()
        loss_B.backward()
        total_loss += loss_B.item()

        # Clip gradients and update weights
        torch.nn.utils.clip_grad_norm_(attCls.parameters(), grad_clipping)
        optimizer_A.step()
        optimizer_B.step()

    # Step schedulers
    scheduler_A.step()
    scheduler_B.step()

    end = time.time()
    total_loss /= len(loader)
    total_time = end - start

    return total_loss, total_time


def dtfd_val_loop(device,num_classes,model_list,loader,criterion,num_Group,grad_clipping,distill,total_instance,retrun_WSI_feature = False,return_WSI_attn=False):
    WSI_features = []
    WSI_attns = []
    instance_per_group = total_instance // num_Group
    classifier,attention,dimReduction,attCls = model_list
    classifier.eval()
    attention.eval()
    dimReduction.eval()
    attCls.eval()
    total_loss = 0
    y_score=[]
    y_true=[]
    for i, data in enumerate(loader):
        label = data[1].long().to(device)
        bag = data[0].to(device).float()

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
        gSlidePred = torch.softmax(attCls(slide_pseudo_feat)['logits'], dim=1)
        forward_return = attCls(slide_pseudo_feat, return_WSI_attn = return_WSI_attn, return_WSI_feature = retrun_WSI_feature)
        if retrun_WSI_feature:
            WSI_feature = forward_return['WSI_feature']
            WSI_features.append(WSI_feature)
            continue
        if return_WSI_attn:
            WSI_attn = forward_return['WSI_attn']
            WSI_attns.append(WSI_attn)
            continue
        loss = criterion(forward_return['logits'], label)
        total_loss += loss.item()
        pred=(gSlidePred.cpu().data.numpy()).tolist()
        y_score.extend(pred)
        y_true.extend(label)
    if retrun_WSI_feature:
        WSI_features = torch.cat(WSI_features, dim=0).cpu().detach().numpy()
        return WSI_features
    if return_WSI_attn:
        return WSI_attns
    
    total_loss /= len(loader)
    val_metrics= cal_scores(y_score,y_true,num_classes)
    return total_loss,val_metrics
