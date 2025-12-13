from torchvision import transforms
import os
from ..models.resnet_custom import resnet50_baseline
from transformers import CLIPModel, CLIPProcessor
import timm
from ..conch.open_clip_custom import *
from typing import List, Union, Tuple
import torch.nn as nn
import torch
def get_transforms(backbone_name:str):
    assert backbone_name in ['vit_s_imagenet','plip','uni','resnet50_imagenet','conch','ctranspath','gigapath','virchow','virchow_v2','conch_v1_5']
    if backbone_name == 'vit_s_imagenet':
        mean = (0.5,0.5,0.5)
        std = (0.5,0.5,0.5)
        vit_s_transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = mean, std = std)])
        return vit_s_transforms
    elif backbone_name == 'resnet50_imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        r50_transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = mean, std = std)])
        return r50_transforms
    elif backbone_name == 'plip':
        mean = (0.48145466,0.4578275,0.40821073)
        std = (0.26862954,0.26130258,0.27577711)
        plip_transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = mean, std = std)])
        return plip_transforms
    elif backbone_name == 'uni':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        uni_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = mean, std = std)])
        return uni_transform
    elif backbone_name == 'gigapath':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        gigapath_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = mean, std = std)])
        return gigapath_transform
    elif backbone_name == 'conch':
        mean=(0.48145466, 0.4578275, 0.40821073)
        std=(0.26862954, 0.26130258, 0.27577711)
        conch_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = mean, std = std)])
        return conch_transform
    elif backbone_name == 'ctranspath':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        ctranspath_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = mean, std = std)])
        return ctranspath_transform
    elif backbone_name == 'virchow' or backbone_name == 'virchow_v2':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        virchow_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = mean, std = std)])
        return virchow_transform
    elif backbone_name == 'conch_v1_5':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        virchow_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = mean, std = std)])
        return virchow_transform
    elif backbone_name == 'uni_v2':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        uni_v2_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = mean, std = std)])
        return uni_v2_transform
    elif backbone_name == 'hoptimus_v1':
        mean=(0.707223, 0.578729, 0.703617)
        std=(0.211883, 0.230117, 0.177517)
        hoptimus_v1_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = mean, std = std)])
        return hoptimus_v1_transform
    elif backbone_name == 'midnight':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        midnight_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = mean, std = std)])
        return midnight_transform
    else:
        return None
    

        
        
def load_plip_model( name: str,
                device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
                auth_token=None):

    model = CLIPModel.from_pretrained(name, use_auth_token=auth_token)
    preprocessing = CLIPProcessor.from_pretrained(name, use_auth_token=auth_token)

    return model, preprocessing, hash
    
    
def get_backbone(backbone_name:str,device,pretrained_weights_dir):
    model_dir = pretrained_weights_dir
    if backbone_name == 'resnet50_imagenet':
        model = resnet50_baseline(pretrained=False)
        checkpoint_path = os.path.join(model_dir, "resnet50-19c8e357.pth")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device,weights_only=True), strict=True)
        model = model.to(device)
    elif backbone_name == 'vit_s_imagenet':
        model = timm.create_model('vit_small_patch16_224.augreg_in21k_ft_in1k')
        checkpoint_path = os.path.join(model_dir, "pytorch_model.bin")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device,weights_only=True), strict=True)
        model.head = nn.Identity()
        model = model.to(device)
    elif backbone_name == 'plip':
        model,_,_ = load_plip_model(name=model_dir, auth_token=None)
        model = model.to(device)
    elif backbone_name == 'uni':
        model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
        model.load_state_dict(torch.load(os.path.join(model_dir, "pytorch_model.bin"), map_location=device,weights_only=True), strict=True)
        model = model.to(device)
    elif backbone_name == 'conch':
        checkpoint_path = os.path.join(model_dir, "pytorch_model.bin")
        model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=checkpoint_path)
        model = model.to(device)
    elif backbone_name == 'gigapath':
        gig_config = {
        "architecture": "vit_giant_patch14_dinov2",
        "num_classes": 0,
        "num_features": 1536,
        "global_pool": "token",
        "model_args": {
        "img_size": 224,
        "in_chans": 3,
        "patch_size": 16,
        "embed_dim": 1536,
        "depth": 40,
        "num_heads": 24,
        "init_values": 1e-05,
        "mlp_ratio": 5.33334,
        "num_classes": 0}} 
        checkpoint_path = os.path.join(model_dir, "pytorch_model.bin")
        model = timm.create_model("vit_giant_patch14_dinov2", pretrained=False, **gig_config['model_args'])
        state_dict = torch.load(checkpoint_path , map_location="cpu",weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
    elif backbone_name == 'ctranspath':
        from clam_base_utils.Ctranspath.ctrans import ctranspath
        checkpoint_path = os.path.join(model_dir, "ctranspath.pth")
        model = ctranspath()
        model.head = nn.Identity()
        state_dict = torch.load(checkpoint_path,weights_only=True)
        model.load_state_dict(state_dict['model'], strict=True)
        model = model.to(device)
    elif backbone_name == 'virchow':
       from timm.layers import SwiGLUPacked
       virchow_config = {
        "img_size": 224,
        "init_values": 1e-5,
        "num_classes": 0,
        "mlp_ratio": 5.3375,
        "global_pool": "",
        "dynamic_img_size": True}
       checkpoint_path = os.path.join(model_dir, "pytorch_model.bin")
       model = timm.create_model("vit_huge_patch14_224", pretrained=False,mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU,**virchow_config)
       state_dict = torch.load(checkpoint_path, map_location="cpu",weights_only=True)
       model.load_state_dict(state_dict, strict=True)
       model = model.to(device)
    elif backbone_name == 'virchow_v2':
        virchow_config = {
        "img_size": 224,
        "init_values": 1e-5,
        "num_classes": 0,
        "mlp_ratio": 5.3375,
        "reg_tokens": 4,
        "global_pool": "",
        "dynamic_img_size": True}
        checkpoint_path = os.path.join(model_dir, "pytorch_model.bin")
        model = timm.create_model("vit_huge_patch14_224", pretrained=False,mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU,**virchow_config)
        state_dict = torch.load(checkpoint_path, map_location="cpu",weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
    elif backbone_name == 'conch_v1_5':
      from .conch_v1_5_config import ConchConfig
      from .build_conch_v1_5 import build_conch_v1_5
      checkpoint_path = os.path.join(model_dir, "conch_v1_5_pytorch_model.bin")
      conch_v1_5_config = ConchConfig()
      model = build_conch_v1_5(conch_v1_5_config, checkpoint_path)
      model = model.to(device)
    elif backbone_name == 'uni_v2':
        uni_v2_config = {
            'img_size': 224,
            'patch_size': 14,
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5,
            'embed_dim': 1536,
            'mlp_ratio': 2.66667 * 2,
            'num_classes': 0,
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked,
            'act_layer': torch.nn.SiLU,
            'reg_tokens': 8,
            'dynamic_img_size': True
        }
        model = timm.create_model(model_name='vit_giant_patch14_224', pretrained=False, **uni_v2_config)
        checkpoint_path = os.path.join(model_dir, "pytorch_model.bin")
        state_dict = torch.load(checkpoint_path, map_location="cpu",weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
    elif backbone_name == 'hoptimus_v1':
        assert timm.__version__ == '0.9.16', f"H-Optimus requires timm version 0.9.16, but found {timm.__version__}. Please install the correct version using `pip install timm==0.9.16`"
        hoptimus_v1_config = {
            "num_classes": 0,
            "img_size": 224,
            "global_pool": "token",
            'init_values': 1e-5,
            'dynamic_img_size': False
        }
        model = timm.create_model("vit_giant_patch14_reg4_dinov2", pretrained=False, **hoptimus_v1_config)
        checkpoint_path = os.path.join(model_dir, "pytorch_model.bin")
        state_dict = torch.load(checkpoint_path, map_location="cpu",weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
    elif backbone_name == 'midnight':
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_dir)
        model = model.to(device)
    model.eval()
    return model
        

def feature_extractor_adapter(model, batch,model_name):
    if model_name == 'plip':
        features = model.get_image_features(batch)
    elif model_name == 'conch':
        features = model.encode_image(batch)
    elif model_name == 'virchow':
        features = model(batch)
        class_token = features[:, 0]    
        patch_tokens = features[:, 1:]  
        features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1) 
    elif model_name == 'virchow_v2':
        features = model(batch)
        class_token = features[:, 0]    
        patch_tokens = features[:, 5:] 
        features = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
    elif model_name == 'midnight':
        out = model(batch).last_hidden_state
        features = out[:, 0, :]
    else:
        features = model(batch)
    if len(features.shape) == 3:
        features = features.squeeze(0)
    return features
