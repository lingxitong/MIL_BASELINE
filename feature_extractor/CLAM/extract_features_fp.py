import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
from torchvision import transforms
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Union, Tuple
from torch.utils.data import DataLoader
import PIL
from transformers import CLIPModel, CLIPProcessor
import timm
from datasets import Dataset, Image
import time
from clam_datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline
import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import openslide
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def plip_transforms(pretrained=False):
	if pretrained:
		mean = (0.48145466,0.4578275,0.40821073)
		std = (0.26862954,0.26130258,0.27577711)

	else:
		mean = (0.48145466,0.4578275,0.40821073)
		std = (0.26862954,0.26130258,0.27577711)

	trnsfrms_val = transforms.Compose(
					[
					 transforms.ToTensor(),
					 transforms.Normalize(mean = mean, std = std)
					]
				)

	return trnsfrms_val

def uni_transforms():
    uni_transform = transforms.Compose(
    [	transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
    return uni_transform
def load_plip_model( name: str,
                device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
                auth_token=None):

    model = CLIPModel.from_pretrained(name, use_auth_token=auth_token)
    preprocessing = CLIPProcessor.from_pretrained(name, use_auth_token=auth_token)

    return model, preprocessing, hash
def compute_w_loader(args,file_path, output_path, wsi, model,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""
	if args.backbone == 'plip':
		custom_transforms = plip_transforms(pretrained=pretrained)
	elif args.backbone == 'uni':
		custom_transforms = uni_transforms()
	else:
		custom_transforms = None
     
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
		custom_downsample=custom_downsample,custom_transforms=custom_transforms,target_patch_size=target_patch_size)
	x, y = dataset[0]
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	for count, (batch, coords) in enumerate(loader):
		
		with torch.no_grad():	
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)
			# print(batch.shape)
			if args.backbone == 'plip':
				features = model.get_image_features(batch)
			else:
				features = model(batch)
			features = features.cpu().numpy()

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--backbone', type=str,choices=['vit_s_imagenet','plip','uni','resnet50_imagenet'])
args = parser.parse_args()


if __name__ == '__main__':

	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))

	print('loading model checkpoint')
	if args.backbone == 'resnet50_imagenet':

		model = resnet50_baseline(pretrained=True)
		model = model.to(device)
	
	elif args.backbone == 'plip':
		model_name = '/path/to/your/plip'
		model,_,_ = load_plip_model(name=model_name, auth_token=None)
		model = model.to(device)
	elif args.backbone == 'vit_s_imagenet':
		model = timm.create_model('vit_small_patch16_224', pretrained=True)
		model.head = nn.Identity()
		model = model.to(device)
	elif args.backbone == 'uni':
		model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True)
		local_dir = '/path/to/your/uni'
		model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location=device), strict=True)
		model = model.to(device)

	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		
	model.eval()
	total = len(bags_dataset)

	for bag_candidate_idx in range(total):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 
		new_slide_id = slide_id.replace('normal_', '').replace('tumor_', '')

			

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()
		if slide_file_path.endswith('.sdpc'):
			try:
				import sdpc
			except:
				raise ImportError('sdpc not installed')
			wsi = sdpc.Sdpc(slide_file_path)
		else:
			wsi = openslide.open_slide(slide_file_path)
		output_file_path = compute_w_loader(args,h5_file_path, output_path, wsi, 
		model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
		custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size)
		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
		file = h5py.File(output_file_path, "r")

		features = file['features'][:]
		print('features size: ', features.shape)
		print('coordinates size: ', file['coords'].shape)
		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))



