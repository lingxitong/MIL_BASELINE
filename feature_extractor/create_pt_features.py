import torch
import torch.nn as nn
from math import floor
import os
from torchvision import transforms
import time
from clam_base_utils.clam_datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
import argparse
from clam_base_utils.utils.utils import collate_features
from clam_base_utils.utils.file_utils import save_hdf5
import h5py
from clam_base_utils.utils.model_utils import get_transforms, get_backbone, feature_extractor_adapter
import openslide


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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

	custom_transforms = get_transforms(args.backbone)
     
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
			features = feature_extractor_adapter(model, batch,args.backbone)
			features = features.cpu().numpy()

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path


def main(args):

	print('initializing dataset')
	csv_path = os.path.join(args.data_h5_dir, 'process_list_autogen.csv')
	if csv_path is None:
		raise NotImplementedError
	custom_downsample = 1
	bags_dataset = Dataset_All_Bags(csv_path)
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))
	print('loading model checkpoint')
	model = get_backbone(args.backbone, device, args.pretrained_weights_dir)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
	model.eval()
	total = len(bags_dataset)
	for bag_candidate_idx in range(total):
		now_slide_ext = '.'+bags_dataset[bag_candidate_idx].split('.')[-1]
		slide_file_path = bags_dataset[bag_candidate_idx]
		slide_id = os.path.basename(bags_dataset[bag_candidate_idx]).split(now_slide_ext)[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()
		if slide_file_path.endswith('.sdpc'):
			try:
				import opensdpc
			except:
				raise ImportError('opensdpc has not been installed, please run pip install opensdpc (https://github.com/WonderLandxD/opensdpc)')
			try:
				wsi = opensdpc.OpenSdpc(slide_file_path)
			except:
				print(f'{slide_file_path} can not open')
				continue
		else:
			try:
				wsi = openslide.open_slide(slide_file_path)
			except:
				print(f'{slide_file_path} can not open')
				continue
		try:
			output_file_path = compute_w_loader(args,h5_file_path, output_path, wsi, 
		model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
		custom_downsample=custom_downsample, target_patch_size=args.target_patch_size)
		except:
			continue
		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
		file = h5py.File(output_file_path, "r")

		features = file['features'][:]
		print('features size: ', features.shape)
		print('coordinates size: ', file['coords'].shape)
		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', default = '/mnt/net_sda/lxt/MB工程化/save_dir' ,type=str)
parser.add_argument('--feat_dir', type=str, default='/mnt/net_sda/lxt/MB工程化/feat_dir')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
parser.add_argument('--backbone', default='uni',type=str,choices=['vit_s_imagenet','resnet50_imagenet','plip','conch','uni','ctranspath','gigapath','virchow','virchow_v2','conch_v1_5'], help='backbone model')
parser.add_argument('--pretrained_weights_dir', type=str, default='/mnt/net_sda/lxt/MB工程化/UNI_weights', help='dir to the pretrained backbone')
args = parser.parse_args()
main(args)
