# internal imports
from clam_base_utils.wsi_core.WholeSlideImage import WholeSlideImage
from clam_base_utils.wsi_core.wsi_utils import StitchCoords
from clam_base_utils.wsi_core.batch_process_utils import initialize_df
# other imports
import os
import numpy as np
import time
import argparse
import tqdm
import h5py
import pdb
import pandas as pd
from concurrent.futures import ThreadPoolExecutor,as_completed
def recursion_get_wsis(source, ext_list: list):
	slides = []
	for root, dirs, files in os.walk(source):
		for file in files:
			if file.endswith(tuple(ext_list)):
				slides.append(os.path.join(root, file))
	return slides

def adjust_coords_order(h5_path):
	with h5py.File(h5_path, 'r') as source_file:
		coords_dataset = source_file['coords']
		coords_data = coords_dataset[:]
		sorted_indices = np.lexsort((coords_data[:, 1], coords_data[:, 0]))
		coords_data = coords_data[sorted_indices]
		coords_attrs = dict(coords_dataset.attrs)
		new_coords_data = coords_data + 1
	with h5py.File(h5_path, 'w') as new_file:
		new_coords_dataset = new_file.create_dataset('coords', data=new_coords_data)
		for key, value in coords_attrs.items():
			new_coords_dataset.attrs[key] = value


def magnification_to_level_transfer(target_magnification:int, wsi_object:WholeSlideImage):
	"""
	Convert magnification to level
	"""
	ori_magnification = int(wsi_object.get_original_magnification())
	if target_magnification > ori_magnification:
		return None,None
	elif target_magnification == ori_magnification:
		return 0,None
	else:
		level_downsamples = wsi_object.level_down
		num_levels = len(level_downsamples)
		wsi_zoom = int(level_downsamples[1])//int(level_downsamples[0]) # 2 or 4 for most cases
		has_magnifications = [ori_magnification/(wsi_zoom**k) for k in range(num_levels)]
		if float(target_magnification) in has_magnifications:
			return has_magnifications.index(float(target_magnification)),None
		else:
			reverse_maginifications = has_magnifications[::-1]
			for now_mag in reverse_maginifications:
				if target_magnification < now_mag:
					close_mag = now_mag
					close_level = has_magnifications.index(close_mag)
					scale = close_mag/target_magnification
					return close_level,scale
					
def stitching(file_path, wsi_object, downscale = 64):
	start = time.time()
	heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0,0,0), alpha=-1, draw_grid=False)
	total_time = time.time() - start
	
	return heatmap, total_time

def segment(WSI_object, seg_params = None, filter_params = None, mask_file = None):
	### Start Seg Timer
	start_time = time.time()
	# Use segmentation file
	if mask_file is not None:
		WSI_object.initSegmentation(mask_file)
	# Segment	
	else:
		WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

	### Stop Seg Timers
	seg_time_elapsed = time.time() - start_time   
	return WSI_object, seg_time_elapsed

def save_one_img(wsi, X,Y, patch_level, patch_size, real_patch_size, this_patch_img_save_dir):
	patch = wsi.read_region((X,Y), patch_level, (patch_size, patch_size)).convert('RGB').resize((real_patch_size, real_patch_size))
	patch.save(os.path.join(this_patch_img_save_dir, '{}_{}.jpg'.format(X,Y)))
def save_patch_img_to_dir(WSI_object, h5_file, patch_level, patch_size, real_patch_size, this_patch_img_save_dir,multiprocess_save_patch):
    h5_file = h5py.File(h5_file, 'r')
    coords = h5_file['coords'][:]
    wsi = WSI_object.getOpenSlide()
    with ThreadPoolExecutor(max_workers=multiprocess_save_patch) as executor:
        futures = []
        for coord in coords:
            X, Y = int(coord[0]), int(coord[1])
            futures.append(
                executor.submit(save_one_img, wsi, X, Y, patch_level, patch_size, real_patch_size, this_patch_img_save_dir)
            )
        for _ in tqdm.tqdm(as_completed(futures), total=len(futures), desc="saving patch images"):
            pass


def split_index_by_num_workers(num_workers, total_num):
    results = []
    base = total_num // num_workers
    remainder = total_num % num_workers
    start = 0
    for i in range(num_workers):
        end = start + base + (1 if i < remainder else 0)
        results.append(list(range(start, end)))
        start = end
    results = [result for result in results if len(result) != 0]
    return results
	

def patching(WSI_object, **kwargs):
	### Start Patch Timer
	start_time = time.time()
	# Patch
	file_path = WSI_object.process_contours(strict_control=True,**kwargs)
	### Stop Patch Timer
	patch_time_elapsed = time.time() - start_time
	return file_path, patch_time_elapsed

def patching_index_list(index_list:list,df,process_stack,source,patch_level,patch_size,step_size,level_or_magnification_control,magnification,
						use_default_params,legacy_support,seg,save_mask,stitch,patch,save_patch_img,mask_save_dir,patch_save_dir,stitch_save_dir,multiprocess_save_patch,save_dir,patch_img_save_dir):
	for i in tqdm.tqdm(index_list):
		idx = process_stack.index[i]
		slide = process_stack.loc[idx, 'slide_id']
		# print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i, total))
		print('processing {}'.format(slide))

		df.loc[idx, 'process'] = 0
		slide_id = '.'.join(os.path.basename(slide).split('.')[:-1])
		now_patch_img_save_dir = None
		if patch_img_save_dir != None:
			now_patch_img_save_dir = os.path.join(patch_img_save_dir,slide_id)
			os.makedirs(now_patch_img_save_dir,exist_ok=True)
		now_ext = '.' + os.path.basename(slide).split('.')[-1]

		if os.path.exists(os.path.join(patch_save_dir, slide_id + '.h5')):
			print('{} already exist in destination location, skipped'.format(slide_id))
			df.loc[idx, 'status'] = 'already_exist'
			continue

		# Inialize WSI
		full_path = os.path.join(source, slide)
		try:
			WSI_object = WholeSlideImage(full_path)
		except:
			print('failed to open {}'.format(slide))
			df.loc[idx, 'status'] = 'failed_open'
			continue
		# level_or_magnification_control_transfer
		now_WSI_level = patch_level
		now_WSI_patch_size = patch_size
		now_WSI_step_size = step_size
		if level_or_magnification_control == 'magnification':
			now_level,now_scale = magnification_to_level_transfer(magnification,WSI_object)
			if now_level == None:
				print('magnification {} is not available in {}'.format(magnification,slide_id + now_ext))
				df.loc[idx, 'status'] = 'failed_for_mag'
				continue
			elif now_scale == None:
				now_WSI_level = now_level
			else:
				now_WSI_level = now_level
				now_WSI_patch_size = int(patch_size*now_scale)
				now_WSI_step_size = int(step_size*now_scale)
			print('magnification {} is transfered to level {} in {}'.format(magnification,now_WSI_level,slide_id + now_ext))
			print('patch_size is transfered to {} in {}'.format(now_WSI_patch_size,slide_id))
		elif level_or_magnification_control == 'level':
			total_levels = len(WSI_object.level_downsamples)
			if patch_level >= total_levels:
				print('level {} is not available in {}'.format(patch_level,slide_id))
				df.loc[idx, 'status'] = 'failed_for_level'
				continue
			
		if use_default_params:
			current_vis_params = vis_params.copy()
			current_filter_params = filter_params.copy()
			current_seg_params = seg_params.copy()
			current_patch_params = patch_params.copy()
			
		else:
			current_vis_params = {}
			current_filter_params = {}
			current_seg_params = {}
			current_patch_params = {}

			for key in vis_params.keys():
				if legacy_support and key == 'vis_level':
					df.loc[idx, key] = -1
				current_vis_params.update({key: df.loc[idx, key]})

			for key in filter_params.keys():
				if legacy_support and key == 'a_t':
					old_area = df.loc[idx, 'a']
					seg_level = df.loc[idx, 'seg_level']
					scale = WSI_object.level_downsamples[seg_level]
					adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
					current_filter_params.update({key: adjusted_area})
					df.loc[idx, key] = adjusted_area
				current_filter_params.update({key: df.loc[idx, key]})

			for key in seg_params.keys():
				if legacy_support and key == 'seg_level':
					df.loc[idx, key] = -1
				current_seg_params.update({key: df.loc[idx, key]})

			for key in patch_params.keys():
				current_patch_params.update({key: df.loc[idx, key]})

		if current_vis_params['vis_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_vis_params['vis_level'] = 0
			
			else:	
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_vis_params['vis_level'] = best_level

		if current_seg_params['seg_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_seg_params['seg_level'] = 0
			
			else:
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_seg_params['seg_level'] = best_level

		keep_ids = str(current_seg_params['keep_ids'])
		if keep_ids != 'none' and len(keep_ids) > 0:
			str_ids = current_seg_params['keep_ids']
			current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['keep_ids'] = []

		exclude_ids = str(current_seg_params['exclude_ids'])
		if exclude_ids != 'none' and len(exclude_ids) > 0:
			str_ids = current_seg_params['exclude_ids']
			current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['exclude_ids'] = []

		w, h = WSI_object.level_dim[current_seg_params['seg_level']] 
		if w * h > 1e8:
			print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
			df.loc[idx, 'status'] = 'failed_seg'
			continue

		df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
		df.loc[idx, 'seg_level'] = current_seg_params['seg_level']


		seg_time_elapsed = -1
		if seg:
			WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params) 

		if save_mask:
			mask = WSI_object.visWSI(**current_vis_params)
			mask_path = os.path.join(mask_save_dir, slide_id+'.jpg')
			mask.save(mask_path)

		patch_time_elapsed = -1 # Default time
		if patch:
			current_patch_params.update({'patch_level': now_WSI_level, 'patch_size': now_WSI_patch_size, 'real_patch_size':patch_size, 'step_size': now_WSI_step_size, 
											'save_path': patch_save_dir})
			file_path, patch_time_elapsed = patching(WSI_object = WSI_object,  **current_patch_params,)
			h5_path = os.path.join(patch_save_dir, slide_id+'.h5')
			adjust_coords_order(h5_path)
			if save_patch_img:
				save_patch_img_to_dir(WSI_object, h5_path, now_WSI_level, now_WSI_patch_size, patch_size,now_patch_img_save_dir,multiprocess_save_patch)
		stitch_time_elapsed = -1
		if stitch:
			file_path = os.path.join(patch_save_dir, slide_id+'.h5')
			if os.path.isfile(file_path):
				heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
				stitch_path = os.path.join(stitch_save_dir, slide_id+'.jpg')
				heatmap.save(stitch_path)

		print("segmentation took {} seconds".format(seg_time_elapsed))
		print("patching took {} seconds".format(patch_time_elapsed))
		print("stitching took {} seconds".format(stitch_time_elapsed))
		df.loc[idx, 'status'] = 'processed'
		df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'),index=False)

	return 0


def seg_and_patch(source, source_csv, save_dir, patch_save_dir,patch_img_save_dir,mask_save_dir, stitch_save_dir, 
				  patch_size = 256, step_size = 256, 
				  seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'},
				  filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}, 
				  vis_params = {'vis_level': -1, 'line_thickness': 500},
				  patch_params = {'use_padding': True, 'contour_fn': 'four_pt'},
      		      level_or_magnification_control = 'level',
				  patch_level = 0,
				  magnification = None,
				  use_default_params = False, 
				  seg = False, save_mask = True, 
				  stitch= False, 
				  patch = False, auto_skip=True, process_list = None, ext_list = None,
				  multiprocess_slide = 1,multiprocess_save_patch = 16):

	# slides = sorted(os.listdir(source))
	# slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]
	assert ext_list is not None, 'ext_list must be provided'
	if os.path.exists(source_csv):
		source_df = pd.read_csv(source_csv)
		slides = source_df['wsi_path'].to_list()
	else:
		slides = recursion_get_wsis(source, ext_list)
	if process_list is None:
		df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
	
	else:
		df = pd.read_csv(process_list)
		df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

	mask = df['process'] == 1
	process_stack = df[mask]

	total = len(process_stack)

	legacy_support = 'a' in df.keys()
	if legacy_support:
		print('detected legacy segmentation csv file, legacy support enabled')
		df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
		'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
		'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
		'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
		'contour_fn': np.full((len(df)), patch_params['contour_fn'])})
	df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
	num_workers = multiprocess_slide
	indexs_list = split_index_by_num_workers(num_workers,total)
	num_workers = len(indexs_list)
	save_patch_img = False
	if patch_img_save_dir != None:
		save_patch_img = True
	with ThreadPoolExecutor(max_workers=num_workers) as executor:
		futures = [executor.submit(patching_index_list,index_list,df,process_stack,source,patch_level,patch_size,step_size,level_or_magnification_control,magnification,
						use_default_params,legacy_support,seg,save_mask,stitch,patch,save_patch_img,mask_save_dir,patch_save_dir,stitch_save_dir,multiprocess_save_patch,save_dir,patch_img_save_dir) for index_list in indexs_list]
		for future in futures:
			future.result()
	df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)

	return 0,0

parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', default = '',type = str,
					help='path to folder containing raw wsi image files')
parser.add_argument('--source_csv', default='',type= str,
					help='csv contain wsi paths, head -> wsi_path, prior control than --source')
parser.add_argument('--multiprocess_slide', default=1,type=int,help='multprocess threads')
parser.add_argument('--step_size', type = int, default=256,
					help='step_size')
parser.add_argument('--patch_size', type = int, default=256,
					help='patch_size')
parser.add_argument('--patch', default=True, action='store_true')
parser.add_argument('--seg', default=True, action='store_true')
parser.add_argument('--use_otsu', default=False, action='store_true')
parser.add_argument('--stitch', default=True, action='store_true')
parser.add_argument('--save_patch_img', default=True, action='store_true')
parser.add_argument('--multiprocess_save_patch', default=32,type=int,help='multprocess threads')
parser.add_argument('--save_ext', default='.jpg', choices=['.jpg','.png'])
parser.add_argument('--ext_list', default=['.svs','.mrxs'], type=list,help='list of file extensions to process, .svs, .tif, .sdpc, .ndpi, .mrxs .etc')
parser.add_argument('--no_auto_skip', default=True, action='store_false')
parser.add_argument('--save_dir', default='', type = str,
					help='directory to save processed data')
parser.add_argument('--preset', default='./MIL_BASELINE-main/feature_extractor/presets/tcga.csv', type=str,
					help='predefined profile of default segmentation and filter parameters (.csv)')
parser.add_argument('--level_or_magnification_control', type=str, default='level', choices=['level', 'magnification'],
                    help='control whether to use patch level or magnification for segmentation and visualization')
parser.add_argument('--patch_level', type=int, default=3, 
					help='downsample level at which to patch')
parser.add_argument('--magnification', type=int, default=30,help='magnification level at which to patch')
parser.add_argument('--process_list',  type = str, default=None,
					help='name of list of images to process with parameters (.csv)')

if __name__ == '__main__':
	args = parser.parse_args()
	patch_save_dir = os.path.join(args.save_dir, 'patches')
	if args.save_patch_img:
		patch_img_save_dir = os.path.join(args.save_dir,'patch_imgs')
	else:
		patch_img_save_dir = None
	mask_save_dir = os.path.join(args.save_dir, 'masks')
	stitch_save_dir = os.path.join(args.save_dir, 'stitches')

	if args.process_list:
		process_list = os.path.join(args.save_dir, args.process_list)
	else:
		process_list = None
	print('source: ', args.source)
	print('patch_save_dir: ', patch_save_dir)
	if args.save_patch_img:
		print('patch_imgs_save_dir:', patch_img_save_dir)
		os.makedirs(patch_img_save_dir,exist_ok=True)
	print('mask_save_dir: ', mask_save_dir)
	print('stitch_save_dir: ', stitch_save_dir)
	
	directories = {'source': args.source, 
				   'source_csv': args.source_csv,
				   'save_dir': args.save_dir,
				   'patch_save_dir': patch_save_dir, 
				   'patch_img_save_dir':patch_img_save_dir,
				   'mask_save_dir' : mask_save_dir, 
				   'stitch_save_dir': stitch_save_dir} 

	for key, val in directories.items():
		print("{} : {}".format(key, val))
		if key not in ['source','source_csv','patch_img_save_dir']:  
			os.makedirs(val, exist_ok=True)

	seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'}
	filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}
	vis_params = {'vis_level': -1, 'line_thickness': 250}
	patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

	if args.preset:
		preset_df = pd.read_csv(os.path.join('presets', args.preset))
		for key in seg_params.keys():
			seg_params[key] = preset_df.loc[0, key]

		for key in filter_params.keys():
			filter_params[key] = preset_df.loc[0, key]

		for key in vis_params.keys():
			vis_params[key] = preset_df.loc[0, key]

		for key in patch_params.keys():
			patch_params[key] = preset_df.loc[0, key]
	
	parameters = {'seg_params': seg_params,
				  'filter_params': filter_params,
	 			  'patch_params': patch_params,
				  'vis_params': vis_params}
	seg_params['use_otsu'] = args.use_otsu
	print(parameters)

	seg_times, patch_times = seg_and_patch(**directories, **parameters,
											patch_size = args.patch_size, step_size=args.step_size, 
											seg = args.seg,  use_default_params=False, save_mask = True, 
											stitch= args.stitch,
											patch_level=args.patch_level, patch = args.patch,
											process_list = process_list, auto_skip=args.no_auto_skip, 
           									ext_list = args.ext_list, level_or_magnification_control = args.level_or_magnification_control,
                    						magnification = args.magnification, multiprocess_slide= args.multiprocess_slide,
											multiprocess_save_patch=args.multiprocess_save_patch)
