from __future__ import print_function
import numpy as np
import argparse
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(grandparent_dir)
import torch
import torch.nn as nn
import pdb
import pandas as pd
from math import floor
from types import SimpleNamespace
from collections import namedtuple
import h5py
import yaml
from feature_extractor.clam_base_utils.wsi_core.batch_process_utils import initialize_df
from vis_utils.heatmap_utils import drawHeatmap, compute_from_patches,initialize_wsi,infer_single_slide
from feature_extractor.clam_base_utils.wsi_core.wsi_utils import sample_rois
from feature_extractor.clam_base_utils.utils.file_utils import save_hdf5
from feature_extractor.clam_base_utils.utils.utils import *
from feature_extractor.clam_base_utils.utils.model_utils import get_backbone,get_transforms
from feature_extractor.clam_base_utils.utils.eval_utils import initiate_model as initiate_model
from tqdm import tqdm
from utils.model_utils import get_model_from_yaml
from utils.yaml_utils import read_yaml


def load_params(df_entry, params):
    for key in params.keys():
        if key in df_entry.index:
            dtype = type(params[key])
            val = df_entry[key] 
            val = dtype(val)
            if isinstance(val, str):
                if len(val) > 0:
                    params[key] = val
            elif not np.isnan(val):
                params[key] = val
            else:
                pdb.set_trace()
    return params

def main(args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = args.heatmap_config_yaml
    config_dict = yaml.safe_load(open(config_path, 'r'))

    for key, value in config_dict.items():
        if isinstance(value, dict):
            print('\n'+key)
            for value_key, value_value in value.items():
                print (value_key + " : " + str(value_value))
        else:
            print ('\n'+key + " : " + str(value))
            
    args = config_dict
    patch_args = argparse.Namespace(**args['patching_arguments'])
    data_args = argparse.Namespace(**args['data_arguments'])
    exp_args = argparse.Namespace(**args['exp_arguments'])
    heatmap_args = argparse.Namespace(**args['heatmap_arguments'])
    sample_args = argparse.Namespace(**args['sample_arguments'])
    model_args = argparse.Namespace(**args['model_arguments'])
    encoder_args = argparse.Namespace(**args['encoder_arguments'])
    
    patch_size = tuple([patch_args.patch_size for i in range(2)])
    step_size = tuple((np.array(patch_size) * (1 - patch_args.overlap)).astype(int))
    print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1], patch_args.overlap, step_size[0], step_size[1]))

    preset = data_args.preset
    def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False, 
                      'keep_ids': 'none', 'exclude_ids':'none'}
    def_filter_params = {'a_t':50.0, 'a_h': 8.0, 'max_n_holes':10}
    def_vis_params = {'vis_level': -1, 'line_thickness': 250}
    def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    if preset is not None:
        preset_df = pd.read_csv(preset)
        for key in def_seg_params.keys():
            def_seg_params[key] = preset_df.loc[0, key]

        for key in def_filter_params.keys():
            def_filter_params[key] = preset_df.loc[0, key]

        for key in def_vis_params.keys():
            def_vis_params[key] = preset_df.loc[0, key]

        for key in def_patch_params.keys():
            def_patch_params[key] = preset_df.loc[0, key]
            
    # Load model
    mil_model_ckpt_path = model_args.ckpt_path
    print('\nmil_model_ckpt_path: {}'.format(mil_model_ckpt_path))
    mil_model_yaml_path = model_args.yaml_path
    mil_yaml_args = read_yaml(mil_model_yaml_path)
    mil_model_name = mil_yaml_args.General.MODEL_NAME
    if mil_model_name == 'CDP_MIL':
        raise NotImplementedError("CDP_MIL model is not supported for attention map visualization now.")
    if mil_model_name == 'DTFD_MIL':
        classifier,attention,dimReduction,attCls = get_model_from_yaml(mil_yaml_args)
        attention_kwargs = {'total_instance': mil_yaml_args.Model.total_instance, 'num_Group': mil_yaml_args.Model.num_Group, 'grad_clipping': mil_yaml_args.Model.grad_clipping, 'distill': mil_yaml_args.Model.distill}
        state_dict = torch.load(mil_model_ckpt_path,weights_only=True)
        classifier.load_state_dict(state_dict['classifier'])
        attention.load_state_dict(state_dict['attention'])
        dimReduction.load_state_dict(state_dict['dimReduction'])
        attCls.load_state_dict(state_dict['attCls'])
        mil_model = [classifier,attention,dimReduction,attCls]
        mil_model = tuple([model.to(device).eval() for model in mil_model])
    else:
        attention_kwargs = {}
        mil_model = get_model_from_yaml(mil_yaml_args)
        mil_model = mil_model.to(device)
        mil_model.load_state_dict(torch.load(mil_model_ckpt_path,weights_only=True))
        mil_model.eval()
    # Load feature extractor
    feature_extractor_name = encoder_args.model_name
    feature_extractor = get_backbone(feature_extractor_name, device, encoder_args.model_weights_dir)
    feature_extractor.eval()
    feature_extractor_transforms = get_transforms(feature_extractor_name)
    feature_extractor_transforms.transforms.insert(0, transforms.Resize((encoder_args.target_img_size, encoder_args.target_img_size)))
    print('Done!')
    

    os.makedirs(exp_args.save_dir, exist_ok=True)
    # total_save_dir: save_dir/exp_code
    total_save_dir = os.path.join(exp_args.save_dir, exp_args.save_exp_code)
    os.makedirs(total_save_dir, exist_ok=True)
    
    blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': patch_size, 
    'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}
    wsi_csv_path = data_args.wsi_csv
    wsi_df = pd.read_csv(wsi_csv_path)
    wsi_paths = wsi_df['wsi_path']

    for i in tqdm(range(len(wsi_paths))):
        slide_path = wsi_paths[i]
        slide_name = os.path.basename(slide_path)
        now_ext = slide_name.split('.')[-1]
        slide_id = slide_name[0:-len(now_ext)-1]

        # heatmap_save_dir = os.path.join(exp_args.save_dir, exp_args.save_exp_code)
        # os.makedirs(heatmap_save_dir, exist_ok=True)

        # process_save_dir = os.path.join(raw_save_dir, exp_args.save_exp_code, slide_id)
        # os.makedirs(process_save_dir, exist_ok=True)
        
        # slide_save_dir: total_save_dir/slide_id
        slide_save_dir = os.path.join(total_save_dir, slide_id)
        os.makedirs(slide_save_dir, exist_ok=True)
        process_save_dir = os.path.join(slide_save_dir, 'process')
        os.makedirs(process_save_dir, exist_ok=True)
        heatmap_save_dir = os.path.join(slide_save_dir, 'heatmaps')
        os.makedirs(heatmap_save_dir, exist_ok=True)
        
        if heatmap_args.use_roi:
            x1, x2 = wsi_df.loc[i, 'x1'], wsi_df.loc[i, 'x2']
            y1, y2 = wsi_df.loc[i, 'y1'], wsi_df.loc[i, 'y2']
            top_left = (int(x1), int(y1))
            bot_right = (int(x2), int(y2))
        else:
            top_left = None
            bot_right = None
        
        print('slide id: ', slide_id)
        print('top left: ', top_left, ' bot right: ', bot_right)


        mask_file = os.path.join(process_save_dir, slide_id+'_mask.pkl')
        
        # Load segmentation and filter parameters
        seg_params = def_seg_params.copy()
        filter_params = def_filter_params.copy()
        vis_params = def_vis_params.copy()

        seg_params = load_params(wsi_df.loc[i], seg_params)
        filter_params = load_params(wsi_df.loc[i], filter_params)
        vis_params = load_params(wsi_df.loc[i], vis_params)

        keep_ids = str(seg_params['keep_ids'])
        if len(keep_ids) > 0 and keep_ids != 'none':
            seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
        else:
            seg_params['keep_ids'] = []

        exclude_ids = str(seg_params['exclude_ids'])
        if len(exclude_ids) > 0 and exclude_ids != 'none':
            seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
        else:
            seg_params['exclude_ids'] = []

        for key, val in seg_params.items():
            print('{}: {}'.format(key, val))

        for key, val in filter_params.items():
            print('{}: {}'.format(key, val))

        for key, val in vis_params.items():
            print('{}: {}'.format(key, val))
        
        print('Initializing WSI object')
        wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
        print('Done!')

        wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]

        # the actual patch size for heatmap visualization should be the patch size * downsample factor * custom downsample factor
        vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))

        block_map_save_path = os.path.join(process_save_dir, '{}_blockmap.h5'.format(slide_id))
        mask_path = os.path.join(process_save_dir, '{}_mask.jpg'.format(slide_id))
        if vis_params['vis_level'] < 0:
            best_level = wsi_object.wsi.get_best_level_for_downsample(32)
            vis_params['vis_level'] = best_level
        mask = wsi_object.visWSI(**vis_params, number_contours=True)
        mask.save(mask_path)
        
        features_path = os.path.join(process_save_dir, slide_id+'.pt')
        h5_path = os.path.join(process_save_dir, slide_id+'.h5')
    

        ##### check if h5_features_file exists ######
        if not os.path.isfile(h5_path) :
            _, _, wsi_object = compute_from_patches(wsi_object=wsi_object, 
                                            mil_model=mil_model,
                                            mil_model_name=mil_model_name, 
                                            feature_extractor=feature_extractor,
                                            feature_extractor_transforms=feature_extractor_transforms,
                                            feature_extractor_name=feature_extractor_name, 
                                            batch_size=exp_args.batch_size, **blocky_wsi_kwargs, 
                                            attn_save_path=None, feat_save_path=h5_path, 
                                            ref_scores=None,attention_kwargs=attention_kwargs)				
        
        ##### check if pt_features_file exists ######
        if not os.path.isfile(features_path):
            file = h5py.File(h5_path, "r")
            features = torch.tensor(file['features'][:])
            torch.save(features, features_path)
            file.close()

        # load features 
        features = torch.load(features_path)
        wsi_df.loc[i, 'bag_size'] = len(features)
        
        wsi_object.saveSegmentation(mask_file)
        A = infer_single_slide(mil_model, mil_model_name, features,attention_kwargs)
        del features
        
        if not os.path.isfile(block_map_save_path): 
            file = h5py.File(h5_path, "r")
            coords = file['coords'][:]
            file.close()
            asset_dict = {'attention_scores': A, 'coords': coords}
            block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')
        
        # save top 3 predictions

        # os.makedirs('heatmaps/results/', exist_ok=True)

        wsi_df.to_csv('{}/{}.csv'.format(total_save_dir,exp_args.save_exp_code), index=False)
        
        file = h5py.File(block_map_save_path, 'r')
        dset = file['attention_scores']
        coord_dset = file['coords']
        scores = dset[:]
        coords = coord_dset[:]
        file.close()

        samples = sample_args.samples
        for sample in samples:
            if sample['sample']:
                sample_save_dir =  os.path.join(slide_save_dir, f'{sample["name"]}_samples')
                os.makedirs(sample_save_dir, exist_ok=True)
                print('sampling {}'.format(sample['name']))
                sample_results = sample_rois(scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'], 
                    score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))
                for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
                    print('coord: {} score: {:.3f}'.format(s_coord, s_score))
                    patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level, (patch_args.patch_size, patch_args.patch_size)).convert('RGB')
                    patch.save(os.path.join(sample_save_dir, '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, slide_id, s_coord[0], s_coord[1], s_score)))

        wsi_kwargs = {'top_left': top_left, 'bot_right': bot_right, 'patch_size': patch_size, 'step_size': step_size, 
        'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}

        heatmap_save_name = '{}_blockmap.tiff'.format(slide_id)
        if os.path.isfile(os.path.join(process_save_dir, heatmap_save_name)):
            pass
        else:
            heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
                            thresh=-1, patch_size = vis_patch_size, convert_to_percentiles=True)
        
            heatmap.save(os.path.join(process_save_dir, '{}_blockmap.png'.format(slide_id)))
            del heatmap

        save_path = os.path.join(process_save_dir, '{}_{}_roi_{}.h5'.format(slide_id, patch_args.overlap, heatmap_args.use_roi))

        if heatmap_args.use_ref_scores:
            ref_scores = scores
        else:
            ref_scores = None
        
        if heatmap_args.calc_heatmap:
            compute_from_patches(wsi_object=wsi_object, 
                                mil_model=mil_model,
                                mil_model_name=mil_model_name, 
                                feature_extractor=feature_extractor,
                                feature_extractor_transforms=feature_extractor_transforms,
                                feature_extractor_name=feature_extractor_name, 
                                batch_size=exp_args.batch_size, **wsi_kwargs, 
                                attn_save_path=save_path, ref_scores=ref_scores,attention_kwargs=attention_kwargs)

        if not os.path.isfile(save_path):
            print('heatmap {} not found'.format(save_path))
            if heatmap_args.use_roi:
                save_path_full = os.path.join(process_save_dir, '{}_{}_roi_False.h5'.format(slide_id, patch_args.overlap))
                print('found heatmap for whole slide')
                save_path = save_path_full
            else:
                continue
        
        with h5py.File(save_path, 'r') as file:
            file = h5py.File(save_path, 'r')
            dset = file['attention_scores']
            coord_dset = file['coords']
            scores = dset[:]
            coords = coord_dset[:]

        heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': heatmap_args.vis_level, 'blur': heatmap_args.blur, 'custom_downsample': heatmap_args.custom_downsample}
        if heatmap_args.use_ref_scores:
            heatmap_vis_args['convert_to_percentiles'] = False

        heatmap_save_name = '{}_{}_roi_{}_blur_{}_rs_{}_bc_{}_a_{}_l_{}_bi_{}_{}.{}'.format(slide_id, float(patch_args.overlap), int(heatmap_args.use_roi),
                                                                                        int(heatmap_args.blur), 
                                                                                        int(heatmap_args.use_ref_scores), int(heatmap_args.blank_canvas), 
                                                                                        float(heatmap_args.alpha), int(heatmap_args.vis_level), 
                                                                                        int(heatmap_args.binarize), float(heatmap_args.binary_thresh), heatmap_args.save_ext)


        if os.path.isfile(os.path.join(heatmap_save_dir, heatmap_save_name)):
            pass
        
        else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
            heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object,  
                                  cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, **heatmap_vis_args, 
                                  binarize=heatmap_args.binarize, 
                                    blank_canvas=heatmap_args.blank_canvas,
                                    thresh=heatmap_args.binary_thresh,  patch_size = vis_patch_size,
                                    overlap=patch_args.overlap, 
                                    top_left=top_left, bot_right = bot_right)
            if heatmap_args.save_ext == 'jpg':
                heatmap.save(os.path.join(heatmap_save_dir, heatmap_save_name), quality=100)
            else:
                heatmap.save(os.path.join(heatmap_save_dir, heatmap_save_name))
        
        if heatmap_args.save_orig:
            if heatmap_args.vis_level >= 0:
                vis_level = heatmap_args.vis_level
            else:
                vis_level = vis_params['vis_level']
            heatmap_save_name = '{}_orig_{}.{}'.format(slide_id,int(vis_level), heatmap_args.save_ext)
            if os.path.isfile(os.path.join(heatmap_save_dir, heatmap_save_name)):
                pass
            else:
                heatmap = wsi_object.visWSI(vis_level=vis_level, view_slide_only=True, custom_downsample=heatmap_args.custom_downsample)
                if heatmap_args.save_ext == 'jpg':
                    heatmap.save(os.path.join(heatmap_save_dir, heatmap_save_name), quality=100)
                else:
                    heatmap.save(os.path.join(heatmap_save_dir, heatmap_save_name))

    with open(os.path.join(total_save_dir, 'heatmap_config.yaml'), 'w') as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False)



    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Heatmap inference script')
    parser.add_argument('--heatmap_config_yaml', type=str, default= '/path/to/your/heatmap_yaml', help="path of heatmap_config_template.yaml")
    args = parser.parse_args()
    main(args)
    
    
    
    
    
    