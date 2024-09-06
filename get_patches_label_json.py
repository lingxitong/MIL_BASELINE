import os
import argparse
import pandas as pd
import h5py
import json
import openslide
from tqdm import tqdm
import numpy as np
from xml.etree import ElementTree as ET
from shapely.geometry import Polygon, box

def is_intersection(xmin, ymin, xmax, ymax, polygon_coords: Polygon):
    rect = box(xmin, ymin, xmax, ymax)
    polygon = Polygon(polygon_coords)
    return rect.intersects(polygon)

def parse_annotations_to_list(xml_ann_path):
    # parse xml file to list of annotations
    tree = ET.parse(xml_ann_path)
    root = tree.getroot()
    annotations_list = []
    for annotation in root.find('Annotations').findall('Annotation'):
        part_of_group = str.lower(annotation.attrib['PartOfGroup'])
        coordinates_list = []
        for coordinate in annotation.find('Coordinates').findall('Coordinate'):
            x = float(coordinate.attrib['X'])
            y = float(coordinate.attrib['Y'])
            coordinates_list.append([x, y])
        annotation_dict = {'groups':part_of_group,'coords':coordinates_list}
        annotations_list.append(annotation_dict)
    return annotations_list

# this script only work for the dataset with same init Magnification
def get_patch_label(parser_polygons,polygons_labels_id,ori_coords,consistency):
    xmin,ymin,xmax,ymax = ori_coords
    for polygon,label_id in zip(parser_polygons,polygons_labels_id):
        has_intersection = is_intersection(xmin, ymin, xmax, ymax, polygon)
        if has_intersection:
            return label_id
    if consistency:
        return 0
    return None

def find_element_containing_string(my_list, str):
    for ele in my_list:
        if str in ele:
            return ele
    return None

def get_ori_coords(coord,patch_level,patch_size):
    xmin,ymin = coord
    xmax = xmin + patch_size*(2**patch_level)
    ymax = ymin + patch_size*(2**patch_level)
    return xmin,ymin,xmax,ymax

def check_cover_relation(polygon1,polygon2):
    if polygon1.covers(polygon2):
        return 1
    elif polygon2.covers(polygon1):
        return 2
    else:
        return 0
    
def get_ann_polygons(parser_coords,parser_labels,label2id):
    ann_polygons = []
    for coords in parser_coords:
        closed_coords = coords + [coords[0]]
        ann_polygons.append(Polygon(closed_coords))
    # process intersection polygon
    L = len(ann_polygons)
    for i in range(L):
        for j in range(i+1,L):
            if i == j:
                continue
            if not ann_polygons[i].intersects(ann_polygons[j]):
                continue
            cover_relation = check_cover_relation(ann_polygons[i],ann_polygons[j])
            if cover_relation == 0:
                continue
            elif cover_relation == 1:
                new_polygon = Polygon(ann_polygons[i].exterior.coords, [ann_polygons[j].exterior.coords])
                ann_polygons[i] = new_polygon
            elif cover_relation == 2:
                new_polygon = Polygon(ann_polygons[j].exterior.coords, [ann_polygons[i].exterior.coords])
                ann_polygons[j] = new_polygon
            else:
                assert False
    return ann_polygons
 
def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(i) for i in obj]
    else:
        return obj

'''
h5_path_csv:
    h5_path,
    /path/to/slide_name_1.h5
    /path/to/slide_name_2.h5
    /path/to/slide_name_n.h5
'''

'''
mask_ann_csv: 
    xml_path,
    /path/to/slide_name_1.xml
    /path/to/slide_name_2.xml
    /path/to/slide_name_n.xml
'''

'''
label2id:
    you should ensure that the label names are same in the group of xml files 
'''

'''
consistency:
    for example, you will do the normal/macro/micro/itc classification,
    the slide label can be  normal/macro/micro/itc, and the patch label also can be normal/macro/micro/itc,
    you should set consistency to True
    
    for example, you will do the tumor subtyping classification,
    the slide label can be type1/type2/type3/type4, but the patch label can be normal/type1/type2/type3/type4,
    you should set consistency to False
    
    if consistency is True, The default 0 does not need to be annotated in the xml file, Unless it's hollow
'''

parser = argparse.ArgumentParser()
parser.add_argument('--patch_level',type=int,default=1,help='patch level of slide')
parser.add_argument('--patch_size',type=int,default=256,help='patch size of slide')
parser.add_argument('--label2id',type=dict,default={"normal":0,"tumor":1,"........."}, help='slide label to id dict'  )
parser.add_argument('--h5_path_csv',type=str,default='/path/to/your/h5_path.csv',help='h5 path csv file containing h5 path of slide')
parser.add_argument('--xml_path_csv',type=str,default='/path/to/your/xml_path.csv',help='mask annotation csv file containing xml file path of slide')
parser.add_argument('--consistency',type=str,default=True,help='consistency of slide and patch label')
parser.add_argument('--json_path',type=str,default='/path/to/your/patch_label.csv',help='output json file containing slide path and mask annotation path')


if __name__ == '__main__':
    # read arguments
    args = parser.parse_args()
    xml_path_csv = pd.read_csv(args.xml_path_csv)
    h5_path_csv = pd.read_csv(args.h5_path_csv)
    label2id = args.label2id
    json_path = args.json_path
    patch_level = args.patch_level
    patch_size = args.patch_size
    consistency = args.consistency
    
    # get slide_name with mask annotation
    xml_paths = xml_path_csv['xml_path'].values
    h5_paths = h5_path_csv['h5_path'].values
    xml_names = [os.path.basename(x).split('.')[0] for x in xml_paths]
    h5_names = [os.path.basename(x).split('.')[0] for x in h5_paths]
    slide_names_with_xml_and_h5 = [x for x in h5_names if x in xml_names]
    
    # begin get patch labels
    patch_labes_json = {'anns':[]}
    for slide_name_with_xml_and_h5 in tqdm(slide_names_with_xml_and_h5,desc=f'get patch labels'):
        h5_path = find_element_containing_string(h5_paths,slide_name_with_xml_and_h5)
        xml_path = find_element_containing_string(xml_paths,slide_name_with_xml_and_h5)
        one_ans = {'slide_name':slide_name_with_xml_and_h5,'patch_labels':[]}
        h5_file = h5py.File(h5_path,'r')
        coords = h5_file['coords']
        groups_and_anns = parse_annotations_to_list(xml_path)
        parser_groups = [x['groups'] for x in groups_and_anns]
        parser_label_ids = [label2id[x] for x in parser_groups]
        parser_coords = [x['coords'] for x in groups_and_anns]
        parser_ann_polygons = get_ann_polygons(parser_coords,parser_label_ids,label2id)
        for coord in coords:
            xmin,ymin,xmax,ymax = get_ori_coords(coord,patch_level,patch_size)
            label_id = get_patch_label(parser_ann_polygons,parser_label_ids,(xmin,ymin,xmax,ymax),consistency)
            one_ans['patch_labels'].append({'coord':coord,'label_id':label_id})
        patch_labes_json['anns'].append(one_ans)
        
    patch_labes_json = convert_ndarray_to_list(patch_labes_json)
    with open(json_path,'w') as f:
        json.dump(patch_labes_json,f)
