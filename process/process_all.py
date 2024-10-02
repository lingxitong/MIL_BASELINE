from .MEAN_MIL.process_mean_mil import process_MEAN_MIL
from .MAX_MIL.process_max_mil import process_MAX_MIL
from .AB_MIL.process_ab_mil import process_AB_MIL
from .GATE_AB_MIL.process_gate_ab_mil import process_GATE_AB_MIL
from .WIKG_MIL.process_wikg_mil import process_WIKG_MIL
from .TRANS_MIL.process_trans_mil import process_TRANS_MIL
from .RRT_MIL.process_rrt_mil import process_RRT_MIL
from .CLAM_SB_MIL.process_clam_sb_mil import process_CLAM_SB_MIL
from .CLAM_MB_MIL.process_clam_mb_mil import process_CLAM_MB_MIL
from .DS_MIL.process_ds_mil import process_DS_MIL
from .DTFD_MIL.process_dtfd_mil import process_DTFD_MIL
from .FR_MIL.process_fr_mil import process_FR_MIL
from utils.general_utils import *
def process(args,yaml_path,now_fold=None):
    save_dataset_csv(args)
    save_yaml(args,yaml_path)
    if args.General.MODEL_NAME == 'MEAN_MIL':
        process_MEAN_MIL(args)
    elif args.General.MODEL_NAME == 'MAX_MIL':
        process_MAX_MIL(args)
    elif args.General.MODEL_NAME == 'AB_MIL':
        process_AB_MIL(args)
    elif args.General.MODEL_NAME == 'GATE_AB_MIL':
        process_GATE_AB_MIL(args)
    elif args.General.MODEL_NAME == 'WIKG_MIL':
        process_WIKG_MIL(args)
    elif args.General.MODEL_NAME == 'TRANS_MIL':
        process_TRANS_MIL(args)
    elif args.General.MODEL_NAME == 'RRT_MIL':
        process_RRT_MIL(args)
    elif args.General.MODEL_NAME == 'CLAM_SB_MIL':
        process_CLAM_SB_MIL(args)
    elif args.General.MODEL_NAME == 'CLAM_MB_MIL':
        process_CLAM_MB_MIL(args)
    elif args.General.MODEL_NAME == 'DS_MIL':
        process_DS_MIL(args)
    elif args.General.MODEL_NAME == 'DTFD_MIL':
        process_DTFD_MIL(args)
    elif args.General.MODEL_NAME == 'FR_MIL':
        process_FR_MIL(args)
    else:
        raise ValueError('Model Not Found')
    
    
    
    
    

    