from utils.general_utils import save_dataset_csv,print_args
from utils.yaml_utils import save_yaml
def process(args,yaml_path,options):
    save_dataset_csv(args)
    save_yaml(args,yaml_path,options)
    print_args(args)
    if args.General.MODEL_NAME == 'MEAN_MIL':
        from .MEAN_MIL.process_mean_mil import process_MEAN_MIL
        process_MEAN_MIL(args)
    elif args.General.MODEL_NAME == 'MAX_MIL':
        from .MAX_MIL.process_max_mil import process_MAX_MIL
        process_MAX_MIL(args)
    elif args.General.MODEL_NAME == 'AB_MIL':
        from .AB_MIL.process_ab_mil import process_AB_MIL
        process_AB_MIL(args)
    elif args.General.MODEL_NAME == 'GATE_AB_MIL':
        from .GATE_AB_MIL.process_gate_ab_mil import process_GATE_AB_MIL
        process_GATE_AB_MIL(args)
    elif args.General.MODEL_NAME == 'WIKG_MIL':
        from .WIKG_MIL.process_wikg_mil import process_WIKG_MIL
        process_WIKG_MIL(args)
    elif args.General.MODEL_NAME == 'TRANS_MIL':
        from .TRANS_MIL.process_trans_mil import process_TRANS_MIL
        process_TRANS_MIL(args)
    elif args.General.MODEL_NAME == 'CLAM_SB_MIL':
        from .CLAM_SB_MIL.process_clam_sb_mil import process_CLAM_SB_MIL
        process_CLAM_SB_MIL(args)
    elif args.General.MODEL_NAME == 'CLAM_MB_MIL':
        from .CLAM_MB_MIL.process_clam_mb_mil import process_CLAM_MB_MIL
        process_CLAM_MB_MIL(args)
    elif args.General.MODEL_NAME == 'DS_MIL':
        from .DS_MIL.process_ds_mil import process_DS_MIL
        process_DS_MIL(args)
    elif args.General.MODEL_NAME == 'DTFD_MIL':
        from .DTFD_MIL.process_dtfd_mil import process_DTFD_MIL
        process_DTFD_MIL(args)
    elif args.General.MODEL_NAME == 'FR_MIL':
        from .FR_MIL.process_fr_mil import process_FR_MIL
        process_FR_MIL(args)
    elif args.General.MODEL_NAME == 'ILRA_MIL':
        from .ILRA_MIL.process_ilra_mil import process_ILRA_MIL
        process_ILRA_MIL(args)
    elif args.General.MODEL_NAME == 'DGR_MIL':
        from .DGR_MIL.process_dgr_mil import process_DGR_MIL
        process_DGR_MIL(args)
    elif args.General.MODEL_NAME == 'CDP_MIL':
        from .CDP_MIL.process_cdp_mil import process_CDP_MIL
        process_CDP_MIL(args)
    elif args.General.MODEL_NAME == 'LONG_MIL':
        from .LONG_MIL.process_long_mil import process_LONG_MIL
        process_LONG_MIL(args)
    elif args.General.MODEL_NAME == 'AMD_MIL':
        from .AMD_MIL.process_amd_mil import process_AMD_MIL
        process_AMD_MIL(args)
    elif args.General.MODEL_NAME == 'AC_MIL':
        from .AC_MIL.process_ac_mil import process_AC_MIL
        process_AC_MIL(args)
    elif args.General.MODEL_NAME == 'ADD_MIL':
        from .ADD_MIL.process_add_mil import process_ADD_MIL
        process_ADD_MIL(args)
    elif args.General.MODEL_NAME == 'CA_MIL':
        from .CA_MIL.process_ca_mil import process_CA_MIL
        process_CA_MIL(args)
    elif args.General.MODEL_NAME == 'DyHG_MIL':
        from .DyHG_MIL.process_dyhg_mil import process_DyHG_MIL
        process_DyHG_MIL(args)
    elif args.General.MODEL_NAME == 'DG_MIL':
        from .DG_MIL.process_dg_mil import process_DG_MIL
        process_DG_MIL(args)
    elif args.General.MODEL_NAME == 'DT_MIL':
        from .DT_MIL.process_dt_mil import process_DT_MIL
        process_DT_MIL(args)
    elif args.General.MODEL_NAME == 'Mamba_MIL':
        from .Mamba_MIL.process_mamba_mil import process_Mamba_MIL
        process_Mamba_MIL(args)
    elif args.General.MODEL_NAME == 'MHIM_MIL':
        from .MHIM_MIL.process_mhim_mil import process_MHIM_MIL
        process_MHIM_MIL(args)
    elif args.General.MODEL_NAME == 'Micro_MIL':
        from .Micro_MIL.process_micro_mil import process_Micro_MIL
        process_Micro_MIL(args)
    elif args.General.MODEL_NAME == 'MSM_MIL':
        from .MSM_MIL.process_msm_mil import process_MSM_MIL
        process_MSM_MIL(args)
    elif args.General.MODEL_NAME == 'PA_MIL':
        from .PA_MIL.process_pa_mil import process_PA_MIL
        process_PA_MIL(args)
    else:
        raise ValueError('Model Not Found')
    # save every slide softmax-logits (confidence for each slide towards each class)
    
    
    
    
    

    
