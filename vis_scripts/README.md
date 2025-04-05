## :ledger: **Draw Feature Map**
Plot the feature map distribution of features after MIL aggregation
```shell
python draw_feature_map.py --yaml_path Config.yaml --ckpt_path your_pretrained_weigths --id2class '{0:"Normal",1:"Tumor"...}' --save_path your_png_save_path --test_dataset_csv your_dataset_csv --fig_size '(10,8)'
```
- `--dataset_csv` : ensure it contains `test_slide_path` and `test_label`, the other headers are not affected
    ```bash
    test_slide_path,test_label
    /path/to/slide_1.pt,0
    /path/to/slide_2.pt,0
    /path/to/slide_3.pt,2
    /path/to/slide_4.pt,1
    /path/to/slide_5.pt,2
    /path/to/slide_6.pt,1
    ```
- `--fig_size` : tuple of draw figure size


## :ledger: **Draw Attention Map**
Plot the attention(importance) map distribution of features within MIL aggregation process. Please note that not all MIL methods have the concept of Attention scores. For MIL methods without Attention scores, the visualization can be considered as contribution or importance scores. Since many MIL methods do not provide a way to visualize attention (or importance), the Representation for importance visualization implemented in this repository may differ somewhat from the original work. For some self-Attention based MIL Framework (with cls-token), we use the similarity between cls-token and patch-tokens to represent the importance of each patch. You can also use the last attention block attn score to represent. 
```shell
python draw_attention_map.py --heatmap_config_yaml your_heatmap_config_yaml
```
- `--heatmap_config_yaml` : We porvide a basic version of the config_yaml in `/vis_scripts/heatmap_config.yaml`. The key parameters in the configuration file are described as follows:
    - `data_arguments.wsi_csv` :  A CSV with the column name `wsi_path` containing the absolute paths of WSI files.
        ```bash
        wsi_path
        /path/to/slide_1.svs
        /path/to/slide_2.svs
        /path/to/slide_3.sdpc
        /path/to/slide_4.sdpc
        /path/to/slide_5.tiff
        /path/to/slide_6.ndpi
        ```
    - `encoder_arguments.model_name` : Patch encoder name
    - `encoder_arguments.model_weights_dir` : Patch encoder weights dir 
    - `model_arguments.yaml_path` : MIL model yaml config
    - `model_arguments.ckpt_path` : MIL model pretrained weights
