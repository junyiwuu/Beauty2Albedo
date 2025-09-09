**This project aims to demonstrate the learning process by re-implementing the idea from the paper: <[Boosting 3D Object Generation through PBR Materials](https://snowflakewang.github.io/PBR_Boost_3DGen/)> ([github](https://github.com/snowflakewang/PBR_Boost_3DGen))**


# Beauty2Albedo
## Goal:
Fine-tune the Stable Diffusion model to predict Albedo maps from Beauty renders by modifying the UNet component.  (Refer to the paper [4.1 Albedo and Normal Estimation](https://arxiv.org/pdf/2411.16080))



## Project overview:
**Dataset preparation**:
The original dataset is from Megascan library. I created the blender script to batch read and render the Megascan assets and generate the dataset pair (Beauty pass and Albedo pass). 

**Training and Inference**:
I adapted the Marigold training code to train on my generated dataset, modifying the original training logic from predicting Depth (from Beauty) to predicting Albedo (from Beauty pass). Then I made slightly modifications to the pipeline and inference code from the paper's implementation to integrate my trained weights.


- The training code is adpated from [Marigold](https://github.com/prs-eth/Marigold/tree/62413d56099d36573b2de1eb8c429839734b7782) 
- The pipeline and inference code is adapted from [the paper's code](https://github.com/snowflakewang/PBR_Boost_3DGen/tree/aaebb46b74c4f0d6d9edc8a2a7cc5a9144a43806/albedo_mesh_gen/MonoAlbedo)



**Training Details**:
Run on Linux machine (Rocky) with one NVIDIA RTX 5090. Cuda version 12.9.
33 Megascan assets were used, render with 6 random angles per asset, generated 198 pairs of dataset. The training run for 8000 iterations, converge observed around 5000 iterations. Training and inference resolution is 256x256.

**Note:** This project was developed before the release of Marigold Multimodal version. At the time of development, predicting Albedo was not available in the Marigold repository. Please refer to this [Marigold commit](https://github.com/prs-eth/Marigold/tree/62413d56099d36573b2de1eb8c429839734b7782).

## How to use:
Download the repository: `git clone `
### Try the weight:
(since I only trained on the small datasets and the project is for demonstrating learning process purpose, I recommend you use the provided beauty images to see the weights is working)

1. Download weights and put them in `Beauty2Albedo/MonoSD/safetensors` or your customized path
2. `cd Beauty2Albedo/MonoSD/Marigold/marigold`
3. Example: `python albedo_infer.py --src_path ../../test_images/rgb_images/rlCay.png  --dst_path ../../test_images/infer_images/rlCay_inferAlbedo.png --input_weights ../../safetensors`
 
Alternatively if you want to try other image:  `python albedo_infer.py --src_path <your_beauty_image> --dst_path <output_path> --input_weights ../../safetensors` (or you can replace the input_weights to where you download the weights)


### Train on your own dataset
#### Dataset preparation
We assume you donwload from Megascan library, so the asset folder has the same structure as from Megascan downloaded asset.

1. `python ./Megascan_Processing/batch_process.py --asset_folder <your_megascan_assets_folder> --hdri_path <you_HDRI_path> --output_dir <output_folder> --num_angles 6 --res 256`

Description: Batch render Megascan assets in the Blender and render out Beauty pass and Albedo pass with 6 random angles and resolution 256. "filename_lst" and "filename_lst_val" files are generated at the same time.
- filename_lst: datasets that will be used for training
- filename_lst_val: datasets that will be used for evaluation during the training. Now it random select 10 pairs from training dataset. You can modify and input your own evaluation datasets.


```
Dataset_Folder
-- Albedo
    -- xxx_angle1.png
    -- xxx_angle2.png
        ...
-- Beauty
    -- xxx_angle1.png
    -- xxx_angle2.png
        ...
--filename_lst
--filename_lst_val
```



#### Training

1. `python ./MonoSD/Marigold/training.py --training_data ./Megascan_Processing/output`

**Check the tensorboard**:
`tensorboard --logdir ./output/train/tensorboard`


#### Inference
1. `python ./MonoSD/Marigold/marigold/albedo_infer.py --src_path <your_beauty_image> --dst_path <output_path>` (It automatically read the latest saved weight from your training)





## Output:
**Loss**:
![loss](./images/train_loss.png)

![beauty2albedo](./images/Beauty2Albedo.jpg)



