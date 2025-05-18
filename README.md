
The purpose of this repository is implement the idea from the paper: <Boosting 3D Object Generation through PBR Materials> ([github](https://github.com/snowflakewang/PBR_Boost_3DGen)) 

This is just the learning process representation, not commercial use.

## Fine Tune Stable Diffusion
**Goal:** Fine-tune the Stable Diffusion model to predict Albedo maps from beauty renders by modifying the UNet component.  (Refer to the paper 4.1 Albedo and Normal Estimation)


Since the original authors did not release training code and their implementation is built on top of the Marigold project, I re-implemented the training and inference pipeline as follows:
- `src/marigold_trainer.py`: Training pipeline based on the Marigold project (which originally outputs depth).
- `albedo_pipeline.py`: Adapted from [paper author's code](https://github.com/snowflakewang/PBR_Boost_3DGen/blob/main/albedo_mesh_gen/MonoAlbedo/albedo_pipeline.py), modified to fit my dataset and training setup.
- `src/dataset.py`: Dataloader for my preprocessed Megascan dataset.
- `albedo_infer.py`: Inference script that put everything together.
- `train.yaml`

Due to computational limitations (training on a local RTX 5090), I only used a 27 image-set dataset and limited the training to 200 iterations.

**Note:** This project was developed before the release of Marigold Multimodal version. At the time of development, predicting Albedo was not available in the Marigold repository. Please refer to this commit [Marigold](https://github.com/prs-eth/Marigold/tree/62413d56099d36573b2de1eb8c429839734b7782)




## Dataset preparation
The dataset I am using is from Megascan (legacy version, I purchased many). The script load assets in the blender, render from here and produce Beauty, Albedo and Normal layers. 

I developed custom scripts to automate dataset generation--automate the process of importing Megascans (legacy) assets into Blender and batch rendering multi-view outputs (Beauty, Albedo, Normal).


