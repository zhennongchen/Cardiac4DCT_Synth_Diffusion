# Synthesize cardiac 4DCT using diffusion model
**Author: Zhennong Chen, PhD**<br />

This is the GitHub repo based on an paper under rewivew: <br />
*Diffusion-based Synthesis of Cardiac 4D Computed Tomography via Learned Cardiac Motion Vector Fields*<br />
Authors: Zhennong Chen, Dufan Wu, Quanzheng Li<br />

**Citation**: TBD

## Description
We have proposed the first method to synthesize a cardiac 4DCT sequence given a patient's 3DCT data and a specified ejection fraction (EF).<br />
The main contributions  are as follows:<br />
(1) instead of directly synthesizing 4DCT data, we propose to learn and synthesize cardiac motion vector fields (MVF) using a diffusion model. MVFs can be compactly synthesized in a downsampled form and later applied to the 3DCT template via warping to reconstruct 4DCT sequences.<br />
(2) we introduce a novel method for counterfactual 4DCT synthesis conditioned on varying values of left ventricular ejection fraction (LVEF). Please refer "mapping function" in our paper.<br />


## User Guideline
### Environment Setup
The entire code is [containerized](https://www.docker.com/resources/what-container). This makes setting up environment swift and easy. Make sure you have nvidia-docker and Docker CE [installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) on your machine before going further. <br />
- You can build your own docker from the folder ```docker```. The docker image can be built by ```./docker_build.sh```, after that the docker container can be built by ```./docker_run.sh```. The installed packages can be referred to ```dockerfile``` and ```requirements.txt``` <br />
- You'll need  ```docker/docker_tensorflow``` for step 1 and ```docker/docker_torch``` for the rest steps<br />

### Data Preparation (we have examples available)
You should prepare two things before running this step. Please refer to the `example_data` folder for guidance:

- **NIfTI images** of 4DCT resampled to a voxel size of **[1.5, 1.5, 1.5] mm³**.  
   - Each cardiac phase should be saved as a separate file.  
   - All files should be placed in a folder named:  
     `img-nii-resampled-1.5mm`.

- **A patient list** that enumerates all your cases.  
   - To understand the expected format, please refer to the file:  
     `example_data/Patient_lists/example_data/patient_list.xlsx`.
   - Make sure the number of time frames is equal to the number of nii files

- Please refer ```example_data``` folder for examples.


### Experiments
we have design our study into 5 steps, with each step having its own jupyter notebook.<br /> 
**step1: get ground truth MVF**: use ```step1_get_MVF.ipynb```, it uses Voxelmorph generated ground truth MVF for each 4DCT case saved in a folder called ```mvf_warp0_onecase``` <br /> 

**step2: data preparation**: use ```step2_data_preprocessing.ipynb```, it does the following tasks:
1. Prepare the left ventricular (LV) segmentation masks using pre-trained network (in the folder ```segmentation_network```). it should save the segmentation in a folder called ```predicted_seg``` <br /> 
2. Sample the original time frames into **10 evenly spaced cardiac phases** and Get ground truth Ejection fraction (LVEF) for each case. it should save these info in a spreadsheet in ```Patient_lists/example_data/patient_list_final_selection_timeframes``` <br /> 
3. Prepare augmented data for training (since data is large, on-the-fly augmentation will be time-consuming). it should save the augmented data in a folder called ```mvf_aug``` <br /> 

**step3: train diffusion model**: use ```step3_train_model.ipynb``` <br /> 

**step4: synthesize MVF**: use ```step4_MVF_synthesis.ipynb``` <br /> 

**step5: synthesize 4DCT**: use ```step5_4DCT_synthesis.ipynb```, it applies the synthesized MVF to the 3DCT template for 4DCT generation <br /> 

### Additional guidelines 
Please contact chenzhennong@gmail.com for any further questions.



