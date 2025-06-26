import sys 
sys.path.append('/workspace/Documents')
import os
import torch
import numpy as np
import nibabel as nb
import Diffusion_motion_field.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion_3D as ddpm_3D
import Diffusion_motion_field.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_EDM as edm
import Diffusion_motion_field.functions_collection as ff
import Diffusion_motion_field.Build_lists.Build_list as Build_list
import Diffusion_motion_field.Generator as Generator
main_path = '/mnt/camca_NAS/4DCT'

#######################
trial_name = 'MVF_EDM_latent_ES_EFcon'

pre_trained_model = os.path.join('/mnt/camca_NAS/4DCT','models', trial_name, 'models/previous_model-1000.pt')
start_step = 1000
latent = True
slice_range = [0,96] if not latent else [0,24]
image_size_3D = [80,80,slice_range[1]-slice_range[0]] if not latent else [40,40,24]
cutoff = [-20,20] if not latent else [-30,30]
mvf_folder = '/mnt/camca_NAS/4DCT/mvf_warp0_onecase' if not latent else '/mnt/camca_NAS/4DCT/models/VAE_embed3/pred_mvf'
augment_range = [[-15,15], [-10,10]]

downsample_list = (True,False,True,False)  # default is (True, True, True, False)

conditional_diffusion_timeframe = False
conditional_diffusion_image = False
conditional_diffusion_EF = True


#######################
# # define train
data_sheet = os.path.join(main_path,'Patient_lists/patient_list_MVF_diffusion_train_test_filtered.xlsx')
b = Build_list.Build(data_sheet)
patient_class_train_list, patient_id_train_list,_ = b.__build__(batch_list = [0,1,2,3,4])
patient_class_train_list = patient_class_train_list[:-20]
patient_id_train_list = patient_id_train_list[:-20]
patient_class_val_list, patient_id_val_list,_ = b.__build__(batch_list = [5])
patient_class_val_list = patient_class_val_list[-20:]
patient_id_val_list = patient_id_val_list[-20:]

print(patient_id_train_list.shape, patient_id_val_list.shape)


model = ddpm_3D.Unet3D_tfcondition(
    init_dim = 64,
    channels = 3,
    out_dim = 3 ,
    conditional_timeframe_input_dim = None,
    conditional_diffusion_timeframe = conditional_diffusion_timeframe,
    conditional_diffusion_image = conditional_diffusion_image,
    conditional_diffusion_EF = conditional_diffusion_EF,
    dim_mults = (1, 2, 4, 8),
    downsample_list = downsample_list,
    upsample_list = (downsample_list[2], downsample_list[1], downsample_list[0], False),
    flash_attn = False, 
    full_attn = (None, None, False, True), # (None, None, None,False),
)
 
diffusion_model = edm.EDM(
    model,
    image_size = image_size_3D,
    num_sample_steps = 50,
    clip_or_not = False,
)

generator_train = Generator.Dataset_dual_3D_singletf(
    latent_input = latent,
    patient_class_list = patient_class_train_list,
    patient_id_list = patient_id_train_list,
    mvf_folder = mvf_folder,
    image_size_3D = image_size_3D,
    slice_range = slice_range,
    picked_tf = 'ES', #'random' or specific tf or 'ES'
    condition_on_image = conditional_diffusion_image,
    normalize_factor = 'equation',
    maximum_cutoff = cutoff[1],
    minimum_cutoff = cutoff[0],
    shuffle = True,
    augment = True, augment_frequency = 0.8, augment_range = augment_range)

generator_val = Generator.Dataset_dual_3D_singletf(
    latent_input = latent,
    patient_class_list = patient_class_val_list,
    patient_id_list = patient_id_val_list,
    mvf_folder = mvf_folder,
    image_size_3D = image_size_3D,
    slice_range = slice_range,
    picked_tf = 'ES', #'random' or specific tf or 'ES'
    condition_on_image = conditional_diffusion_image,
    normalize_factor = 'equation',
    maximum_cutoff = cutoff[1],
    minimum_cutoff = cutoff[0],
    shuffle = False,augment = False)

trainer = edm.Trainer(
    diffusion_model= diffusion_model,
    generator_train = generator_train,
    generator_val = generator_val,
    train_batch_size = 10,

    train_num_steps = 5000, # total training epochs
    results_folder = os.path.join('/mnt/camca_NAS/4DCT','models', trial_name, 'models'),
   
    train_lr = 1e-4,
    train_lr_decay_every = 500, 
    save_models_every = 5,
    validation_every = 5,
)

trainer.train(pre_trained_model=pre_trained_model, start_step= start_step)