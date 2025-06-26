import sys 
sys.path.append('/workspace/Documents')
import os
import torch
import numpy as np
import Diffusion_motion_field.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion_3D as ddpm_3D
import Diffusion_motion_field.functions_collection as ff
import Diffusion_motion_field.Build_lists.Build_list as Build_list
import Diffusion_motion_field.Generator as Generator
main_path = '/mnt/camca_NAS/4DCT'

#######################
trial_name = 'MVF_DDPM_down_tf1_onecase_noaug'
how_many_timeframes_together = 1

pre_trained_model = os.path.join('/mnt/camca_NAS/4DCT','models', trial_name, 'models/model-2600.pt')
start_step = 2600
slice_range = [0,96]
image_size_3D = [80,80,slice_range[1]-slice_range[0]]
print(image_size_3D)

objective = 'pred_noise'
timesteps = 1000

condition_on_image =False

#######################
# # define train
data_sheet = os.path.join(main_path,'Patient_lists/patient_list_MVF_diffusion_train_test.xlsx')
b = Build_list.Build(data_sheet)
patient_class_train_list, patient_id_train_list,_ = b.__build__(batch_list = [0,1,2,3,4 ])
patient_class_train_list = patient_class_train_list[0:1]
patient_id_train_list = patient_id_train_list[0:1]
patient_class_val_list, patient_id_val_list,_ = b.__build__(batch_list = [0])
patient_class_val_list = patient_class_val_list[0:1]
patient_id_val_list = patient_id_val_list[0:1]

print(patient_id_train_list.shape, patient_id_val_list.shape)


model = ddpm_3D.Unet3D_tfcondition(
    init_dim = 64,
    channels = 3 * how_many_timeframes_together, 
    out_dim = 3 * how_many_timeframes_together,
    conditional_timeframe_input_dim = how_many_timeframes_together,
    conditional_diffusion_timeframe = True,
    conditional_diffusion_image = condition_on_image,
    dim_mults = (1, 2, 4, 8),
    flash_attn = False, 
    full_attn = (None, None, None,False),
)


diffusion_model = ddpm_3D.GaussianDiffusion3D(
    model,
    image_size_3D = image_size_3D,
    timesteps = timesteps,           # number of steps
    sampling_timesteps = 250,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    auto_normalize=False,
    objective = objective,
    clip_or_not = False,
)

generator_train = Generator.Dataset_dual_3D(
    patient_class_list = patient_class_train_list,
    patient_id_list = patient_id_train_list,
    mvf_folder = '/mnt/camca_NAS/4DCT/mvf_warp0_onecase',
    how_many_timeframes_together = how_many_timeframes_together,
    image_size_3D = image_size_3D,
    slice_range = slice_range,
    normalize_factor = 'equation',
    maximum_cutoff = 20,
    minimum_cutoff = -20,
    condition_on_image = condition_on_image,
    shuffle = True,
    augment = False, augment_frequency = 0.35,)

generator_val = Generator.Dataset_dual_3D(
    patient_class_list = patient_class_val_list,
    patient_id_list = patient_id_val_list,
    mvf_folder = '/mnt/camca_NAS/4DCT/mvf_warp0_onecase',
    how_many_timeframes_together = how_many_timeframes_together,
    image_size_3D = image_size_3D,
    slice_range = slice_range,
    normalize_factor = 'equation',
    maximum_cutoff = 20,
    minimum_cutoff = -20,
    condition_on_image = condition_on_image,
    shuffle = False,
    augment = False,)

trainer = ddpm_3D.Trainer(
    diffusion_model= diffusion_model,
    generator_train = generator_train,
    generator_val = generator_val,
    train_batch_size = 1,

    train_num_steps = 500000, # total training epochs
    results_folder = os.path.join('/mnt/camca_NAS/4DCT','models', trial_name, 'models'),
   
    train_lr = 1e-4,
    train_lr_decay_every = 1000, 
    save_models_every = 50,
    validation_every = 50,)


trainer.train(pre_trained_model=pre_trained_model, start_step= start_step )
