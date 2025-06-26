import sys 
sys.path.append('/workspace/Documents')
import os
import torch
import numpy as np
import Diffusion_motion_field.latent_diffusion.VAE_model as VAE_model
import Diffusion_motion_field.functions_collection as ff
import Diffusion_motion_field.Build_lists.Build_list as Build_list
import Diffusion_motion_field.latent_diffusion.VAE_Generator as VAE_Generator
main_path = '/mnt/camca_NAS/4DCT'

#######################
trial_name = 'VAE_embed3'
emb_channels = 3 if '3' in trial_name else 4

pre_trained_model = os.path.join('/mnt/camca_NAS/4DCT','models', trial_name, 'models_uc/model-54.pt')
start_step = 54
slice_range = [0,96]
image_size_3D = [160,160,slice_range[1]-slice_range[0]]
print(image_size_3D)

#######################
# # define train
# data_sheet = os.path.join(main_path,'Patient_lists/uc/patient_list_MVF_diffusion_train_test_filtered_at_least_10tf.xlsx')

data_sheet = os.path.join(main_path,'Patient_lists/mgh/patient_list_MVF_diffusion_train_test.xlsx')
b = Build_list.Build(data_sheet)
patient_class_train_list, patient_id_train_list,_ = b.__build__(batch_list = [0,1,2,3,4,5])
patient_class_train_list = patient_class_train_list[:-20]
patient_id_train_list = patient_id_train_list[:-20]
patient_class_val_list, patient_id_val_list,_ = b.__build__(batch_list = [5])
# patient_class_val_list = patient_class_val_list[0:1]
# patient_id_val_list = patient_id_val_list[0:1on ]

print(patient_id_train_list.shape, patient_id_val_list.shape)

model = VAE_model.VAE(
        in_channels= 3 ,
        out_channels= 3,
        spatial_dims = 3,
        emb_channels = emb_channels,
        hid_chs =   [64,128,256] if emb_channels == 3 else [64,128,256,512],
        kernel_sizes=[3,3,3] if emb_channels == 3 else [3,3,3,3],
        strides =    [1,2,2] if emb_channels == 3 else [1,2,2,2],)
 
generator_train = VAE_Generator.Dataset_dual_3D(
    patient_class_list = patient_class_train_list,
    patient_id_list = patient_id_train_list,
    mvf_folder = '/mnt/camca_NAS/4DCT/mvf_warp0_onecase',
    image_size_3D = image_size_3D,
    slice_range = slice_range,
    normalize_factor = 'equation',
    maximum_cutoff = 20,
    minimum_cutoff = -20,
    shuffle = True,
    augment = True, augment_frequency = 0.5,pre_done_aug=False)

generator_val = VAE_Generator.Dataset_dual_3D(
    patient_class_list = patient_class_val_list,
    patient_id_list = patient_id_val_list,
    mvf_folder = '/mnt/camca_NAS/4DCT/mvf_warp0_onecase',
    image_size_3D = image_size_3D,
    slice_range = slice_range,
    normalize_factor = 'equation',
    maximum_cutoff = 20,
    minimum_cutoff = -20,
    shuffle = False,
    augment = False,)

trainer = VAE_model.Trainer(
    vae_model= model,
    generator_train = generator_train,
    generator_val = generator_val,
    train_batch_size = 1,

    train_num_steps = 100, # total training epochs
    results_folder = os.path.join('/mnt/camca_NAS/4DCT','models', trial_name, 'models'),
   
    train_lr = 1e-4,
    train_lr_decay_every = 100, 
    save_models_every = 1,
    validation_every = 1000,
)

trainer.train(pre_trained_model=pre_trained_model, start_step= start_step)