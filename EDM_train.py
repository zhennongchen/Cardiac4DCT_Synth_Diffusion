import sys 
sys.path.append('/workspace/Documents')
import os
import torch
import numpy as np
import pandas as pd
import nibabel as nb
from ema_pytorch import EMA
import Diffusion_motion_field.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion_3D as ddpm_3D
import Diffusion_motion_field.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_EDM as edm
import Diffusion_motion_field.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_EDM_EF_predictor as edm_EF_predictor
import Diffusion_motion_field.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_EDM_warp as edm_warp
import Diffusion_motion_field.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_EDM_adversarial as edm_adversarial
import Diffusion_motion_field.Build_lists.Build_list as Build_list
import Diffusion_motion_field.EF_predictor.model as ef_model
import Diffusion_motion_field.Discriminator.model as discriminator_model
import Diffusion_motion_field.Generator as Generator
main_path = '/mnt/camca_NAS/4DCT'

#######################
trial_name = 'MVF_EDM_down_10tf_imgcon_EFcon_warp_adver'
latent = True if 'latent' in trial_name else False 
EF_loss_weight = 0
average_loss_weight = 0.1 if 'adver' in trial_name else 0.0 # 0.1 for adversarial training, 0.0 for normal training

how_many_timeframes_together = 10
picked_tf = 'ES' if how_many_timeframes_together == 1 else [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # 'random' or specific tf or 'ES'

pre_trained_model = os.path.join('/mnt/camca_NAS/4DCT','models','MVF_EDM_down_10tf_imgcon_EFcon_warmup', 'models/model-1000.pt')
start_step = 1000

mvf_size_3D = [160,160,96] if latent else [40,40,24]#[80,80,96] # downsampled
mvf_slice_range = [0,96]
latent_size_3D = [40,40,24]; latent_slice_range = [0,24]
mvf_folder = '/workspace/Documents/Data/mvf' #'/mnt/camca_NAS/4DCT/mvf_warp0_onecase'
VAE_model_path = os.path.join(main_path, 'models/VAE_embed3/models/model-54.pt')

downsample_list = (True,False,True,False)  if latent else (True, True, False, False) # default is (True, True, True, False) 

augment_pre_done = True
conditional_diffusion_timeframe = False
conditional_diffusion_image = True
conditional_diffusion_EF = True if 'EFcon' in trial_name else False
conditional_diffusion_seg = False # if 'warp' in trial_name else False

#######################
# # define training and validation data
data_sheet1 = os.path.join(main_path,'Patient_lists/uc/patient_list_MVF_diffusion_train_test_filtered_at_least_10tf.xlsx')
data_sheet2 = os.path.join(main_path,'Patient_lists/mgh/patient_list_MVF_diffusion_train_test.xlsx')

b1 = Build_list.Build(data_sheet1)
patient_class_train_list1, patient_id_train_list1,_ = b1.__build__(batch_list = [0,1,2,3,4])
patient_class_train_list1 = patient_class_train_list1[:-20]; patient_id_train_list1 = patient_id_train_list1[:-20]
b2 = Build_list.Build(data_sheet2)
patient_class_train_list2, patient_id_train_list2,_ = b2.__build__(batch_list = [0,1,2,3,4])
patient_class_train_list2 = patient_class_train_list2[:-20]; patient_id_train_list2 = patient_id_train_list2[:-20]
patient_class_train_list = np.concatenate((patient_class_train_list1, patient_class_train_list2), axis = 0)
patient_id_train_list = np.concatenate((patient_id_train_list1, patient_id_train_list2), axis = 0)

patient_class_val_list1, patient_id_val_list1,_ = b1.__build__(batch_list = [0,1,2,3,4])
patient_class_val_list1 = patient_class_val_list1[-20:];patient_id_val_list1 = patient_id_val_list1[-20:]
patient_class_val_list2, patient_id_val_list2,_ = b2.__build__(batch_list = [0,1,2,3,4])
patient_class_val_list2 = patient_class_val_list2[-20:]; patient_id_val_list2 = patient_id_val_list2[-20:]
patient_class_val_list = np.concatenate((patient_class_val_list1, patient_class_val_list2), axis = 0)
patient_id_val_list = np.concatenate((patient_id_val_list1, patient_id_val_list2), axis = 0)
print('patient_class_train_list:', len(patient_class_train_list), ' patient_class_val_list:', len(patient_class_val_list))

# data_sheet1 = os.path.join(main_path,'Patient_lists/uc/patient_list_MVF_diffusion_train_test_filtered_at_least_10tf.xlsx')

# b1 = Build_list.Build(data_sheet1)
# patient_class_train_list, patient_id_train_list,_ = b1.__build__(batch_list = [0])
# patient_class_train_list = patient_class_train_list[0:1]; patient_id_train_list = patient_id_train_list[0:1]
# patient_class_val_list = patient_class_train_list; patient_id_val_list = patient_id_train_list

# define diffusion model
model = ddpm_3D.Unet3D_tfcondition(
    init_dim = 64,
    channels = 3 * how_many_timeframes_together,
    out_dim = 3 * how_many_timeframes_together,
    # conditional_timeframe_input_dim = None,
    # conditional_diffusion_timeframe = conditional_diffusion_timeframe,
    conditional_diffusion_image = conditional_diffusion_image,
    conditional_diffusion_EF = conditional_diffusion_EF,
    conditional_diffusion_seg = conditional_diffusion_seg,
    dim_mults = (1, 2, 4, 8),
    downsample_list = downsample_list,
    upsample_list = (downsample_list[2], downsample_list[1], downsample_list[0], False),
    flash_attn = False, 
    full_attn = (None, None, False, False), # (None, None, None,False),
)
 
diffusion_model = edm.EDM(
    model,
    image_size = latent_size_3D,
    num_sample_steps = 50,
    clip_or_not = False,) 

# define VAE model
vae_model = Generator.VAE_process(model_path = VAE_model_path) if latent else False


# define generator
generator_train = Generator.Dataset_dual_3D(
    VAE_process = vae_model, # False or VAE_process object

    patient_class_list = patient_class_train_list,
    patient_id_list = patient_id_train_list,
    mvf_folder = mvf_folder,
    how_many_timeframes_together = how_many_timeframes_together,

    mvf_size_3D = mvf_size_3D,
    latent_size_3D = latent_size_3D,
    slice_range = mvf_slice_range,
    
    picked_tf = picked_tf, #'random' or specific tf or 'ES'
    condition_on_image = True,
    condition_on_seg = True,
    mvf_cutoff = [-20,20],
    latent_cutoff = [-30,30],
    shuffle = True,
    augment = True,
    augment_frequency = 0.8, 
    augment_pre_done = augment_pre_done,augment_aug_index = [1,5])

generator_val = Generator.Dataset_dual_3D(
    VAE_process = vae_model,

    patient_class_list = patient_class_val_list,
    patient_id_list = patient_id_val_list,
    mvf_folder = mvf_folder,

    how_many_timeframes_together = how_many_timeframes_together,

    mvf_size_3D = mvf_size_3D,
    latent_size_3D = latent_size_3D,
    slice_range = mvf_slice_range,
    
    picked_tf = picked_tf, 
    condition_on_image = True,
    condition_on_seg = True,
    mvf_cutoff = [-20,20],
    latent_cutoff = [-30,30],
    augment_pre_done = augment_pre_done,)

# define trainer
if 'EFpredict' in trial_name:
    # define EF predictor
    model = ef_model.CNN_EF_predictor_LSTM(init_dim = 16,channels = 1,dim_mults = (2,4,8),full_attn = (None,None, None),act = 'LeakyReLU',)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ema = EMA(model);ema.to(device)  
    data = torch.load('/mnt/camca_NAS/4DCT/models/EF_predictor_LSTM_noisy_trial2/models/model-196.pt', map_location=device)
    model.load_state_dict(data['model']); ema.load_state_dict(data["ema"])

    trainer = edm_EF_predictor.Trainer(diffusion_model= diffusion_model, EF_predictor = ema.ema_model, EF_loss_weight = EF_loss_weight,
        generator_train = generator_train, generator_val = generator_val, 
        train_batch_size = 2 if latent == False else 5,
        results_folder = os.path.join(main_path,'models', trial_name, 'models'),)
    
elif ('warp' in trial_name) and ('adver' not in trial_name):
    trainer = edm_warp.Trainer(diffusion_model= diffusion_model, generator_train = generator_train, generator_val = generator_val,  EF_loss_weight = EF_loss_weight,
                               train_batch_size = 2 if latent == False else 5,
                        results_folder = os.path.join(main_path,'models', trial_name, 'models'),)
elif 'adver' in trial_name:
    # define discriminator
    discriminator = discriminator_model.CNN_temporalConv(init_dim = 16,channels = 1,dim_mults = (2,4,8),full_attn = (None,None, None),act = 'LeakyReLU',)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data = torch.load('/mnt/camca_NAS/4DCT/models/MVF_EDM_down_10tf_imgcon_EFcon_warp_adver/models/discriminator-580.pt', map_location=device)
    # discriminator.load_state_dict(data['D_model'])

    trainer = edm_adversarial.Trainer(diffusion_model= diffusion_model, discriminator_model = discriminator, generator_train = generator_train, generator_val = generator_val,  EF_loss_weight = EF_loss_weight, adversarial_loss_weight = average_loss_weight,
                               train_batch_size = 2 if latent == False else 5,results_folder = os.path.join(main_path,'models', trial_name, 'models'),)

else:
    trainer = edm.Trainer(diffusion_model= diffusion_model, generator_train = generator_train, generator_val = generator_val, train_batch_size = 2 if latent == False else 5,
        results_folder = os.path.join(main_path,'models', trial_name, 'models'),)

trainer.train_num_steps = 1500
trainer.train_lr = 1e-4
trainer.train_lr_decay_every = 500
trainer.save_models_every = 10
trainer.validation_every = 10


trainer.train(pre_trained_model=pre_trained_model, start_step= start_step)