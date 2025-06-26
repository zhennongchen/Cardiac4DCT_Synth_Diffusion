import sys 
sys.path.append('/workspace/Documents')
import os
import torch
import numpy as np
import Diffusion_motion_field.segmentation.model as seg_model
import Diffusion_motion_field.functions_collection as ff
import Diffusion_motion_field.Build_lists.Build_list as Build_list
import Diffusion_motion_field.segmentation.Generator as Generator

main_path = '/mnt/camca_NAS/4DCT'

#######################
trial_name = 'seg_3D'
num_classes = 4 # 4 classes: background, LV, LA, LVOT 

pre_trained_model = None#os.path.join('/mnt/camca_NAS/4DCT','models', trial_name, 'models/model-147.pt')
start_step = 0

img_size_3D = [160,160,96]
#######################
# # define train
data_sheet = os.path.join(main_path,'Patient_lists/patient_list_MVF_diffusion_train_test_filtered.xlsx')
b = Build_list.Build(data_sheet)
patient_class_train_list, patient_id_train_list,_ = b.__build__(batch_list = [0,1,2,3,4])
# patient_class_train_list = patient_class_train_list[0:1]
# patient_id_train_list = patient_id_train_list[0:1]
patient_class_val_list, patient_id_val_list,_ = b.__build__(batch_list = [5])
# patient_class_val_list = patient_class_val_list[0:1]
# patient_id_val_list = patient_id_val_list[0:1]


# build model
model = seg_model.Unet3D(
    init_dim = 16,
    channels = 1,
    num_classes = num_classes,
    dim_mults = (2,4,8,16),
    full_attn = (None,None, None, None),
    act = 'LeakyReLU',
)

# build generator
generator_train = Generator.Dataset_3D(
    patient_class_train_list,
    patient_id_train_list,
    img_size_3D = img_size_3D,
    picked_tf = 'random', #'random' or specific tf or 'ES'
    relabel_LVOT = True,
    shuffle = True,
    augment = True,
    augment_frequency = 0.7,)

generator_val = Generator.Dataset_3D(
    patient_class_val_list,
    patient_id_val_list,
    img_size_3D = img_size_3D,
    picked_tf = 'random', #'random' or specific tf or 'ES'
    relabel_LVOT = True,)

# train
trainer = seg_model.Trainer(
    model,
    generator_train,
    generator_val,
    train_batch_size = 1,

    accum_iter = 5, # gradient accumulation steps
    train_num_steps = 10000, # total training epochs
    results_folder =  os.path.join('/mnt/camca_NAS/4DCT','models', trial_name, 'models'),
       
    train_lr_decay_every = 100, 
    save_models_every = 1,
    validation_every = 1,
)

trainer.train(pre_trained_model=pre_trained_model, start_step= start_step)