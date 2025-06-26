import sys 
sys.path.append('/workspace/Documents')
import os
import torch
import numpy as np
import Diffusion_motion_field.EF_predictor.model as ef_model
import Diffusion_motion_field.functions_collection as ff
import Diffusion_motion_field.Build_lists.Build_list as Build_list
import Diffusion_motion_field.EF_predictor.Generator as Generator


main_path = '/mnt/camca_NAS/4DCT'

#######################
trial_name = 'EF_predictor_temporalConv_noisy'

pre_trained_model = None#os.path.join('/mnt/camca_NAS/4DCT','models', trial_name, 'models_previous/model-167.pt')
start_step = 0
img_size_3D = [40,40,24]

#######################
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

# patient_id_train_list = patient_id_train_list[0:1]
# patient_class_train_list = patient_class_train_list[0:1]
# patient_id_val_list = patient_id_train_list
# patient_class_val_list = patient_class_train_list
print('patient_class_train_list:', len(patient_class_train_list), ' patient_class_val_list:', len(patient_class_val_list))


######## build model
if 'LSTM' in trial_name:
    model = ef_model.CNN_EF_predictor_LSTM(
        init_dim = 16,
        channels = 1,
        dim_mults = (2,4,8),
        full_attn = (None,None, None),
        act = 'LeakyReLU',)
elif 'temporalConv' in trial_name:
    model = ef_model.CNN_EF_predictor_temporalConv(
        init_dim = 16,
        channels = 1,
        dim_mults = (2,4,8),
        full_attn = (None,None, None),
        act = 'LeakyReLU',)
    
# input = torch.randn(1,30,40,40,24)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
# out = model(input.to(device))


# build generator
generator_train = Generator.Dataset_MVF(
    patient_class_train_list,
    patient_id_train_list,
    mvf_folder = '/workspace/Documents/Data/mvf',
    mvf_size_3D = img_size_3D,
    picked_tf = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    mvf_cutoff = [-20,20],
    shuffle = True,
    augment = True,
    augment_frequency = 0.8, 
    noise_add_frequency = 0.4,
    augment_pre_done = True,augment_aug_index = [1,5])

generator_val = Generator.Dataset_MVF(
    patient_class_val_list,
    patient_id_val_list,
    mvf_folder = '/workspace/Documents/Data/mvf',
    mvf_size_3D = img_size_3D,
    picked_tf = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    mvf_cutoff = [-20,20],
    shuffle =False,
    augment = False,
    noise_add_frequency = 0.2,
    augment_pre_done = True)

    
# train
trainer = ef_model.Trainer(
    model,
    generator_train,
    generator_val,
    train_batch_size = 10,

    accum_iter = 2, # gradient accumulation steps
    train_num_steps = 1000, # total training epochs
    results_folder =  os.path.join('/mnt/camca_NAS/4DCT','models', trial_name, 'models'),
       
    train_lr_decay_every = 500, 
    save_models_every = 1,
    validation_every = 1,
)

trainer.train(pre_trained_model=pre_trained_model, start_step= start_step)