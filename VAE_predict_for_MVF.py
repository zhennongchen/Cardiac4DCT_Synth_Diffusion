# code for VAE decoding the MVF latent space
import sys 
sys.path.append('/workspace/Documents')
import os
import torch
import ast
import numpy as np
import nibabel as nb
import pandas as pd
import Diffusion_motion_field.latent_diffusion.VAE_model as VAE_model
import Diffusion_motion_field.functions_collection as ff
import Diffusion_motion_field.Build_lists.Build_list as Build_list
import Diffusion_motion_field.latent_diffusion.VAE_Generator as VAE_Generator
main_path = '/mnt/camca_NAS/4DCT'
from ema_pytorch import EMA

timeframe_info = pd.read_excel('/mnt/camca_NAS/4DCT/Patient_lists/mgh/patient_list_final_selection_timeframes.xlsx')

###########
trial_name = 'VAE_embed3'
epoch = 100
trained_model_filename = os.path.join(main_path, 'models', trial_name, 'models_mgh/model-' + str(epoch)+ '.pt')
# save_folder = os.path.join(main_path, 'models', trial_name, 'pred_mvf_')
save_folder = os.path.join(main_path,'models/MVF_EDM_latent_10tf_imgcon_EFcon_mgh/pred_mvf')
os.makedirs(save_folder, exist_ok=True)

slice_range = [0,96]
image_size_3D = [160,160,slice_range[1]-slice_range[0]]

picked_tf = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]#[0.1,0.3,0.5,0.7,0.9]

###########
# data_sheet = os.path.join(main_path,'Patient_lists/uc/patient_list_MVF_diffusion_train_test_filtered_at_least_10tf.xlsx')
data_sheet = os.path.join(main_path,'Patient_lists/mgh/patient_list_MVF_diffusion_train_test.xlsx')
b = Build_list.Build(data_sheet)
patient_class_list, patient_id_list,_ = b.__build__(batch_list = [5])
# patient_class_list = patient_class_list[-20:]
# patient_id_list = patient_id_list[-20:]

model = VAE_model.VAE(
        in_channels= 3, 
        out_channels= 3, 
        spatial_dims = 3,
        emb_channels = 3,
        hid_chs =   [64,128,256], 
        kernel_sizes=[3,3,3], 
        strides =    [1,2,2],)

for i in range(0,10):#len(patient_id_list)):
    
    patient_class = patient_class_list[i]
    patient_id = patient_id_list[i]
    print(i, patient_class, patient_id)


    print(os.path.join(save_folder,patient_class, patient_id))
    save_folder_case_list = ff.find_all_target_files(['epoch2500*'],os.path.join(save_folder,patient_class, patient_id))
    print(save_folder_case_list)

    img_file = os.path.join('/mnt/camca_NAS/4DCT/mgh_data','nii-images',patient_class, patient_id, 'img-nii-resampled-1.5mm/0.nii.gz')
    affine = nb.load(img_file).affine

    row = timeframe_info[timeframe_info['patient_id'] == patient_id]
    es_index = row['es_index'].iloc[0]
    tf_choices = [es_index]
    sampled_time_frame_list = ast.literal_eval(row['sampled_time_frame_list'].iloc[0])
    normalized_time_frame_list = ast.literal_eval(row['normalized_time_frame_list_copy'].iloc[0])

    tf_choices = [sampled_time_frame_list[normalized_time_frame_list.index(picked_tf[iii])] for iii in range(0,len(picked_tf))]
    print(picked_tf)

    for save_folder_case in save_folder_case_list:
        print(save_folder_case)
        print('tf_choices: ', tf_choices)
        if os.path.isfile(os.path.join(save_folder_case,'pred_decode_direct_tf'+str(tf_choices[-1])+'_x.nii.gz'))==1:
            print('done, continue')
            continue

        for tf_choice in tf_choices:
            if 1==1:#os.path.isfile(os.path.join(save_folder_case, 'pred-epoch-' + str(epoch) + '.nii.gz')) == 0:
                generator = VAE_Generator.Dataset_dual_3D(
                    patient_class_list = np.asarray([patient_class]),
                    patient_id_list = np.asarray([patient_id]),
                    picked_tf = tf_choice,
                    mvf_folder = '/mnt/camca_NAS/4DCT/mvf_warp0_onecase',#'/workspace/Documents/Data/mvf',
                    latent_space_folder = False,

                    image_size_3D = image_size_3D,
                    slice_range = slice_range,
                    normalize_factor = 'equation',
                    maximum_cutoff = 20,
                    minimum_cutoff = -20,
                    shuffle = False,)

                # # sample:
                # sampler = VAE_model.Predict(
                #     model,
                #     generator,
                #     batch_size = 1)

                save_file = os.path.join(save_folder_case, 'pred-epoch-' + str(epoch) + '.nii.gz')
                # # make the latent space
                # all_mae,x_mae, y_mae,z_mae = sampler.sample_3D_w_trained_model(trained_model_filename=trained_model_filename,affine = affine, save_file = save_file, 
                #                                                                save_latent = True, save_decode = False, only_do_decode = False)
                
                # use latent space to do the decoding
                generator.latent_space_folder = save_folder_case
                sampler_2 = VAE_model.Predict(
                    model,
                    generator,
                    batch_size = 1)

                all_mae,x_mae, y_mae,z_mae = sampler_2.sample_3D_w_trained_model(trained_model_filename=trained_model_filename,affine = affine, save_file = save_file, 
                                                                            save_latent = False, save_decode = False, only_do_decode = True)
                
                # print('tf_choice: ', tf_choice, 'mae: ', all_mae, x_mae, y_mae, z_mae)
    #   