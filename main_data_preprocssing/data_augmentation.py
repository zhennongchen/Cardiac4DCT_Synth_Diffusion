import sys 
sys.path.append('/workspace/Documents')
import os
import torch
import numpy as np
import pandas as pd
import nibabel as nb
import random
import ast
import shutil
import pandas as pd
from scipy import ndimage
from skimage.measure import block_reduce

import Diffusion_motion_field.Build_lists.Build_list as Build_list
import Diffusion_motion_field.Generator as Generator
import Diffusion_motion_field.Data_processing as Data_processing
import Diffusion_motion_field.functions_collection as ff
import Diffusion_motion_field.latent_diffusion.VAE_model as VAE_model

main_path = '/mnt/camca_NAS/4DCT'
timeframe_info = pd.read_excel(os.path.join(main_path,'Patient_lists/uc/patient_list_final_selection_timeframes.xlsx'))

# define the patient list
data_sheet = os.path.join(main_path,'Patient_lists/uc/patient_list_MVF_diffusion_train_test_filtered_at_least_10tf.xlsx')
# data_sheet = os.path.join(main_path,'Patient_lists/mgh/patient_list_MVF_diffusion_train_test.xlsx')
b = Build_list.Build(data_sheet)
patient_class_list, patient_id_list,_ = b.__build__(batch_list = [0,1,2,3,4,5])
print('patient_class_list.shape:', patient_class_list.shape)

# define the VAE model
model_path = os.path.join(main_path,'models/VAE_embed3/models_uc/model-54.pt') if 'uc' in data_sheet else os.path.join(main_path,'models/VAE_embed3/models_mgh/model-100.pt')
vae_model = Generator.VAE_process(model_path = model_path)

# do augmentation, 5 augmentation for each case
for i in range(0, patient_id_list.shape[0] // 2):
    patient_class = patient_class_list[i]
    patient_id = patient_id_list[i]
  
    print('i:', i, 'patient_class:', patient_class, 'patient_id:', patient_id)

    # save folder 
    save_folder = os.path.join('/mnt/camca_NAS/4DCT/mvf_aug/', patient_class, patient_id); ff.make_folder([os.path.dirname(save_folder),save_folder])

    # load the original MVF
    # path = os.path.join('/workspace/Documents/Data/mvf' ,patient_class, patient_id,'voxel_final')
    path = os.path.join('/mnt/camca_NAS/4DCT/mvf_warp0_onecase',patient_class,patient_id,'voxel_final')
    files = ff.find_all_target_files(['*.nii.gz'],path)
    final_files = np.copy(files)
    for f in files:
        if 'moved' in f or 'original' in f:
            # remove it from the numpy array
            final_files = np.delete(final_files, np.where(final_files == f))
    files = ff.sort_timeframe(final_files,2)

    # get time frames
    row = timeframe_info[timeframe_info['patient_id'] == patient_id]
    sampled_time_frame_list = ast.literal_eval(row['sampled_time_frame_list'].iloc[0])
    print('time_frame_list:', sampled_time_frame_list)

    for aug_index in range(0,6):
        print('aug_index:', aug_index)
        # set augmentation parameters
        if os.path.isfile(os.path.join(save_folder, 'aug_'+str(aug_index), 'aug_parameter.npy')):
            print('load aug_parameter from file')
            aug_parameter = np.load(os.path.join(save_folder, 'aug_'+str(aug_index), 'aug_parameter.npy'))
            z_rotate_degree = aug_parameter[0]
            x_translate = int(aug_parameter[1])
            y_translate = int(aug_parameter[2])
        else:
            z_rotate_degree = random.uniform(-15,15) if aug_index != 0 else 0
            x_translate = int(round(random.uniform(-15,15))) if aug_index != 0 else 0
            y_translate = int(round(random.uniform(-15,15))) if aug_index != 0 else 0
            aug_parameter = [z_rotate_degree, x_translate, y_translate] if aug_index != 0 else [0,0,0]
            save_folder_aug = os.path.join(save_folder, 'aug_'+str(aug_index)); ff.make_folder([save_folder_aug])
            np.save(os.path.join(save_folder_aug, 'aug_parameter.npy'), np.asarray(aug_parameter))
        print('z_rotate_degree:', z_rotate_degree, 'x_translate:', x_translate, 'y_translate:', y_translate)
        
        ######### load each MVF and do augmentation as well as latent encoding
        # for j in range(0,len(sampled_time_frame_list)):
            
        #     j = sampled_time_frame_list[j]
        #     print('current time frame: ', j , ' file:', files[j])
            
        #     ##### aug_index = 0, just copy the original MVF
        #     if aug_index == 0:
        #         ff.make_folder([os.path.join(save_folder, 'aug_'+str(aug_index), 'mvf'), os.path.join(save_folder, 'aug_'+str(aug_index), 'latent'), os.path.join(save_folder, 'aug_'+str(aug_index), 'mvf_downsampled')])
        #         # shutil.copy(files[j], os.path.join(save_folder, 'aug_'+str(aug_index), 'mvf', os.path.basename(files[j])))
        #         # shutil.copy(os.path.join('/mnt/camca_NAS/4DCT/models/VAE_embed3/pred_mvf',patient_class, patient_id,'epoch100', 'pred_latent_tf'+ str(j) + '.nii.gz'), os.path.join(save_folder, 'aug_'+str(aug_index), 'latent', str(j)+'.nii.gz'))
        #         mvf = nb.load(files[j]).get_fdata()
        #         downsample_mvf = block_reduce(np.copy(mvf), (4,4,4,1), func=np.mean)
        #         affine = nb.load(files[j]).affine
        #         nb.save(nb.Nifti1Image(downsample_mvf, affine), os.path.join(save_folder, 'aug_'+str(aug_index), 'mvf_downsampled', str(j)+'.nii.gz'))
        #         continue
       
        #     ###### MVF
        #     if os.path.isfile(os.path.join(save_folder, 'aug_'+str(aug_index), 'mvf', str(j)+'.nii.gz')):
        #         print('Aug mvf file exists:', os.path.join(save_folder, 'aug_'+str(aug_index), 'mvf', str(j)+'.nii.gz'))
        #         mvf_aug = nb.load(os.path.join(save_folder, 'aug_'+str(aug_index), 'mvf', str(j)+'.nii.gz'))
        #         affine = mvf_aug.affine
        #         mvf_aug = mvf_aug.get_fdata()
        #     else:
        #         mvf = nb.load(files[j]).get_fdata()
        #         affine = nb.load(files[j]).affine
        #         mvf_aug = Data_processing.random_move(mvf,x_translate,y_translate,z_rotate_degree, fill_val=0, do_augment=True)
        #         # print('mvf aug shape:', mvf_aug.shape)

        #         # save the augmented MVF
        #         save_folder_aug = os.path.join(save_folder, 'aug_'+str(aug_index), 'mvf'); ff.make_folder([os.path.dirname(save_folder_aug),save_folder_aug])
        #         save_path = os.path.join(save_folder_aug, str(j)+'.nii.gz')
        #         img = nb.Nifti1Image(mvf_aug, affine)
        #         nb.save(img, save_path)

        #     ###### downsample MVF:
        #     if os.path.isfile(os.path.join(save_folder, 'aug_'+str(aug_index), 'mvf_downsampled', str(j)+'.nii.gz'))==1:
        #         print('Aug downsampled mvf file exists:', os.path.join(save_folder, 'aug_'+str(aug_index), 'mvf_downsampled', str(j)+'.nii.gz'))
        #     else:
        #         downsample_mvf_aug =  block_reduce(np.copy(mvf_aug), (4,4,4,1), func=np.mean)
        #         save_folder_aug_downsample = os.path.join(save_folder, 'aug_'+str(aug_index), 'mvf_downsampled'); ff.make_folder([os.path.dirname(save_folder_aug_downsample),save_folder_aug_downsample])
        #         save_path = os.path.join(save_folder_aug_downsample, str(j)+'.nii.gz')
        #         nb.save( nb.Nifti1Image(downsample_mvf_aug, affine), save_path)
            

            ######## latent encoding
            # if os.path.isfile(os.path.join(save_folder, 'aug_'+str(aug_index), 'latent', str(j)+'.nii.gz')):
            #     print('Aug latent file exists:', os.path.join(save_folder, 'aug_'+str(aug_index), 'latent', str(j)+'.nii.gz'))
            # else:
            #     mvf_aug_for_latent = np.copy(mvf_aug)
            #     mvf_aug_for_latent = Data_processing.cutoff_intensity(mvf_aug_for_latent, cutoff_low = -20, cutoff_high = 20)
            #     mvf_aug_for_latent = Data_processing.normalize_image(mvf_aug_for_latent, normalize_factor = 'equation', image_max = 20, image_min = -20, invert = False)
            #     latent = vae_model.VAE_encode(mvf_aug_for_latent)
            #     # print('latent shape:', latent.shape)
                
            #     # # save the latent
            #     save_folder_aug = os.path.join(save_folder, 'aug_'+str(aug_index), 'latent'); ff.make_folder([os.path.dirname(save_folder_aug),save_folder_aug])
            #     save_path = os.path.join(save_folder_aug, str(j)+'.nii.gz')
            #     latent = np.transpose(latent, (1,2,3,0))
            #     img = nb.Nifti1Image(latent, affine)
            #     nb.save(img, save_path)

        # do for condition image as well
        # if os.path.isfile(os.path.join(save_folder, 'aug_'+str(aug_index), 'condition_img', '0.nii.gz')):
        #     print('condition image file exists:', os.path.join(save_folder, 'aug_'+str(aug_index), 'condition_img', '0.nii.gz'))
        # else:
        #     # img_path = os.path.join('/workspace/Documents/Data/mvf',patient_class, patient_id, 'img-nii-resampled-1.5mm/0.nii.gz')
        #     img_path = os.path.join('/mnt/camca_NAS/4DCT/nii-images', patient_class, patient_id, 'img-nii-resampled-1.5mm/0.nii.gz')
        
        #     con_img = nb.load(img_path).get_fdata()
        #     affine = nb.load(img_path).affine
        #     if len(con_img.shape) == 4:
        #         con_img = con_img[:,:,:,0]
        #     con_img = Data_processing.crop_or_pad(con_img, [160,160,96], value = np.min(con_img))

        #     # move for latent purpose (move then downsample)
        #     con_img1 = np.copy(con_img)
        #     con_img1 = Data_processing.random_move(con_img1,x_translate,y_translate,z_rotate_degree,do_augment = True, fill_val = np.min(con_img))
        #     con_img1 = block_reduce(con_img1, (160//40,160//40, 96//24), func=np.mean)

        #     save_folder_aug = os.path.join(save_folder, 'aug_'+str(aug_index), 'condition_img'); ff.make_folder([os.path.dirname(save_folder_aug),save_folder_aug])
        #     nb.save(nb.Nifti1Image(con_img1, affine), os.path.join(save_folder_aug, '0.nii.gz'))

        # do for segmentation as well
        if os.path.isfile(os.path.join(save_folder, 'aug_'+str(aug_index), 'segmentation', '0.nii.gz')) and os.path.isfile(os.path.join(save_folder, 'aug_'+str(aug_index), 'segmentation_original_res', '0.nii.gz')):
            print('segmentation file exists:', os.path.join(save_folder, 'aug_'+str(aug_index), 'segmentation', '0.nii.gz'))
        else:
            seg_path = os.path.join('/mnt/camca_NAS/4DCT/predicted_seg', patient_class, patient_id,'seg-pred-0.625-4classes-connected-retouch-resampled-1.5mm/pred_s_0.nii.gz')
            # seg_path = os.path.join('/mnt/camca_NAS/4DCT/mgh_data/predicted_seg', patient_class, patient_id,'pred_s_0.nii.gz')
            seg_img = nb.load(seg_path).get_fdata(); seg_img = np.round(seg_img).astype(np.int16)
            affine = nb.load(seg_path).affine
            if len(seg_img.shape) == 4:
                seg_img = seg_img[:,:,:,0]
            # make it binary
            seg_img[seg_img != 1] = 0
            # crop
            seg_img = Data_processing.crop_or_pad(seg_img, [160,160,96], value = 0)
            # augmentation
            seg_img1 = np.copy(seg_img)
            seg_img1 = Data_processing.random_move(seg_img1,x_translate,y_translate,z_rotate_degree,do_augment = True, fill_val = 0, order = 0)
            save_folder_aug = os.path.join(save_folder, 'aug_'+str(aug_index), 'segmentation_original_res'); ff.make_folder([os.path.dirname(save_folder_aug),save_folder_aug])
            nb.save(nb.Nifti1Image(seg_img1, affine), os.path.join(save_folder, 'aug_'+str(aug_index), 'segmentation_original_res', '0.nii.gz'))
            seg_img1 = block_reduce(seg_img1, (160//40,160//40, 96//24), func=np.max)
            save_folder_aug = os.path.join(save_folder, 'aug_'+str(aug_index), 'segmentation'); ff.make_folder([os.path.dirname(save_folder_aug),save_folder_aug])
            nb.save(nb.Nifti1Image(seg_img1, affine), os.path.join(save_folder_aug, '0.nii.gz'))
    
        
    



    