import sys
sys.path.append('/workspace/Documents')
# imports
import os, sys

# third party imports
import torch
import numpy as np 
import pandas as pd
import random
import nibabel as nb
import ast
from skimage.measure import block_reduce
from scipy.ndimage import zoom
import Diffusion_motion_field.Build_lists.Build_list as Build_list
import Diffusion_motion_field.functions_collection as ff
import Diffusion_motion_field.Data_processing as Data_processing
from Diffusion_motion_field.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_EDM_warp import *

main_path = '/mnt/camca_NAS/4DCT'
timeframe_info = pd.read_excel('/mnt/camca_NAS/4DCT/Patient_lists/mgh/patient_list_final_selection_timeframes.xlsx')

trial_name = 'MVF_EDM_down_10tf_imgcon_EFcon_warp_orires'
epoch = 1930

how_many_time_frames = 10 if '10tf' in trial_name else 5
latent = True if 'latent' in trial_name else False
save_path = os.path.join(main_path, 'models', trial_name, 'pred_mvf')

###########
# data_sheet = os.path.join(main_path,'Patient_lists/uc/patient_list_MVF_diffusion_train_test_filtered_at_least_10tf.xlsx')
data_sheet = os.path.join(main_path,'Patient_lists/mgh/patient_list_MVF_diffusion_train_test.xlsx')
b = Build_list.Build(data_sheet)
patient_class_list, patient_id_list,_ = b.__build__(batch_list = [5])
# patient_class_list = patient_class_list[-20:]
# patient_id_list = patient_id_list[-20:]

results = []
for i in range(0,patient_class_list.shape[0]):
    
    patient_class = patient_class_list[i]
    patient_id = patient_id_list[i]
    print(i, patient_class, patient_id)

    img_path = os.path.join(main_path,'mgh_data/nii-images' ,patient_class, patient_id,'img-nii-resampled-1.5mm')
    save_folder = os.path.join(save_path, patient_class, patient_id)
    ff.make_folder([os.path.dirname(save_folder), save_folder])

    tf_files = ff.sort_timeframe(ff.find_all_target_files(['*.nii.gz'],img_path),2)
    template_image = nb.load(tf_files[0]).get_fdata()
    if len(template_image.shape) == 4:
        template_image = template_image[:,:,:,0]
    template_image = Data_processing.crop_or_pad(template_image, [160,160,96], value = np.min(template_image))
    affine = nb.load(tf_files[0]).affine

    row = timeframe_info[timeframe_info['patient_id'] == patient_id]
    sampled_time_frame_list = ast.literal_eval(row['sampled_time_frame_list'].iloc[0])
    normalized_time_frame_list = ast.literal_eval(row['normalized_time_frame_list_copy'].iloc[0])

    picked_tf_normalized = [0.1,0.3,0.5,0.7,0.9] if how_many_time_frames == 5 else [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    picked_tf = [sampled_time_frame_list[normalized_time_frame_list.index(picked_tf_normalized[iii])] for iii in range(0,len(picked_tf_normalized))]   
    print('picked tf:' ,picked_tf)

    save_folder_sub_list = ff.find_all_target_files(['epoch' + str(epoch) + '*'],save_folder)
    for ss in range(0, save_folder_sub_list.shape[0]):
        save_folder_sub = save_folder_sub_list[ss]
        print('current folder:', save_folder_sub)
        nb.save(nb.Nifti1Image(template_image, affine), os.path.join(save_folder_sub, 'template_img.nii.gz'))

        if os.path.isfile(os.path.join(save_folder_sub, 'warped_img_pred_tf'+str(picked_tf[-1])+'.nii.gz')) == 1 :#and os.path.isfile(os.path.join(save_folder_sub, 'warped_seg_pred_tf'+str(picked_tf[-1])+'.nii.gz')) == 1:
            print('already done')
        
        else:
            for tf in picked_tf:
                warped_img_gt = nb.load(tf_files[tf]).get_fdata()
                if len(warped_img_gt.shape) == 4:
                    warped_img_gt = warped_img_gt[:,:,:,0]
                warped_img_gt = Data_processing.crop_or_pad(warped_img_gt, [160,160,96], value = np.min(warped_img_gt))

                # load mvf
                if latent != True:
                    mvf_x = nb.load(os.path.join(save_folder_sub, 'pred_tf'+str(tf)+'_x.nii.gz')).get_fdata()
                    mvf_y = nb.load(os.path.join(save_folder_sub, 'pred_tf'+str(tf)+'_y.nii.gz')).get_fdata()
                    mvf_z = nb.load(os.path.join(save_folder_sub, 'pred_tf'+str(tf)+'_z.nii.gz')).get_fdata()
                else:
                    mvf_x = nb.load(os.path.join(save_folder_sub, 'pred_decode_direct_tf'+str(tf)+'_x.nii.gz')).get_fdata()
                    mvf_y = nb.load(os.path.join(save_folder_sub, 'pred_decode_direct_tf'+str(tf)+'_y.nii.gz')).get_fdata()
                    mvf_z = nb.load(os.path.join(save_folder_sub, 'pred_decode_direct_tf'+str(tf)+'_z.nii.gz')).get_fdata()
                
                mvf = np.stack([mvf_x, mvf_y, mvf_z], axis = -1)
                if tf == 0:
                    mvf = np.zeros_like(mvf)
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                template_image_torch = torch.from_numpy(template_image).unsqueeze(0).unsqueeze(0).float().to(device)
                mvf_torch = torch.from_numpy(np.transpose(mvf, (3,0,1,2))).unsqueeze(0).float().cuda()
                # apply deformation field to template image
                warped_img_torch = warp_segmentation_from_mvf(template_image_torch, mvf_torch)
                warped_img = warped_img_torch.cpu().numpy().squeeze()

                # apply deformation field to image
                # warped_img = Data_processing.apply_deformation_field_numpy(np.copy(template_image), mvf, order = 1)
                # warped_img = np.copy(template_image) if tf == 0 else warped_img

                # # save
                nb.save(nb.Nifti1Image(warped_img, affine), os.path.join(save_folder_sub, 'warped_img_pred_tf'+str(tf)+'.nii.gz'))

                # apply deformation field to segmentation
                # if os.path.isfile(os.path.join('/mnt/camca_NAS/4DCT/predicted_seg', patient_class, patient_id,'seg-pred-0.625-4classes-connected-retouch-resampled-1.5mm/pred_s_0.nii.gz')):
                #     seg_path = os.path.join('/mnt/camca_NAS/4DCT/predicted_seg', patient_class, patient_id,'seg-pred-0.625-4classes-connected-retouch-resampled-1.5mm/pred_s_0.nii.gz')
                # else:
                #     seg_path = os.path.join('/mnt/camca_NAS/4DCT/mgh_data/predicted_seg', patient_class, patient_id,'pred_s_0.nii.gz')
                
                # seg_img = nb.load(seg_path).get_fdata(); seg_img = np.round(seg_img).astype(np.int16)
                # if len(seg_img.shape) == 4:
                #     seg_img = seg_img[:,:,:,0]
                # # make it binary
                # seg_img[seg_img != 1] = 0
                # # crop
                # seg_img = Data_processing.crop_or_pad(seg_img, [160,160,96], value = 0)

                # seg_img_torch = torch.from_numpy(seg_img).unsqueeze(0).unsqueeze(0).float().cuda()
                # warped_seg_torch = warp_segmentation_from_mvf(seg_img_torch, mvf_torch)
                # warped_seg = warped_seg_torch.cpu().numpy().squeeze()

                # nb.save(nb.Nifti1Image(warped_seg, affine), os.path.join(save_folder_sub, 'warped_seg_pred_tf'+str(tf)+'.nii.gz'))


        # # calculate EF now
        # LV_volume_list = []
        # for tf in picked_tf:
        #     warped_seg = nb.load(os.path.join(save_folder_sub, 'warped_seg_pred_tf'+str(tf)+'.nii.gz')).get_fdata()
        #     warped_seg = np.round(warped_seg).astype(np.int16)
        #     LV_volume = np.sum(warped_seg)
        #     LV_volume_list.append(LV_volume)
        # LV_volume_list = np.asarray(LV_volume_list)
        # print('LV volume list:', LV_volume_list)
        # EF_generated = round((LV_volume_list[0] - np.min(LV_volume_list)) / LV_volume_list[0], 5)
        # with open(os.path.join(save_folder_sub,'EF.txt'), "r") as f:
        #     content = f.read()
        # EF_set = float(content) 
        # print('EF set:', EF_set, 'EF generated:', EF_generated)
        # EF_original = row['EF_sampled_in_10tf_by_mvf'].iloc[0] if how_many_time_frames == 10 else row['EF_sampled_in_5tf_by_mvf'].iloc[0]

        # results.append([patient_class, patient_id, ss, EF_original , EF_set, EF_generated])
        # df = pd.DataFrame(results, columns = ['patient_class', 'patient_id', 'sample', 'EF_original', 'EF_set', 'EF_AI'])
        # df.to_excel(os.path.join(main_path, 'models', trial_name, 'EF_from_seg_results_epoch' + str(epoch) + '.xlsx'), index = False)

