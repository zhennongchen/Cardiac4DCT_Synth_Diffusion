import sys
sys.path.append('/workspace/Documents')
# imports
import os, sys
import numpy as np 
import pandas as pd
import ast
import nibabel as nb
from skimage.measure import block_reduce
from scipy.ndimage import zoom

import Diffusion_motion_field.Build_lists.Build_list as Build_list
import Diffusion_motion_field.functions_collection as ff
import Diffusion_motion_field.Data_processing as Data_processing
from Diffusion_motion_field.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_EDM_warp import *
main_path = '/mnt/camca_NAS/4DCT'

# get patient list
patient_list = pd.read_excel(os.path.join('/mnt/camca_NAS/4DCT/Patient_lists/mgh/patient_list_MVF_diffusion_train_test.xlsx'))
# patient_list = patient_list[patient_list['batch'] == 5]
print('Number of patients in the list:', len(patient_list))
timeframe_info = pd.read_excel('/mnt/camca_NAS/4DCT/Patient_lists/mgh/patient_list_final_selection_timeframes.xlsx')


for i in range(0, len(patient_list)):
    row = patient_list.iloc[i]
    patient_class = row['patient_class']
    patient_id = row['patient_id']

    img_path = os.path.join(main_path,'mgh_data/nii-images' ,patient_class, patient_id,'img-nii-resampled-1.5mm')
    mvf_path  = os.path.join(main_path,'mvf_warp0_onecase', patient_class, patient_id, 'voxel_final')
    save_path = os.path.join(main_path,'reference', patient_class, patient_id)
    ff.make_folder([os.path.dirname(save_path), save_path])


    row = timeframe_info[timeframe_info['patient_id'] == patient_id]
    sampled_time_frame_list = ast.literal_eval(row['sampled_time_frame_list'].iloc[0])
    normalized_time_frame_list = ast.literal_eval(row['normalized_time_frame_list_copy'].iloc[0])
    picked_tf_normalized = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    picked_tf = [sampled_time_frame_list[normalized_time_frame_list.index(picked_tf_normalized[iii])] for iii in range(0,len(picked_tf_normalized))]   
    print('picked_tf', picked_tf)

    for tf in picked_tf:
        
        # get original image
        img_file = os.path.join(img_path, str(tf) + '.nii.gz')
        affine = nb.load(img_file).affine
        img = nb.load(img_file).get_fdata()
        if len(img.shape) == 4:
            img = img[:,:,:,0]
        img = Data_processing.crop_or_pad(img, [160,160,96],value = np.min(img))

        # set template image
        if tf == 0:
            template_image = img.copy()

        # get motion field
        mvf_file = os.path.join(mvf_path, str(tf) + '.nii.gz')
        mvf = nb.load(mvf_file).get_fdata()
        # print('img shape:', img.shape, 'mvf shape:', mvf.shape)

        # get downsampled motion field 
        mvf_down = block_reduce(mvf, (4,4,4,1), np.mean)

        # first, save the original image
        save_folder = os.path.join(save_path,'image_original')
        ff.make_folder([save_folder])
        nb.save(nb.Nifti1Image(img, affine), os.path.join(save_folder, str(tf) + '.nii.gz'))

        # then we do the downsampling then upsampling of img
        img_down = block_reduce(img, (4,4,4), np.mean)
        img_down_up = zoom(img_down, (4,4,4), order=1)
        save_folder = os.path.join(save_path,'image_down_up')
        ff.make_folder([save_folder])
        nb.save(nb.Nifti1Image(img_down_up, affine), os.path.join(save_folder, str(tf) + '.nii.gz'))

        # then we apply original mvf onto the original image
        img_torch = torch.from_numpy(template_image).unsqueeze(0).unsqueeze(0).float().to('cuda')
        mvf_torch = torch.from_numpy(np.transpose(mvf, (3,0,1,2))).unsqueeze(0).float().to('cuda')
        img_warped = warp_segmentation_from_mvf(img_torch, mvf_torch)
        # print('img_warped shape:', img_warped.shape)
        img_warped = img_warped.squeeze().cpu().numpy()
        save_folder = os.path.join(save_path,'image_warped_mvf_original')
        ff.make_folder([save_folder])
        nb.save(nb.Nifti1Image(img_warped, affine), os.path.join(save_folder, str(tf) + '.nii.gz'))

        # then we apply downsampled-upsampled mvf 
        mvf_down_torch = torch.from_numpy(np.transpose(mvf_down, (3,0,1,2))).unsqueeze(0).float().to('cuda')
        mvf_down_up_torch = F.interpolate(mvf_down_torch, size = (160,160,96), mode='trilinear', align_corners=True)
        img_warped_down_up = warp_segmentation_from_mvf(img_torch, mvf_down_up_torch)
        img_warped_down_up = img_warped_down_up.squeeze().cpu().numpy()
        save_folder = os.path.join(save_path,'image_warped_mvf_down_up')
        ff.make_folder([save_folder])
        nb.save(nb.Nifti1Image(img_warped_down_up, affine), os.path.join(save_folder, str(tf) + '.nii.gz'))

        # print('difference between img_warped and img_warped_down_up:', np.mean(np.abs(img_warped - img_warped_down_up)))
        # print('difference between img_warped and original img:', np.mean(np.abs(img_warped - img)))




    
