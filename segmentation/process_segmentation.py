import sys
sys.path.append('/workspace/Documents')

import os
import nibabel as nb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shutil
from scipy.ndimage import center_of_mass

import Diffusion_motion_field.functions_collection as ff
import Diffusion_motion_field.Data_processing as Data_processing

main_path = '/mnt/camca_NAS/4DCT'

##### resample to 1.5mm
# patient_list = ff.find_all_target_files(['Normal/*', 'Abnormal/*'], os.path.join(main_path,'nii-images'))
# results = []

# for i in range(0, patient_list.shape[0]):
#     patient_id = os.path.basename(patient_list[i])
#     patient_class = os.path.basename(os.path.dirname(patient_list[i]))
#     print(patient_class, patient_id)

#     image_folder = os.path.join(main_path,'nii-images',patient_class,patient_id,'img-nii-resampled-1.5mm')
#     seg_folder = os.path.join(main_path,'predicted_seg',patient_class,patient_id,'seg-pred-0.625-4classes-connected-retouch')

#     seg_files = ff.sort_timeframe(ff.find_all_target_files(['pred_*.nii.gz'],seg_folder),2,'_')

#     if len(seg_files) < 3:
#         continue

#     else:
#         ff.make_folder([os.path.join(main_path,'predicted_seg',patient_class,patient_id,'seg-pred-0.625-4classes-connected-retouch-resampled-1.5mm')])
#         for seg_file in seg_files:
#             seg = nb.load(seg_file)
            
#             # check whether done
#             if os.path.isfile(os.path.join(main_path,'predicted_seg',patient_class,patient_id,'seg-pred-0.625-4classes-connected-retouch-resampled-1.5mm',os.path.basename(seg_file))):
#                 print('done')
#                 continue
            
#             lr_dim = [1.5, 1.5, 1.5]
#             hr_resample = Data_processing.resample_nifti(seg, order=0,  mode = 'nearest',  cval = 0, in_plane_resolution_mm=lr_dim[0], slice_thickness_mm=lr_dim[-1])
            
#             nb.save(hr_resample,os.path.join(main_path,'predicted_seg',patient_class,patient_id,'seg-pred-0.625-4classes-connected-retouch-resampled-1.5mm',os.path.basename(seg_file)))

#####  some cases need to shift so that when crop to [160,160,96] we can then include the entire LV
patient_list = ff.find_all_target_files(['Normal/*/img-nii-resampled-1.5mm-before-shift', 'Abnormal/*/img-nii-resampled-1.5mm-before-shift'], os.path.join(main_path,'nii-images'))

for i in range(0,patient_list.shape[0]):
    patient_id =  os.path.basename(os.path.dirname(patient_list[i]))
    patient_class = os.path.basename(os.path.dirname(os.path.dirname(patient_list[i])))
    print(patient_class, patient_id)

    if os.path.isdir(os.path.join(main_path,'predicted_seg',patient_class,patient_id,'seg-pred-0.625-4classes-connected-retouch-resampled-1.5mm-before-shift')) == 1:
        print('done')
        continue

    image_folder = os.path.join(main_path,'nii-images',patient_class,patient_id,'img-nii-resampled-1.5mm-before-shift')
    seg_folder = os.path.join(main_path,'predicted_seg',patient_class,patient_id,'seg-pred-0.625-4classes-connected-retouch-resampled-1.5mm')

    seg_ref = nb.load(os.path.join(seg_folder,'pred_s_0.nii.gz')).get_fdata()
    affine = nb.load(os.path.join(seg_folder,'pred_s_0.nii.gz')).affine
    if len(seg_ref.shape) == 4:
        seg_ref = seg_ref[...,0]
    seg_ref = np.round(seg_ref); seg_ref[seg_ref!=1]= 0

    # compute the center of mass of the LV 
    com = center_of_mass(seg_ref)
    x_com, y_com, z_com = int(com[0]), int(com[1]), int(com[2])

    # shift 
    shift_x = x_com - seg_ref.shape[0]//2   
    shift_y = y_com - seg_ref.shape[1]//2 
    shift_z = z_com - seg_ref.shape[2]//2
    print(shift_x, shift_y, shift_z)
   
    shutil.copytree(seg_folder, os.path.join(main_path,'predicted_seg',patient_class,patient_id,'seg-pred-0.625-4classes-connected-retouch-resampled-1.5mm-before-shift'))

    seg_files = ff.find_all_target_files(['pred_*.nii.gz'],os.path.join(main_path,'predicted_seg',patient_class,patient_id,'seg-pred-0.625-4classes-connected-retouch-resampled-1.5mm-before-shift'))
    for ii in range(0,seg_files.shape[0]):
        seg_data = nb.load(seg_files[ii]).get_fdata()
        if len(seg_data.shape) == 4:
            seg_data = seg_data[...,0]
        seg_data = np.round(seg_data)
        seg_data_shifted = Data_processing.translate_image(seg_data, [shift_x, shift_y, shift_z])
        nb.save(nb.Nifti1Image(seg_data_shifted, affine),os.path.join(seg_folder,os.path.basename(seg_files[ii])))
        
           
        