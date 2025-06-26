import sys
sys.path.append('/workspace/Documents')

import os
import nibabel as nb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Diffusion_motion_field.functions_collection as ff
import Diffusion_motion_field.Data_processing as Data_processing
# 
# main_path = '/mnt/camca_NAS/4DCT/nii-images/'

## build a patient list
# patients_list = ff.find_all_target_files(['Normal/*','Abnormal/*'],main_path)
# results = []
# for patient in patients_list:
#     patient_id = os.path.basename(patient)
#     patient_class = os.path.basename(os.path.dirname(patient))
#     img = nb.load(os.path.join(patient,'img-nii/0.nii.gz'))
#     # pixel size
#     pixdim = img.header.get_zooms()
#     # image size
#     img_size = img.header.get_data_shape()
#     files = ff.find_all_target_files(['img-nii/*'],patient)
    
#     results.append([patient_id,patient_class,pixdim,img_size,len(files)])
# df = pd.DataFrame(results,columns=['patient_id','patient_class','pixdim','img_size','timeframes'])
# df.to_excel(os.path.join('/mnt/camca_NAS/4DCT/Patient_lists','patient_list.xlsx'),index=False)


## image resampling
# patient_list = pd.read_excel('/mnt/camca_NAS/4DCT/Patient_lists/patient_list.xlsx').iloc[0:125]
# print(patient_list.shape)

# for i in range(0, patient_list.shape[0]):
#     patient_id = patient_list.iloc[i]['patient_id']
#     patient_class = patient_list.iloc[i]['patient_class']
#     folder = os.path.join(main_path,patient_class,patient_id,'img-nii')
#     files = ff.sort_timeframe(ff.find_all_target_files(['*.nii.gz'],folder),2)
#     print(patient_class, patient_id)
#     for f in files:
#         save_path = os.path.join(main_path,patient_class,patient_id,'img-nii-resampled-1.5mm')
#         if os.path.isfile(os.path.join(save_path, os.path.basename(f))):
#             continue
#         img = nb.load(f)
#         img_data = img.get_fdata()
#         lr_dim = [1.5, 1.5, 1.5]
#         hr_resample = Data_processing.resample_nifti(img, order=3,  mode = 'nearest',  cval = np.min(img_data), in_plane_resolution_mm=lr_dim[0], slice_thickness_mm=lr_dim[-1])
#         ff.make_folder([save_path])
#         nb.save(hr_resample,os.path.join(save_path,os.path.basename(f)))

# check and resample segmentation
# main_path = '/mnt/camca_NAS/4DCT'
# patient_list = ff.find_all_target_files(['Normal/*','Abnormal/*'], os.path.join(main_path,'nii-images'))

# results = []
# for i in range(0, patient_list.shape[0]):
#     patient_id = os.path.basename(patient_list[i])
#     patient_class = os.path.basename(os.path.dirname(patient_list[i]))
#     print(patient_class, patient_id)

#     image_folder = os.path.join(main_path,'nii-images',patient_class,patient_id,'img-nii-resampled-1.5mm')
#     seg_folder = os.path.join(main_path,'predicted_seg',patient_class,patient_id,'seg-pred-0.625-4classes-connected-retouch')
#     if not os.path.isfile(os.path.join(seg_folder,'pred_s_0.nii.gz')):
#         have_seg = False
#         img_shape, seg_shape, difference_in_shape = None, None, None
#     else:
#         have_seg = True
#         img_data = nb.load(os.path.join(image_folder,'0.nii.gz')).get_fdata()
#         if len(img_data.shape) == 4:
#             img_data = img_data[...,0]
#         seg = nb.load(os.path.join(seg_folder,'pred_s_0.nii.gz'))
#         seg_data = nb.load(os.path.join(seg_folder,'pred_s_0.nii.gz')).get_fdata()
#         if len(seg_data.shape) == 4:
#             seg_data = seg_data[...,0]
#         seg_data = np.round(seg_data)

#         ### resample 
#         lr_dim = [1.5, 1.5, 1.5]
#         hr_resample = Data_processing.resample_nifti(seg, order=0,  mode = 'nearest',  cval = 0, in_plane_resolution_mm=lr_dim[0], slice_thickness_mm=lr_dim[-1])
#         ff.make_folder([os.path.join(main_path,'predicted_seg',patient_class,patient_id,'seg-pred-0.625-4classes-connected-retouch-resampled-1.5mm')])
#         nb.save(hr_resample,os.path.join(main_path,'predicted_seg',patient_class,patient_id,'seg-pred-0.625-4classes-connected-retouch-resampled-1.5mm','pred_s_0.nii.gz'))
        
#         ### check shape
#         image_shape = img_data.shape
#         seg_shape = nb.load(os.path.join(main_path,'predicted_seg',patient_class,patient_id,'seg-pred-0.625-4classes-connected-retouch-resampled-1.5mm','pred_s_0.nii.gz')).get_fdata().shape
#         difference_in_shape = np.max(np.abs(np.array(image_shape) - np.array(seg_shape)))
#         difference_in_shape = np.max(np.abs(np.array(image_shape) - np.array(seg_shape)))
#     print(patient_class, patient_id, have_seg, image_shape, seg_shape, difference_in_shape)

#     results.append([patient_id,patient_class,have_seg,image_shape, seg_shape, difference_in_shape])
#     df = pd.DataFrame(results,columns=['patient_id','patient_class','have_seg','img_shape','seg_shape','difference_in_shape'])
    # df.to_excel(os.path.join(main_path,'Patient_lists','patient_list_seg_check.xlsx'),index=False)

### check the shape of image is consistent in all timeframes
# main_path = '/mnt/camca_NAS/4DCT'
# patient_list = pd.read_excel('/mnt/camca_NAS/4DCT/Patient_lists/patient_list_seg_check.xlsx')
# max_diff_list = []
# results = []
# for i in range(0, patient_list.shape[0]):
#     patient_class = patient_list.iloc[i]['patient_class']
#     patient_id = patient_list.iloc[i]['patient_id']
#     print(patient_class, patient_id)

#     have_seg = patient_list.iloc[i]['have_seg']
#     if have_seg == False:
#         image_shape = None
#         max_diff = 0
#     else:
#         image_folder = os.path.join(main_path,'nii-images',patient_class,patient_id,'img-nii-resampled-1.5mm')
#         files = ff.sort_timeframe(ff.find_all_target_files(['*.nii.gz'],image_folder),2)
#         img_template = nb.load(files[0]).get_fdata()
#         if len(img_template.shape) == 4:
#             img_template = img_template[...,0]
#         image_shape = img_template.shape

#         max_list = []
#         for j in range(1,files.shape[0]):
#             img = nb.load(files[j])
#             img_data = img.get_fdata()
#             if len(img_data.shape) == 4:
#                 img_data = img_data[...,0]
#             max_diff = np.max(np.abs(np.array(image_shape) - np.array(img_data.shape)))
#             max_list.append(max_diff)
#         max_diff = np.max(max_list)

#     print(patient_class, patient_id, image_shape, max_diff)
#     max_diff_list.append(max_diff)
#     results.append([patient_class, patient_id, image_shape, max_diff])
#     df = pd.DataFrame(results,columns=['patient_class','patient_id','image_shape','image_shape_consistency'])
#     df.to_excel('/mnt/camca_NAS/4DCT/Patient_lists/patient_list_image_shape_consistency.xlsx',index=False)

# patient_list['image_shape_consistency'] = max_diff_list
# patient_list.to_excel('/mnt/camca_NAS/4DCT/Patient_lists/patient_list_seg_check.xlsx',index=False)


## confirm that after reshape the image to 160x160x96, it will still include all the LV pixels
# main_path = '/mnt/camca_NAS/4DCT'   
# patient_list = pd.read_excel('/mnt/camca_NAS/4DCT/Patient_lists/patient_list_seg_check.xlsx')
# from scipy.ndimage import center_of_mass
# import shutil

# results = []; include_all_LV_list = []; LV_pixel_diff_list = []; need_to_shift_list = []
# for i in range(0, patient_list.shape[0]):
#     patient_class = patient_list.iloc[i]['patient_class']
#     patient_id = patient_list.iloc[i]['patient_id']
#     print(patient_class, patient_id)

#     if (patient_list.iloc[i]['have_seg'] == False) or (patient_list.iloc[i]['image_shape_consistency'] > 0):
#         include_all_LV = None; LV_pixel_diff = None; need_to_shift = None
    
#     else:
#         image_folder = os.path.join(main_path,'nii-images',patient_class,patient_id,'img-nii-resampled-1.5mm')
#         seg_folder = os.path.join(main_path,'predicted_seg',patient_class,patient_id,'seg-pred-0.625-4classes-connected-retouch-resampled-1.5mm')

#         seg_ref = nb.load(os.path.join(seg_folder,'pred_s_0.nii.gz')).get_fdata()
#         if len(seg_ref.shape) == 4:
#             seg_ref = seg_ref[...,0]
#         seg_ref = np.round(seg_ref); seg_ref[seg_ref!=1]= 0

#         # compute the center of mass of the LV 
#         com = center_of_mass(seg_ref)
#         x_com, y_com, z_com = int(com[0]), int(com[1]), int(com[2])

#         # shift 
#         shift_x = x_com - seg_ref.shape[0]//2   
#         shift_y = y_com - seg_ref.shape[1]//2 
#         shift_z = z_com - seg_ref.shape[2]//2

#         seg_ref_pad = Data_processing.crop_or_pad(seg_ref,[160,160,96],value = 0)
#         if np.sum(seg_ref) == np.sum(seg_ref_pad):
#             # print('no change, include all LV pixels')
#             include_all_LV = True
#             LV_pixel_diff = 0
#             need_to_shift = False   
#         else:
#             print('change!!!!!!!, not include all LV pixels')
#             seg_ref_shifted = Data_processing.translate_image(seg_ref, [shift_x, shift_y, shift_z])
#             seg_ref_pad = Data_processing.crop_or_pad(seg_ref_shifted,[160,160,96],value = 0)
            # ValueError('the following two lines need to be re-written according to what we saved/did for segmentation')
#             nb.save(nb.Nifti1Image(seg_ref_shifted, nb.load(os.path.join(seg_folder,'pred_s_0.nii.gz')).affine),os.path.join(seg_folder,'pred_s_0_shifted.nii.gz'))
#             nb.save(nb.Nifti1Image(seg_ref_pad, nb.load(os.path.join(seg_folder,'pred_s_0.nii.gz')).affine),os.path.join(seg_folder,'pred_s_0_shifted_pad.nii.gz'))

#             if np.sum(seg_ref_shifted) == np.sum(seg_ref_pad):
#                 print('after shift, include all LV pixels')
#                 include_all_LV = True
#                 LV_pixel_diff = 0
#                 need_to_shift = True
#             else:
#                 print('after shift, no change!')
#                 include_all_LV = False
#                 LV_pixel_diff = np.sum(seg_ref) - np.sum(seg_ref_pad)
#                 need_to_shift = False

#     include_all_LV_list.append(include_all_LV); LV_pixel_diff_list.append(LV_pixel_diff); need_to_shift_list.append(need_to_shift)

#     if need_to_shift:
#         # copy the folder
#         if os.path.isdir(os.path.join(main_path,'nii-images',patient_class,patient_id,'img-nii-resampled-1.5mm-before-shift')):
#             image_folder = os.path.join(main_path,'nii-images',patient_class,patient_id,'img-nii-resampled-1.5mm-before-shift')
#             continue
#         else:
#             shutil.copytree(image_folder, os.path.join(main_path,'nii-images',patient_class,patient_id,'img-nii-resampled-1.5mm-before-shift'))

#             # do the shift
#             img_files = ff.sort_timeframe(ff.find_all_target_files(['*.nii.gz'],image_folder),2)
#             for tf in range(0,img_files.shape[0]):
#                 img = nb.load(img_files[tf])
#                 img_data = img.get_fdata()
#                 if len(img_data.shape) == 4:
#                     img_data = img_data[...,0]
                
#                 img_data_shifted = Data_processing.translate_image(img_data, [shift_x, shift_y, shift_z])

#                 nb.save(nb.Nifti1Image(img_data_shifted, img.affine),os.path.join(image_folder,os.path.basename(img_files[tf])))
            
# patient_list['include_all_LV'] = include_all_LV_list
# patient_list['LV_pixel_diff'] = LV_pixel_diff_list
# patient_list['need_to_shift'] = need_to_shift_list
# patient_list.to_excel('/mnt/camca_NAS/4DCT/Patient_lists/patient_list_seg_check.xlsx',index=False)


####### the final patient list is derived from patient_list_seg_check.xlsx. remove the cases with have_seg == False, image_shape_consistency > 0, include_all_LV == False

### add the timeframes into the list
# patient_list = pd.read_excel('/mnt/camca_NAS/4DCT/Patient_lists/patient_list_final_selection.xlsx')
# ## build a patient list
# results = []
# for patient in range(0, patient_list.shape[0]):
#     patient_id = patient_list.iloc[patient]['patient_id']
#     patient_class = patient_list.iloc[patient]['patient_class']
#     image_folder = os.path.join('/mnt/camca_NAS/4DCT','nii-images',patient_class,patient_id,'img-nii-resampled-1.5mm')
#     files = ff.find_all_target_files(['*'],image_folder)
#     timeframes = len(files)
#     results.append(timeframes)
    
# patient_list['timeframes'] = results
# patient_list.to_excel(os.path.join('/mnt/camca_NAS/4DCT/Patient_lists','patient_list_final_selection.xlsx'),index=False)

# check intensity range
# patient_list = pd.read_excel('/mnt/camca_NAS/4DCT/Patient_lists/patient_list_final_selection.xlsx')
# results = []
# for patient in range(0, patient_list.shape[0]):
#     patient_id = patient_list.iloc[patient]['patient_id']
#     patient_class = patient_list.iloc[patient]['patient_class']
#     file = os.path.join('/mnt/camca_NAS/4DCT','nii-images',patient_class,patient_id,'img-nii-resampled-1.5mm/0.nii.gz')
#     img = nb.load(file).get_fdata()
#     if len(img.shape) == 4:
#         img = img[...,0]
#     seg_file = os.path.join('/mnt/camca_NAS/4DCT/predicted_seg',patient_class,patient_id,'seg-pred-0.625-4classes-connected-retouch-resampled-1.5mm/pred_s_0.nii.gz')
#     seg = nb.load(seg_file).get_fdata()
#     if len(seg.shape) == 4:
#         seg = seg[...,0]
    
#     img[seg!=1] = 0
#     results.append([patient_id,patient_class,np.min(img),np.max(img)])
# df = pd.DataFrame(results,columns=['patient_id','patient_class','min','max'])
# df.to_excel(os.path.join('/mnt/camca_NAS/4DCT/Patient_lists','patient_list_final_selection_intensity_range.xlsx'),index=False)
   
  