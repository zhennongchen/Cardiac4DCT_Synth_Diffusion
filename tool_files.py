import sys 
sys.path.append('/workspace/Documents')
import os
import pandas as pd
import numpy as np
import shutil
import Diffusion_motion_field.functions_collection as ff

### transfer MVF from NAS to local
# nas_path = '/mnt/camca_NAS/4DCT/mvf_warp0_onecase'
# local_path = '/workspace/Documents/Data/mvf'

# patient_sheet = pd.read_excel(os.path.join('/mnt/camca_NAS/4DCT/Patient_lists','patient_list_MVF_diffusion_train_test_filtered.xlsx'))

  
# for i in range(0, len(patient_sheet)):
#     patient_id = patient_sheet['patient_id'][i]
#     patient_class = patient_sheet['patient_class'][i]
#     print('patient class:', patient_class, 'patient id:', patient_id)

#     path = os.path.join(nas_path,patient_class, patient_id, 'voxel_final')
#     # all files
#     files = ff.find_all_target_files(['*.nii.gz'],path)
#     final_files = np.copy(files)
#     for f in files:
#         if 'moved' in f or 'original' in f:
#             # remove it from the numpy array
#             final_files = np.delete(final_files, np.where(final_files == f))
#     files = ff.sort_timeframe(final_files,2)

#     save_folder = os.path.join(local_path, patient_class, patient_id, 'voxel_final')
#     ff.make_folder([os.path.join(local_path, patient_class), os.path.join(local_path, patient_class, patient_id), save_folder])
#     for f in files:
#         if os.path.isfile(os.path.join(save_folder, os.path.basename(f))):
#             print('file exists:', os.path.join(save_folder, os.path.basename(f)))
#         shutil.copy(f, os.path.join(save_folder, os.path.basename(f)))

#     # copy image tf0 as well
#     img_path = os.path.join('/mnt/camca_NAS/4DCT','nii-images',patient_class, patient_id, 'img-nii-resampled-1.5mm/0.nii.gz')
#     save_img_folder = os.path.join(local_path, patient_class, patient_id, 'img-nii-resampled-1.5mm')
#     ff.make_folder([os.path.join(local_path, patient_class), os.path.join(local_path, patient_class, patient_id), save_img_folder])
#     if os.path.isfile(os.path.join(save_img_folder, os.path.basename(img_path))):
#         print('file exists:', os.path.join(save_img_folder, os.path.basename(img_path)))
#     else:
#         shutil.copy(img_path, os.path.join(save_img_folder, os.path.basename(img_path)))


# transfer MVF from NAS to local
nas_path = '/mnt/camca_NAS/4DCT/mvf_aug'
local_path = '/workspace/Documents/Data/mvf_aug'
ff.make_folder([local_path])

# patient_sheet = pd.read_excel(os.path.join('/mnt/camca_NAS/4DCT/Patient_lists/uc','patient_list_MVF_diffusion_train_test_filtered_at_least_10tf.xlsx'))
patient_sheet = pd.read_excel(os.path.join('/mnt/camca_NAS/4DCT/Patient_lists/mgh','patient_list_MVF_diffusion_train_test.xlsx'))

for i in range( 0, len(patient_sheet)):
    patient_id = patient_sheet['patient_id'][i]
    patient_class = patient_sheet['patient_class'][i]
   
    print('patient class:', patient_class, 'patient id:', patient_id)

    path = os.path.join(nas_path,patient_class, patient_id)
    # all aug folders
    aug_folders = ff.sort_timeframe(ff.find_all_target_files(['aug_0','aug_1','aug_2','aug_3', 'aug_4','aug_5'],path),0,'_')
    print(aug_folders)
    assert len(aug_folders) > 0, 'no aug folder found'

    for aug_folder in aug_folders:
        aug_folder_des = os.path.join(local_path, patient_class, patient_id, os.path.basename(aug_folder))
        ff.make_folder([os.path.join(local_path, patient_class), os.path.join(local_path, patient_class, patient_id), aug_folder_des])
        if os.path.isfile(os.path.join(aug_folder_des,'aug_parameter.npy')) == 0:
            shutil.copy(os.path.join(aug_folder,'aug_parameter.npy'), os.path.join(aug_folder_des,'aug_parameter.npy'))

        # all latent files
        # if os.path.isdir(os.path.join(aug_folder_des,'latent')) == 0:
        #     shutil.copytree(os.path.join(aug_folder,'latent'), os.path.join(aug_folder_des,'latent'))

        # all MVF files
        # if os.path.isdir(os.path.join(aug_folder_des,'mvf')) == 0:
        #     shutil.copytree(os.path.join(aug_folder,'mvf'), os.path.join(aug_folder_des,'mvf'))

        # all MVF downsampled files
        # if os.path.isdir(os.path.join(aug_folder_des,'mvf_downsampled')) == 0:
        #     shutil.copytree(os.path.join(aug_folder,'mvf_downsampled'), os.path.join(aug_folder_des,'mvf_downsampled'))
            
        # condition image
        # if os.path.isdir(os.path.join(aug_folder_des,'condition_img')) == 0:
        #     shutil.copytree(os.path.join(aug_folder,'condition_img'), os.path.join(aug_folder_des,'condition_img'))
        
        # condition seg
        # if os.path.isdir(os.path.join(aug_folder_des,'segmentation')) == 0:
        #     shutil.copytree(os.path.join(aug_folder,'segmentation'), os.path.join(aug_folder_des,'segmentation'))
            
        # condition seg original resolution
        if os.path.isdir(os.path.join(aug_folder_des,'segmentation_original_res')) == 0:
            shutil.copytree(os.path.join(aug_folder,'segmentation_original_res'), os.path.join(aug_folder_des,'segmentation_original_res'))
            

# # delete
# main_path = '/workspace/Documents/Data/mvf_aug'
# cases = ff.find_all_target_files(['Normal/CVC1909301536_AN151/aug_*/mvf_downsampled'],main_path)
# print('cases:', cases)
# for case in cases:
#     shutil.rmtree(case)