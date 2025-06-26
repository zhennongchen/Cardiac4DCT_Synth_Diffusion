import sys
sys.path.append('/workspace/Documents')
# imports
import os, sys

# third party imports
import numpy as np 

import pandas as pd
import random
import nibabel as nb

import Diffusion_motion_field.Build_lists.Build_list as Build_list
import Diffusion_motion_field.functions_collection as ff
import Diffusion_motion_field.Data_processing as Data_processing

main_path = '/mnt/camca_NAS/4DCT'

# check MVF
# p = ff.find_all_target_files(['Normal/*','Abnormal/*'],os.path.join(main_path,'mvf_warp0_onecase'))

# result = []
# for i in range(0,len(p)):
#     path = os.path.join(p[i],'voxel_final' )
#     files = ff.find_all_target_files(['*.nii.gz'],path)
#     final_files = np.copy(files)
#     for f in files:
#         if 'moved' in f or 'original' in f:
#             # remove it from the numpy array
#             final_files = np.delete(final_files, np.where(final_files == f))
#     print(len(final_files))
#     zz = np.zeros([160,160,96,3,len(final_files)])

#     for j in range(0,len(final_files)):
#         zz[:,:,:,:,j] = nb.load(final_files[j]).get_fdata()

#     print(os.path.basename(p[i]),np.max(zz), np.min(zz))
#     result.append([os.path.basename(os.path.dirname(p[i])),os.path.basename(p[i]), np.max(zz), np.min(zz)])
#     df = pd.DataFrame(result, columns = ['patient_class', 'patient_id', 'max', 'min'])
#     df.to_excel(os.path.join(main_path,'mvf_warp0_onecase','check_mvf_max_min.xlsx'), index = False)

# check MVF_latent
result = []
patient_list = pd.read_excel(os.path.join(main_path,'Patient_lists/patient_list_MVF_diffusion_train_test_filtered.xlsx'))
for i in range(0,patient_list.shape[0]):
    patient_class = patient_list['patient_class'][i]
    patient_id = patient_list['patient_id'][i]
    path = os.path.join(main_path,'models/VAE_embed3/pred_mvf',patient_class,patient_id,'epoch54')
    files = ff.sort_timeframe(ff.find_all_target_files(['pred_latent*.nii.gz'],path),2,'f')

    zz = np.zeros([40,40,24,3,len(files)])
    for j in range(0,len(files)):
        zz[:,:,:,:,j] = nb.load(files[j]).get_fdata()
    print(patient_id,np.max(zz), np.min(zz))
    result.append([patient_class,patient_id, np.max(zz), np.min(zz)])
    df = pd.DataFrame(result, columns = ['patient_class', 'patient_id', 'max', 'min'])
    df.to_excel(os.path.join(main_path,'models/VAE_embed3/pred_mvf','check_mvf_latent_max_min.xlsx'), index = False)


# process: make it smooth
# from scipy.ndimage import gaussian_filter
# p = ff.find_all_target_files(['Normal/CVC1803280940'],os.path.join(main_path,'mvf_warp0_onecase'))

# result = []
# for i in range(0,len(p)):
#     path = os.path.join(p[i],'voxel_final' )
#     files = ff.find_all_target_files(['*.nii.gz'],path)
#     final_files = np.copy(files)
#     for f in files:
#         if 'moved' in f or 'original' in f:
#             # remove it from the numpy array
#             final_files = np.delete(final_files, np.where(final_files == f))
#     print(len(final_files))
#     files = ff.sort_timeframe(final_files,2)
#     print(files)

#     for t in range(0, len(files)):
#         img = nb.load(files[t]).get_fdata()
#         img[abs(img)<=0.55] = 0
#         smooth_image = gaussian_filter(img, sigma=1)
#         # smooth_image = np.copy(img)
#         ff.make_folder([os.path.join(p[i],'voxel_final_smooth')])
#         nb.save(nb.Nifti1Image(smooth_image, nb.load(files[t]).affine), os.path.join(p[i],'voxel_final_smooth', os.path.basename(files[t])))

#     break

