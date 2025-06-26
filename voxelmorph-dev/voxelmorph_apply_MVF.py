import sys
sys.path.append('/workspace/Documents')
# imports
import os, sys

# third party imports
import numpy as np 
import pandas as pd
import random
import nibabel as nb
from scipy.ndimage import map_coordinates
import Diffusion_motion_field.Build_lists.Build_list as Build_list
import Diffusion_motion_field.functions_collection as ff
import Diffusion_motion_field.Data_processing as Data_processing
import Diffusion_motion_field.Generator_voxelmorph as Generator_voxelmorph

main_path = '/mnt/camca_NAS/4DCT'
study_name = 'mvf_warp0'
model_name ='voxel_morph_warp0_epoch153'
save_path = os.path.join(main_path,'models/voxel_morph_warp0/warped_images')

def apply_deformation_field_numpy(moving_image, deformation_field):
    """
    Apply a deformation field to the moving image using trilinear interpolation.

    Args:
        moving_image (np.ndarray): The moving image of shape [H, W, D].
        deformation_field (np.ndarray): The deformation field of shape [H, W, D, 3].

    Returns:
        np.ndarray: Warped image of shape [H, W, D].
    """
    # Get the grid of coordinates
    H, W, D = moving_image.shape
    grid_x, grid_y, grid_z = np.meshgrid(
        np.arange(H), np.arange(W), np.arange(D), indexing='ij'
    )

    # Add the deformation field to the grid
    deformed_x = grid_x + deformation_field[..., 0]
    deformed_y = grid_y + deformation_field[..., 1]
    deformed_z = grid_z + deformation_field[..., 2]

    # Flatten the coordinates for interpolation
    coordinates = np.array([deformed_x.flatten(), deformed_y.flatten(), deformed_z.flatten()])

    # Interpolate using map_coordinates
    warped_image = map_coordinates(moving_image, coordinates, order=1, mode='nearest').reshape(H, W, D)

    return warped_image


# set the data
data_sheet = os.path.join(main_path,'Patient_lists/patient_list_train_test.xlsx')

b = Build_list.Build(data_sheet)
patient_class_test_list, patient_id_test_list = b.__build__(batch_list = [0,1,2,3,4,5]) 
print(patient_id_test_list.shape)


for i in range(0,patient_id_test_list.shape[0]):
    patient_id = patient_id_test_list[i]
    patient_class = patient_class_test_list[i]
    print(patient_class, patient_id)

    img_path = os.path.join(main_path,'nii-images' ,patient_class, patient_id,'img-nii-resampled-1.5mm')
    save_folder = os.path.join(save_path, patient_class, patient_id)
    ff.make_folder([os.path.dirname(save_folder), save_folder])

    tf_files = ff.sort_timeframe(ff.find_all_target_files(['*.nii.gz'],img_path),2)
    template_image = nb.load(tf_files[0]).get_fdata()
    if len(template_image.shape) == 4:
        template_image = template_image[:,:,:,0]
    template_image = Data_processing.crop_or_pad(template_image, [160,160,96], value = np.min(template_image))

    affine = nb.load(tf_files[0]).affine

    if os.path.isfile(os.path.join(save_folder, '6.nii.gz')) == 1:
        continue
    
    for timeframe in range(0,len(tf_files)):
        mvf = nb.load(os.path.join(main_path,study_name, patient_class, patient_id, model_name, str(timeframe)+'.nii.gz')).get_fdata()
        warped_img = apply_deformation_field_numpy(np.copy(template_image), mvf)
        nb.save(nb.Nifti1Image(warped_img, affine), os.path.join(save_folder, str(timeframe)+'.nii.gz'))

       