import sys
sys.path.append('/workspace/Documents')
# imports
import os, sys

# third party imports
import numpy as np 
import tensorflow as tf
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'

import voxelmorph as vxm
import neurite as ne
import pandas as pd
import random
import nibabel as nb

from tensorflow.keras.utils import Sequence
import Diffusion_motion_field.Build_lists.Build_list as Build_list
import Diffusion_motion_field.functions_collection as ff
import Diffusion_motion_field.Data_processing as Data_processing
import Diffusion_motion_field.Generator_voxelmorph as Generator_voxelmorph

main_path = '/mnt/camca_NAS/4DCT'

trial_name = 'voxel_morph_warp0_onecase'
which_timeframe_is_template = 'others'
pre_epoch = 300
# pre_model = os.path.join(main_path, 'models', trial_name, 'vxm_model_epoch' + str(pre_epoch) + '.h5')
pre_model = os.path.join(main_path,'models', trial_name, 'individual_models', 'Abnormal/CVC2001241336/models','vxm_'+str(pre_epoch)+'.h5')
print('pred model!', pre_model)
sample_slice_num = None

 
# set the data
data_sheet = os.path.join(main_path,'Patient_lists/patient_list_train_test_reorder.xlsx')

b = Build_list.Build(data_sheet)
patient_class_test_list, patient_id_test_list,_ = b.__build__(batch_list = [0]) 
print(patient_id_test_list.shape)


# set the model
if sample_slice_num == None:
    input_shape = [160,160,96]
else:
    input_shape = [160,160,sample_slice_num]

nb_features = [
    [16, 32, 32, 32],
    [32, 32, 32, 32, 32, 16, 16]
]

vxm_model = vxm.networks.VxmDense(input_shape, nb_features, int_steps=0)
# voxelmorph has a variety of custom loss classes
losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]

# usually, we have to balance the two losses by a hyper-parameter
lambda_param = 0.05
loss_weights = [1, lambda_param]
vxm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=losses, loss_weights=loss_weights)
vxm_model.load_weights(pre_model)


# main test
MAE_list = []
for i in range(0,1):
    patient_id = patient_id_test_list[i]
    patient_class = patient_class_test_list[i]
    print(patient_class, patient_id)

    path = os.path.join(main_path,'nii-images' ,patient_class, patient_id,'img-nii-resampled-1.5mm')
    ff.make_folder([os.path.join('/mnt/camca_NAS/4DCT/mvf_warp0_onecase')])
    save_path = os.path.join('/mnt/camca_NAS/4DCT/mvf_warp0_onecase', patient_class, patient_id, trial_name+'_epoch' + str(pre_epoch))
    ff.make_folder([os.path.dirname(os.path.dirname(save_path)), os.path.dirname(save_path), save_path])

    tf_files = ff.sort_timeframe(ff.find_all_target_files(['*.nii.gz'],path),2)

    affine = nb.load(tf_files[0]).affine
    image_shape = nb.load(tf_files[0]).shape
    
    for timeframe in range(0,len(tf_files)):
        original_image = nb.load(tf_files[timeframe]).get_fdata()
        if timeframe == 0:
            mvf = np.zeros([160,160,96,3]) 
            mae = 0
            moved_pred = nb.load(tf_files[0]).get_fdata()
        else:
            # print('now doing timeframe:', timeframe, tf_files[0], tf_files[timeframe])
            if which_timeframe_is_template == 'others':
                tf1 = nb.load(tf_files[0]).get_fdata()
                tf2 = nb.load(tf_files[timeframe]).get_fdata()
            else:
                tf1 = nb.load(tf_files[timeframe]).get_fdata()
                tf2 = nb.load(tf_files[0]).get_fdata()

            if len(tf1.shape) == 4:
                tf1 = tf1[...,0]
            if len(tf2.shape) == 4:
                tf2 = tf2[...,0]
            
            tf1 = Data_processing.crop_or_pad(tf1, [160,160,96], value = np.min(tf1)) / 1000
            tf2 = Data_processing.crop_or_pad(tf2, [160,160,96], value = np.min(tf2)) / 1000

            if sample_slice_num == None:
                val_input = [ tf1[np.newaxis, ..., np.newaxis],
                    tf2[np.newaxis, ..., np.newaxis]]
                
                val_pred = vxm_model.predict(val_input)
                moved_pred = val_pred[0].squeeze() * 1000
                print('moved_pred shape:', moved_pred.shape)
                pred_warp = val_pred[1]

                mae = np.mean(np.abs(tf2*1000 - moved_pred))
                mvf = pred_warp.squeeze()
            else:
                mvf = np.zeros([160,160,96,3]); mae_list = []
                pred_image = np.zeros([160,160,96])
                for i in range(0,3):
                    tf1_slice = tf1[...,i*sample_slice_num:(i+1)*sample_slice_num]
                    tf2_slice = tf2[...,i*sample_slice_num:(i+1)*sample_slice_num]
                    val_input = [ tf1_slice[np.newaxis, ..., np.newaxis],
                        tf2_slice[np.newaxis, ..., np.newaxis]]
                    val_pred = vxm_model.predict(val_input)
                    pred_warp = val_pred[1]
                    moved_pred = val_pred[0].squeeze() * 1000
                    mae_list.append(np.mean(np.abs(tf2_slice*1000 - moved_pred)))
                    mvf[...,i*sample_slice_num:(i+1)*sample_slice_num,:] = pred_warp.squeeze()
                    pred_image[...,i*sample_slice_num:(i+1)*sample_slice_num] = moved_pred
                mae = np.mean(mae_list)
                moved_pred = np.copy(pred_image)
        
        print('mvf max and min:', np.max(mvf), np.min(mvf) )
            
        save_file = os.path.join(save_path, str(timeframe) + '.nii.gz')
        img = nb.Nifti1Image(mvf, affine)
        nb.save(img, save_file)

        # moved_pred = Data_processing.crop_or_pad(moved_pred, [image_shape[0], image_shape[1], image_shape[2]], value = np.min(moved_pred))
        moved_pred_img = nb.Nifti1Image(moved_pred, affine)
        nb.save(moved_pred_img, os.path.join(save_path, str(timeframe) + '_moved.nii.gz'))

        original_image = Data_processing.crop_or_pad(original_image, [160,160,96], value = np.min(original_image))
        nb.save(nb.Nifti1Image(original_image, affine), os.path.join(save_path, str(timeframe) + '_original.nii.gz'))

        MAE_list.append([patient_class, patient_id, timeframe, mae])
        print('timeframe:', timeframe, 'mae:', mae)

MAE_df = pd.DataFrame(MAE_list, columns = ['patient_class', 'patient_id', 'timeframe', 'MAE'])
MAE_df.to_excel(os.path.join(main_path, 'models', trial_name, 'MAE_list.xlsx'), index = False)