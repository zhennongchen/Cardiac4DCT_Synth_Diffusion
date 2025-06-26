# train and predict voxelmorph model for each individual case
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

main_path = '/mnt/camca_NAS/4DCT/'

trial_name = 'voxel_morph_warp0_onecase'
which_timeframe_is_template = 'others' # 'others' means warp 0 to other time frames, '0' means warp other time frames to 0
sample_slice_num = None # None means all slices

######## Training ########
# save_every_epoch = 10

# ## set patient list
# # set the data
# # data_sheet = os.path.join(main_path,'Patient_lists/uc/patient_list_train_test_reorder.xlsx')
# data_sheet = os.path.join('/mnt/camca_NAS/4DCT/Patient_lists/mgh/patient_list_selected.xlsx')

# b = Build_list.Build(data_sheet)
# patient_class_train_list, patient_id_train_list, tf_train_list = b.__build__(batch_list = [0]) 
# # n=1
# # patient_class_train_list = patient_class_train_list[0:n]
# # patient_id_train_list = patient_id_train_list[0:n]
# # tf_train_list = tf_train_list[0:n]
# print(patient_id_train_list.shape)#,  tf_train_list)

# results = []
# for i in range(patient_id_train_list.shape[0]//5*3, patient_id_train_list.shape[0]):
#     patient_class = patient_class_train_list[i]
#     patient_id = patient_id_train_list[i]
#     tf_num = tf_train_list[i]
    
#     print(patient_class, patient_id, tf_num)

#     ### set save path
#     save_path = os.path.join(main_path, 'models', trial_name, 'individual_models',patient_class,patient_id)
#     ff.make_folder([os.path.dirname(save_path), save_path, os.path.join(save_path, 'logs'),os.path.join(save_path, 'models')])

#     # ### check whether the patient has been processed
#     if os.path.isfile(os.path.join(save_path,'models/vxm_final.h5')):
#         print('patient:', patient_id, 'has been processed')
#         continue

#     ### build the model
#     if sample_slice_num == None:
#         input_shape = [160,160,96]
#     else:
#         input_shape = [160,160,sample_slice_num]
#     nb_features = [[16, 32, 32, 32],[32, 32, 32, 32, 32, 16, 16]]
#     vxm_model = vxm.networks.VxmDense(input_shape, nb_features, int_steps=0)
#     # voxelmorph has a variety of custom loss classes
#     losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
#     loss_weights = [1, 0.05]
#     vxm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=losses, loss_weights=loss_weights)

#     ### set the generator
#     train_generator = Generator_voxelmorph.DataGenerator_alltf(
#             np.asarray([patient_class]),
#             np.asarray([patient_id]),
#             np.asarray([tf_num]),
#             which_timeframe_is_template = which_timeframe_is_template,
#             sample_slice_num = sample_slice_num,
#             patient_num = 1,
#             batch_size = 1,
#             shuffle = False,
#             normalize = True,
#             adapt_shape = [160,160,96],
#             augment = False,augment_frequency = 0,seed = 10)
    
#     ### check whether there is a pre-trained model
#     pre_model_list = ff.find_all_target_files(['vxm_*'],os.path.join(save_path, 'models'))
#     pre_model_list = np.delete(pre_model_list, np.where(pre_model_list == os.path.join(save_path, 'models/vxm_final.h5')))
  
#     if len(pre_model_list) == 0:
#         start_epoch = 0
#         print('no pre-trained model')
#     else:
#         pre_model_list = ff.sort_timeframe(pre_model_list,1,'_')
#         pre_model = pre_model_list[-1]
#         start_epoch = ff.find_timeframe(pre_model,1,'_')
#         # vxm_model.load_weights(pre_model)
#         print('pre-trained model loaded, epoch:', start_epoch)

#     # ### train the model
#     nb_epochs = 2000

#     ### Initialize an Excel sheet data storage
#     loss_results = []

#     ### training loop
#     previous_loss = 100; freeze_count = 0
#     for epoch in range(start_epoch , start_epoch + nb_epochs):
#         print(f"Epoch {epoch + 1}/{nb_epochs}")

#         # Train the model for one epoch
#         hist = vxm_model.fit(train_generator, epochs=1, verbose=1,use_multiprocessing=False,workers = 1, shuffle = False,)

#         # Get the training loss
#         training_loss = hist.history['loss'][0]
#         transformer_loss = hist.history.get('vxm_dense_transformer_loss', [None])[0]
#         flow_loss = hist.history.get('vxm_dense_flow_loss', [None])[0]

#         if (epoch + 1) % save_every_epoch == 0:
#             # save the loss results
#             epoch_results = [epoch + 1, training_loss, transformer_loss, flow_loss]
#             loss_results.append(epoch_results)
#             df = pd.DataFrame(loss_results, columns=['Epoch', 'Training Loss', 'Transformer Loss', 'Flow Loss'])
#             file_name = os.path.join(save_path, 'logs/training_metrics.xlsx')
#             df.to_excel(file_name, index=False)

#             # Save the model parameters for each epoch
#             vxm_model.save(os.path.join(save_path,'models/vxm_'+str(epoch + 1)+'.h5'))

#             training_loss_round = round(training_loss, 4)
#             # check whether we should stop the training
#             if training_loss_round < previous_loss:
#                 previous_loss = training_loss_round; freeze_count = 0
#             else:
#                 freeze_count += 1

#             if epoch <= 150:
#                 continue # at least train 150 epochs

#             if training_loss_round <= 0.0021 or epoch >= 300:
#                 print('training loss is less than 0.0021 or epoch >= 300, stop at epoch:', epoch)
#                 # copy and paste the last model to the final model
#                 vxm_model.save(os.path.join(save_path,'models/vxm_final.h5'))
#                 break
            
#             if freeze_count >= 4: # 40 epochs no improvement
#                 print('training loss has not improved for 40 epochs, stop at epoch:', epoch)
#                 # copy and paste the last model to the final model
#                 vxm_model.save(os.path.join(save_path,'models/vxm_final.h5'))
#                 break
            

####### Testing ########
# set the data
data_sheet = os.path.join(main_path,'Patient_lists/patient_list_train_test_reorder.xlsx')
data_sheet = os.path.join('/mnt/camca_NAS/4DCT/Patient_lists/mgh/patient_list_selected.xlsx')

b = Build_list.Build(data_sheet)
patient_class_test_list, patient_id_test_list, tf_test_list = b.__build__(batch_list = [0]) 

MAE_list = []
for i in range(0, patient_id_test_list.shape[0]):
    patient_class = patient_class_test_list[i]
    patient_id = patient_id_test_list[i]
    tf_num = tf_test_list[i]
   
    print(patient_class, patient_id, tf_num)

    ### check whether the we have the voxel final model
    model_path = os.path.join(main_path, 'models', trial_name, 'individual_models',patient_class,patient_id,'models/vxm_final.h5')
    if not os.path.isfile(model_path):
        print('no model for patient:', patient_id)
        continue

    ### set save path
    save_path = os.path.join(main_path, 'mvf_warp0_onecase',patient_class,patient_id, 'voxel_final')
    ff.make_folder([os.path.dirname(os.path.dirname(save_path)), os.path.dirname(save_path), save_path])

    ### build the model
    if sample_slice_num == None:
        input_shape = [160,160,96]
    else:
        input_shape = [160,160,sample_slice_num]
    nb_features = [[16, 32, 32, 32],[32, 32, 32, 32, 32, 16, 16]]
    vxm_model = vxm.networks.VxmDense(input_shape, nb_features, int_steps=0)
    # voxelmorph has a variety of custom loss classes
    losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
    loss_weights = [1, 0.05]
    vxm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=losses, loss_weights=loss_weights)
    vxm_model.load_weights(model_path)

    ### do the prediction
    image_path = os.path.join(main_path,'mgh_data/nii-images' ,patient_class, patient_id,'img-nii-resampled-1.5mm')
    tf_files = ff.sort_timeframe(ff.find_all_target_files(['*.nii.gz'],image_path),2)

    affine = nb.load(tf_files[0]).affine
    image_shape = nb.load(tf_files[0]).shape
    
    for timeframe in range(0,len(tf_files)):
        if os.path.isfile(os.path.join(save_path, str(timeframe) + '.nii.gz')) == 1:
            print('timeframe:', timeframe, 'has been processed')
            move_pred = nb.load(os.path.join(save_path, str(timeframe) + '_moved.nii.gz')).get_fdata()
            move_original = nb.load(os.path.join(save_path, str(timeframe) + '_original.nii.gz')).get_fdata()
            mae = np.mean(np.abs(move_original - move_pred))
        else:
            original_image = nb.load(tf_files[timeframe]).get_fdata()
            if timeframe == 0:
                mvf = np.zeros([160,160,96,3]) 
                mae = 0
                moved_pred = nb.load(tf_files[0]).get_fdata()
                moved_pred = Data_processing.crop_or_pad(moved_pred, [160,160,96], value = np.min(original_image))
            else:
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
                    pred_warp = val_pred[1]

                    mae = np.mean(np.abs(tf2*1000 - moved_pred))
                    mvf = pred_warp.squeeze()
                
            save_file = os.path.join(save_path, str(timeframe) + '.nii.gz')
            img = nb.Nifti1Image(mvf, affine)
            nb.save(img, save_file)

            moved_pred_img = nb.Nifti1Image(moved_pred, affine)
            nb.save(moved_pred_img, os.path.join(save_path, str(timeframe) + '_moved.nii.gz'))

            original_image = Data_processing.crop_or_pad(original_image, [160,160,96], value = np.min(original_image))
            nb.save(nb.Nifti1Image(original_image, affine), os.path.join(save_path, str(timeframe) + '_original.nii.gz'))

        MAE_list.append([patient_class, patient_id, timeframe, mae])
        print('timeframe:', timeframe, 'mae:', mae)

MAE_df = pd.DataFrame(MAE_list, columns = ['patient_class', 'patient_id', 'timeframe', 'MAE'])
MAE_df.to_excel(os.path.join(main_path, 'models', trial_name, 'MAE_list_mgh.xlsx'), index = False)