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
which_timeframe_is_template = 'others' # 'others' means warp 0 to other time frames, '0' means warp other time frames to 0
sample_slice_num = None # None means all slices

pre_epoch = 140
validation_every_epoch = 10  # Validate every N epochs
nb_epochs = 20000

save_path = os.path.join(main_path, 'models', trial_name)
ff.make_folder([save_path, os.path.join(save_path, 'logs')])


# set the data
data_sheet = os.path.join(main_path,'Patient_lists/patient_list_train_test.xlsx')

b = Build_list.Build(data_sheet)
patient_class_train_list, patient_id_train_list, tf_train_list = b.__build__(batch_list = [0]) 
patient_class_val_list, patient_id_val_list, tf_val_list = b.__build__(batch_list = [0])
# n=1
patient_class_train_list = patient_class_train_list[-1:]
patient_id_train_list = patient_id_train_list[-1:]
tf_train_list = tf_train_list[-1:]
patient_class_val_list = patient_class_val_list[-1:]
patient_id_val_list = patient_id_val_list[-1:]
tf_val_list = tf_val_list[-1:]
print(patient_id_train_list.shape, patient_id_val_list.shape, tf_train_list)


# set the model
if sample_slice_num == None:
    input_shape = [160,160,96]
else:
    input_shape = [160,160,sample_slice_num]

nb_features = [
    [16, 32, 32, 32],
    [32, 32, 32, 32, 32, 16, 16]
]
# nb_features = [
#     [32, 64, 64, 64],
#     [64, 64, 64, 64, 32, 16, 16]
# ]
vxm_model = vxm.networks.VxmDense(input_shape, nb_features, int_steps=0)
# voxelmorph has a variety of custom loss classes
losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]

# usually, we have to balance the two losses by a hyper-parameter
lambda_param = 0.05
loss_weights = [1, lambda_param]
vxm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=losses, loss_weights=loss_weights)

# set the generator
batch_size = 1
train_generator = Generator_voxelmorph.DataGenerator_alltf(
        patient_class_train_list,
        patient_id_train_list,
        tf_train_list,
        which_timeframe_is_template = which_timeframe_is_template,
        sample_slice_num = sample_slice_num,
        patient_num = len(patient_id_train_list),
        batch_size = batch_size,
        shuffle = True,
        normalize = True,
        adapt_shape = [160,160,96],
        augment = False,
        augment_frequency = 0.3,
        seed = 10)

val_generator = Generator_voxelmorph.DataGenerator_alltf(
        patient_class_val_list,
        patient_id_val_list,
        tf_val_list,
        which_timeframe_is_template = which_timeframe_is_template,
        sample_slice_num = sample_slice_num,
        patient_num = len(patient_id_val_list),
        batch_size = 1,
        shuffle = False,
        normalize = True,
        adapt_shape = [160,160,96],
        augment = None,
        augment_frequency = 0,
        seed = 20)

# val_generator_mid = Generator_voxelmorph.DataGenerator(patient_class_val_list, patient_id_val_list, which_timeframe_is_template = which_timeframe_is_template,sample_slice_num = sample_slice_num, patient_num = len(patient_id_val_list), batch_size = 1, shuffle = False, normalize = True, adapt_shape = [160,160,96], augment = None, augment_frequency =0, random_tf_sample= 'mid' ,seed = 20)
# val_generator_systole = Generator_voxelmorph.DataGenerator(patient_class_val_list, patient_id_val_list, which_timeframe_is_template = which_timeframe_is_template,sample_slice_num = sample_slice_num,patient_num = len(patient_id_val_list), batch_size = 1, shuffle = False, normalize = True, adapt_shape = [160,160,96], augment = None, augment_frequency =0, random_tf_sample= 'systole' ,seed = 2200)


# train
if pre_epoch != None:
    pre_model = os.path.join(save_path, 'vxm_model_epoch' + str(pre_epoch) + '.h5')

# Initialize an Excel sheet data storage
excel_results = []

if pre_epoch!= None:
    vxm_model.load_weights(pre_model)
    print('pre-trained model loaded')
    start_epoch = pre_epoch
else:
    start_epoch = 0

# Training loop
for epoch in range(start_epoch , start_epoch + nb_epochs):
    print(f"Epoch {epoch + 1}/{nb_epochs}")

    # Train the model for one epoch
    hist = vxm_model.fit(
        train_generator,
        epochs=1,
        verbose=1,
        use_multiprocessing=False,
        workers = 1,
        shuffle = False,
    )

    # Get the training loss
    training_loss = hist.history['loss'][0]
    transformer_loss = hist.history.get('vxm_dense_transformer_loss', [None])[0]
    flow_loss = hist.history.get('vxm_dense_flow_loss', [None])[0]


    # Calculate validation loss every N epochs
    if (epoch+1) % validation_every_epoch == 0:
        # val_hist_mid = vxm_model.evaluate(val_generator_mid, verbose=1, return_dict=True)
        # val_loss_mid = val_hist_mid['loss']
        # val_transformer_loss_mid = val_hist_mid.get('vxm_dense_transformer_loss', None)
        # val_flow_loss_mid = val_hist_mid.get('vxm_dense_flow_loss', None)

        # val_hist_systole = vxm_model.evaluate(val_generator_systole, verbose=1, return_dict=True)
        # val_loss_systole = val_hist_systole['loss']
        # val_transformer_loss_systole = val_hist_systole.get('vxm_dense_transformer_loss', None)
        # val_flow_loss_systole = val_hist_systole.get('vxm_dense_flow_loss', None)


        val_hist = vxm_model.evaluate(val_generator, verbose=1, return_dict=True)
        val_loss = val_hist['loss']
        val_transformer_loss = val_hist.get('vxm_dense_transformer_loss', None)
        val_flow_loss = val_hist.get('vxm_dense_flow_loss', None)

    
        epoch_results = [epoch + 1, training_loss, transformer_loss, flow_loss, val_loss, val_transformer_loss, val_flow_loss]
        print('epoch results:', epoch_results)
        excel_results.append(epoch_results)
        df = pd.DataFrame(excel_results, columns=['Epoch', 'Training Loss', 'Transformer Loss', 'Flow Loss', 'Validation Loss', 'Validation Transformer Loss', 'Validation Flow Loss'])
        file_name = os.path.join(save_path, 'logs/training_metrics.xlsx')
        df.to_excel(file_name, index=False)

        # Save the model parameters for each epoch
        vxm_model.save(os.path.join(save_path,'vxm_model_epoch'+str(epoch + 1)+'.h5'))
        print(f"Model saved as 'vxm_model_epoch_{epoch + 1}.h5'")