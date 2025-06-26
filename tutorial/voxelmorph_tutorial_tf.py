# imports
import os, sys

# third party imports
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'

import voxelmorph as vxm
import neurite as ne
import pandas as pd

from tensorflow.keras.datasets import mnist

save_path = '/mnt/camca_NAS/4DCT/models/'

# load MNIST data. 
# `mnist.load_data()` already splits our data into train and test.  
(x_train_load, y_train_load), (x_test_load, y_test_load) = mnist.load_data()

digit_sel = 5

# extract only instances of the digit 5
x_train = x_train_load[y_train_load==digit_sel, ...]
y_train = y_train_load[y_train_load==digit_sel]
x_test = x_test_load[y_test_load==digit_sel, ...]
y_test = y_test_load[y_test_load==digit_sel]

# let's get some shapes to understand what we loaded.
print('shape of x_train: {}, y_train: {}'.format(x_train.shape, y_train.shape))

nb_val = 1000  # keep 1,000 subjects for validation
x_val = x_train[-nb_val:, ...]  # this indexing means "the last nb_val entries" of the zeroth axis
y_val = y_train[-nb_val:]
x_train = x_train[:-nb_val, ...]
y_train = y_train[:-nb_val]

# fix data
x_train = x_train.astype('float')/255
x_val = x_val.astype('float')/255
x_test = x_test.astype('float')/255

pad_amount = ((0, 0), (2,2), (2,2))

# fix data
x_train = np.pad(x_train, pad_amount, 'constant')
x_val = np.pad(x_val, pad_amount, 'constant')
x_test = np.pad(x_test, pad_amount, 'constant')

# verify
print('shape of training data', x_train.shape)

# configure unet input shape (concatenation of moving and fixed images)
ndim = 2
unet_input_features = 2
inshape = (*x_train.shape[1:], unet_input_features)

# configure unet features 
nb_features = [
    [32, 32, 32, 32],         # encoder features
    [32, 32, 32, 32, 32, 16]  # decoder features
]

inshape = x_train.shape[1:]
vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)
print('input shape: ', ', '.join([str(t.shape) for t in vxm_model.inputs]))
print('output shape:', ', '.join([str(t.shape) for t in vxm_model.outputs]))

# voxelmorph has a variety of custom loss classes
losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]

# usually, we have to balance the two losses by a hyper-parameter
lambda_param = 0.05
loss_weights = [1, lambda_param]

vxm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss=losses, loss_weights=loss_weights)

def vxm_data_generator(x_data, batch_size=32):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """

    # preliminary sizing
    vol_shape = x_data.shape[1:] # extract data shape
    ndims = len(vol_shape)
    
    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])
    
    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        moving_images = x_data[idx1, ..., np.newaxis]
        idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        fixed_images = x_data[idx2, ..., np.newaxis]
        inputs = [moving_images, fixed_images]
        
        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare 
        # the resulting moved image with the fixed image. 
        # we also wish to penalize the deformation field. 
        outputs = [fixed_images, zero_phi]
        
        yield (inputs, outputs)

batch_size = 32
train_generator = vxm_data_generator(x_train,batch_size = batch_size)
val_generator = vxm_data_generator(x_val, batch_size=batch_size)


print('I am ok here!!!!')

training_steps = x_train.shape[0] // batch_size  # Number of steps for training per epoch
validation_steps =  x_val.shape[0] // batch_size # Number of steps for validation per epoch
validation_every_epoch = 1  # Validate every N epochs

nb_epochs = 1000000000000000000000000000000000000**2**2

# Initialize an Excel sheet data storage
excel_results = []

# Training loop
for epoch in range(nb_epochs):
    print(f"Epoch {epoch + 1}/{nb_epochs}")

    # Train the model for one epoch
    hist = vxm_model.fit(
        train_generator,
        steps_per_epoch= training_steps,
        epochs=1,
        verbose=1
    )

    # Get the training loss
    training_loss = hist.history['loss'][0]
    transformer_loss = hist.history.get('vxm_dense_transformer_loss', [None])[0]
    flow_loss = hist.history.get('vxm_dense_flow_loss', [None])[0]


    # Calculate validation loss every N epochs
    if epoch % validation_every_epoch == 0:
        val_hist = vxm_model.evaluate(val_generator, steps=validation_steps, verbose=1, return_dict=True)
        val_loss = val_hist['loss']
        val_transformer_loss = val_hist.get('vxm_dense_transformer_loss', None)
        val_flow_loss = val_hist.get('vxm_dense_flow_loss', None)
    
        epoch_results = [epoch + 1, training_loss, transformer_loss, flow_loss, val_loss, val_transformer_loss, val_flow_loss]
        print('epoch results:', epoch_results)
        excel_results.append(epoch_results)
        df = pd.DataFrame(excel_results, columns=['Epoch', 'Training Loss', 'Transformer Loss', 'Flow Loss', 'Validation Loss', 'Validation Transformer Loss', 'Validation Flow Loss'])
        file_name = os.path.join(save_path, 'training_metrics.xlsx')
        # df.to_excel(file_name, index=False)

        # Save the model parameters for each epoch
        # vxm_model.save(os.path.join(save_path,'vxm_model_epoch'+str(epoch + 1)+'.h5'))
        print(f"Model saved as 'vxm_model_epoch_{epoch + 1}.h5'")

