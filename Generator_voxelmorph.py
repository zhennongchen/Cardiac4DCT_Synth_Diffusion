import sys
sys.path.append('/workspace/Documents')
# imports
import os, sys

# third party imports
import numpy as np 

import random
import nibabel as nb

from tensorflow.keras.utils import Sequence
import Cardiac4DCT_Synth_Diffusion.functions_collection as ff
import Cardiac4DCT_Synth_Diffusion.Data_processing as Data_processing

# random function
def random_rotate(i, z_rotate_degree = None, z_rotate_range = [0,0], fill_val = None, order = 0):
    # only do rotate according to z (in-plane rotation)
    if z_rotate_degree is None:
        z_rotate_degree = random.uniform(z_rotate_range[0], z_rotate_range[1])

    if fill_val is None:
        fill_val = np.min(i)
    
    if z_rotate_degree == 0:
        return i, z_rotate_degree
    else:
        return Data_processing.rotate_image(np.copy(i), [0,0,z_rotate_degree], order = order, fill_val = fill_val, ), z_rotate_degree

def random_translate(i, x_translate = None,  y_translate = None, translate_range = [-10,10]):
    # only do translate according to x and y
    if x_translate is None or y_translate is None:
        x_translate = int(random.uniform(translate_range[0], translate_range[1]))
        y_translate = int(random.uniform(translate_range[0], translate_range[1]))

    return Data_processing.translate_image(np.copy(i), [x_translate,y_translate,0]), x_translate,y_translate

def sample_consecutive_slices(image, N):
    _, _, D = image.shape  # Get the depth (number of slices)
    
    # Ensure N does not exceed the number of slices
    if N > D:
        raise ValueError("N cannot be greater than the number of slices in the image.")
    
    # Randomly select the starting index for N slices
    start_idx = np.random.randint(0, D - N + 1)
    
    # # Extract the consecutive slices
    # sampled_slices = image[:, :, start_idx:start_idx + N]
    
    return start_idx, start_idx + N
    

class DataGenerator_alltf(Sequence):

    def __init__(self,
        patient_class_list,
        patient_id_list,
        tf_list,
        which_timeframe_is_template = 'others', # 'others' means warp 0 to other time frames, '0' means warp other time frames to 0

        patient_num = None, 
        batch_size = None, 

        main_path = '/mnt/camca_NAS/4DCT',
        shuffle = False,
        normalize = None,
        adapt_shape = [160,160,96],
        augment = False,
        augment_frequency = None,
        seed = 10):

        self.patient_class_list = patient_class_list
        self.patient_id_list = patient_id_list
        self.tf_list = tf_list
        self.which_timeframe_is_template = which_timeframe_is_template

        self.patient_num = patient_num
        self.batch_size = batch_size

        self.main_path = main_path
        self.shuffle = shuffle
        self.normalize = normalize
        self.adapt_shape = adapt_shape
        self.augment = augment
        self.augment_frequency = augment_frequency
        self.seed = seed

        self.on_epoch_end()
        
    def __len__(self):
        # summation of self.tf_list
        summation = sum(self.tf_list)
        # print('summation tf_list: ', summation, 'len calculates as: ', summation // self.batch_size)
        return summation // self.batch_size 

    def on_epoch_end(self):
        
        self.seed += 1

        index_array = []
        if self.shuffle == True:
            case_list = np.random.permutation(self.patient_num)
        else:
            case_list = np.arange(self.patient_num)

        for case in case_list:
            if self.shuffle == True:
                t_list = np.random.permutation(self.tf_list[case])
            else:
                t_list = np.arange(self.tf_list[case])
            for t in t_list:
                index_array.append([case, t]) 
        self.index_array = np.array(index_array)

    def __getitem__(self,index):
        'Generate indexes of the batch'
        total_num = sum(self.tf_list)
        current_index = (index * self.batch_size) % total_num
        if total_num > current_index + self.batch_size:   # the total number of cases is adequate for next loop
            current_batch_size = self.batch_size
        else:
            current_batch_size = total_num - current_index  # approaching to the tend, not adequate, should reduce the batch size

        indexes = self.index_array[current_index : current_index + current_batch_size]

        # set memory 
        batch_moving_image = np.zeros(tuple([current_batch_size]) + tuple([self.adapt_shape[0],self.adapt_shape[1], self.adapt_shape[2]]) + tuple([1]))
        batch_fixed_image = np.zeros(tuple([current_batch_size]) + tuple([self.adapt_shape[0],self.adapt_shape[1], self.adapt_shape[2]]) + tuple([1]))
        batch_zero_phi = np.zeros(tuple([current_batch_size]) + tuple([self.adapt_shape[0],self.adapt_shape[1], self.adapt_shape[2]]) + tuple([len(self.adapt_shape)]))

        for i,j in enumerate(indexes):
            # get patient class and patient id
            patient_class = self.patient_class_list[j[0]]
            patient_id = self.patient_id_list[j[0]]
            path = os.path.join(self.main_path,'example_data/nii-images' ,patient_class, patient_id,'img-nii-resampled-1.5mm')

            # find out all files
            files = ff.sort_timeframe(ff.find_all_target_files(['*.nii.gz'],path),2)
            random_integer = j[1]

            if self.which_timeframe_is_template == '0':
                two_tf = [random_integer, 0]
            else:
                two_tf = [0, random_integer]

            # load two timeframes
            tf1 = nb.load(files[two_tf[0]]).get_fdata()
            tf2 = nb.load(files[two_tf[1]]).get_fdata()
            if  len(tf1.shape) == 4:
                tf1 = tf1[:,:,:,0]
            if len(tf2.shape) == 4:
                tf2 = tf2[:,:,:,0]

            # adapt shape
            if self.adapt_shape != None:
                tf1 = Data_processing.crop_or_pad(tf1, self.adapt_shape, value = np.min(tf1))
                tf2 = Data_processing.crop_or_pad(tf2, self.adapt_shape, value = np.min(tf2))
            
            # normalize
            if self.normalize == True:
                tf1 = Data_processing.normalize_image(tf1, normalize_factor=1000)
                tf2 = Data_processing.normalize_image(tf2, normalize_factor = 1000)

            # augment
            if self.augment == True:
                if random.uniform(0,1) < self.augment_frequency:
                    tf1, z_rotate_degree = random_rotate(tf1,  z_rotate_range = [-10,10], order = 2)
                    tf1, x_translate, y_translate = random_translate(tf1, translate_range = [-10,10])
                    tf2, _ = random_rotate(tf2, z_rotate_degree = z_rotate_degree, order = 2)
                    tf2, _, _ = random_translate(tf2, x_translate = x_translate, y_translate = y_translate)
                    # print('z_rotate_degree: ', z_rotate_degree, ' x_translate: ', x_translate, ' y_translate: ', y_translate)
            
            tf1 = np.expand_dims(tf1, axis = -1); tf2 = np.expand_dims(tf2, axis = -1)

            batch_moving_image[i] = tf1
            batch_fixed_image[i] = tf2
        

        inputs = [batch_moving_image, batch_fixed_image]
        outputs = [batch_fixed_image, batch_zero_phi]

        return (inputs, outputs)