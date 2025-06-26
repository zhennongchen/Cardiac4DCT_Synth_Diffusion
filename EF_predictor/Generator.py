# dataset classes

import os
import numpy as np
import nibabel as nb
import random
import ast
import pandas as pd
from scipy import ndimage
from skimage.measure import block_reduce

import torch
from torch.utils.data import Dataset
import Diffusion_motion_field.Data_processing as Data_processing
import Diffusion_motion_field.functions_collection as ff
import Diffusion_motion_field.latent_diffusion.VAE_model as VAE_model
from ema_pytorch import EMA

timeframe_info1 = pd.read_excel('/mnt/camca_NAS/4DCT/Patient_lists/uc/patient_list_final_selection_timeframes.xlsx')
timeframe_info2 = pd.read_excel('/mnt/camca_NAS/4DCT/Patient_lists/mgh/patient_list_final_selection_timeframes.xlsx')

    
def get_info(patient_id, timeframe_info, how_many_timeframes_together = 10):
    row = timeframe_info[timeframe_info['patient_id'] == patient_id]
    if row.shape[0] == 0:
        return None, 0,None,None, None, None, None
    tf_num = row['total_tf_num'].iloc[0]
    es_index = row['es_index'].iloc[0]
    es_index_normalized = row['es_index_normalized'].iloc[0]
    sampled_time_frame_list = ast.literal_eval(row['sampled_time_frame_list'].iloc[0])
    normalized_time_frame_list = ast.literal_eval(row['normalized_time_frame_list_copy'].iloc[0])
    assert len(sampled_time_frame_list) == len(normalized_time_frame_list)
    if how_many_timeframes_together == 5:
        EF = round(row['EF_sampled_in_5tf_by_mvf'].iloc[0],2)
    elif how_many_timeframes_together == 10:
        EF = round(row['EF_sampled_in_10tf_by_mvf'].iloc[0],2)
    return row, tf_num, es_index, es_index_normalized, sampled_time_frame_list, normalized_time_frame_list, EF

class Dataset_MVF(Dataset):
    def __init__(
        self, 
        
        patient_class_list,
        patient_id_list,
        mvf_folder,

        picked_tf = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        mvf_size_3D = [40,40,24],
        slice_range = [0,96],

        mvf_cutoff = [-20,20],
        normalize_factor = 'equation',
        shuffle = False,
        augment = False,
        augment_frequency = None,
        noise_add_frequency = 0,
        augment_range = [[-10,10], [-10,10]], # translate, rotation
        augment_pre_done = False,
        augment_aug_index = [1,5],
    ):
        super().__init__()
     
        self.patient_class_list = patient_class_list
        self.patient_id_list = patient_id_list
        self.mvf_folder = mvf_folder
        self.how_many_timeframes_together = 10

        self.picked_tf = picked_tf
        self.mvf_size_3D = mvf_size_3D
        self.slice_range = slice_range
        self.normalize_factor = normalize_factor
        self.mvf_cutoff = mvf_cutoff

        self.shuffle = shuffle
        self.augment = augment
        self.augment_frequency = augment_frequency
        self.augment_range = augment_range
        self.augment_pre_done = augment_pre_done
        self.augment_aug_index = augment_aug_index
        self.noise_add_frequency = noise_add_frequency
        self.num_files = len(patient_class_list)

        self.index_array = self.generate_index_array()

    def generate_index_array(self):
        np.random.seed()
        index_array = []
        
        if self.shuffle == True:
            f_list = np.random.permutation(self.num_files)
        else:
            f_list = np.arange(self.num_files)
        
        index_array = np.copy(f_list)

        # print('now after generate the index array, the index array is: ',index_array)
        return index_array
    
    def load_mvf(self, filename, do_augment = False, x_translate = 0, y_translate = 0, z_rotate_degree = 0 ):
        raw_mvf = nb.load(filename)
        raw_mvf = raw_mvf.get_fdata()

        # select slice range
        mvf = raw_mvf[:,:,self.slice_range[0]:self.slice_range[1],:]

        # do augmentation on mvf
        if do_augment == True:
            mvf = Data_processing.random_move(mvf,x_translate,y_translate,z_rotate_degree,fill_val = 0, do_augment = do_augment)
        
        # downsample our mvf
        if self.mvf_size_3D[0] != mvf.shape[0]:
            scale_factors = (mvf.shape[0]//self.mvf_size_3D[0],  mvf.shape[1]//self.mvf_size_3D[1], mvf.shape[2]//self.mvf_size_3D[2], 1)
            mvf = block_reduce(mvf, scale_factors, func=np.mean)
        
        if random.uniform(0,1) < self.noise_add_frequency:
            rand_std = random.uniform(0.005,0.2)
            noise = np.random.normal(0, rand_std, mvf.shape)
            mvf = mvf + noise
       
        # cutoff and normalize mvf
        mvf = Data_processing.cutoff_intensity(mvf, cutoff_low = self.mvf_cutoff[0], cutoff_high = self.mvf_cutoff[1])
        mvf = Data_processing.normalize_image(mvf, normalize_factor = self.normalize_factor, image_max = self.mvf_cutoff[1], image_min = self.mvf_cutoff[0], invert = False)
        return mvf
        
    def __len__(self):
        return self.num_files 

    def __getitem__(self, index):
        # print('in this geiitem, self.index_array is: ', self.index_array)
        f = self.index_array[index]
        # print('index is: ', index, ' now we pick file ', f)
        patient_class = self.patient_class_list[f]
        patient_id = self.patient_id_list[f]
        # print('patient class: ', patient_class, ' patient id: ', patient_id)

        # load the timeframe info
        row, tf_num, es_index, es_index_normalized, sampled_time_frame_list, normalized_time_frame_list, EF = get_info(patient_id, timeframe_info1, self.how_many_timeframes_together)
        if tf_num == 0:
            row, tf_num, es_index, es_index_normalized, sampled_time_frame_list, normalized_time_frame_list, EF = get_info(patient_id, timeframe_info2, self.how_many_timeframes_together)

        ### decide whether to do augmentation
        on_the_fly = False if (self.augment_pre_done == True) else True

        do_augment, aug_index, z_rotate_degree, x_translate, y_translate = False, 0, 0, 0, 0
        if self.augment == True and random.uniform(0,1) < self.augment_frequency:
            do_augment = True
            if on_the_fly == True:
                z_rotate_degree = random.uniform(self.augment_range[1][0],self.augment_range[1][1])
                x_translate = int(round((random.uniform(self.augment_range[0][0],self.augment_range[0][1]))))
                y_translate = int(round(random.uniform(self.augment_range[0][0],self.augment_range[0][1])))
            else:
                aug_index = int(round(random.uniform(self.augment_aug_index[0],self.augment_aug_index[1])))
                aug_parameter = np.load(os.path.join('/workspace/Documents/Data/mvf_aug',patient_class, patient_id, 'aug_'+str(aug_index), 'aug_parameter.npy'))
                z_rotate_degree, x_translate, y_translate = aug_parameter[0], int(aug_parameter[1]), int(aug_parameter[2])
    
        # picked timeframe
        picked_tf_normalized = self.picked_tf
        picked_tf = [sampled_time_frame_list[normalized_time_frame_list.index(picked_tf_normalized[iii])] for iii in range(0,self.how_many_timeframes_together)]
        condition_tf_data = np.asarray(picked_tf); condition_tf_normalized_data = np.asarray(picked_tf_normalized)
        # print('picked timeframes: ', picked_tf, ' picked timeframes normalized: ', picked_tf_normalized)
        
        # load data 
        if on_the_fly == False:   # when we have prepared augmentation beforehand 
            mvf = np.zeros([self.how_many_timeframes_together, 3, self.mvf_size_3D[0], self.mvf_size_3D[1], self.mvf_size_3D[2]])
            for iii in range(0, self.how_many_timeframes_together):
                mvf_filename = os.path.join('/workspace/Documents/Data/mvf_aug',patient_class, patient_id, 'aug_'+str(aug_index), 'mvf_downsampled', str(picked_tf[iii])+'.nii.gz')
                # print('load prepared mvf file:', mvf_filename) 
                affine = nb.load(mvf_filename).affine
                a = nb.load(mvf_filename).get_fdata()
                mvf[iii,:,:,:,:] = np.transpose(a, (3,0,1,2))

            # original_a = mvf[0,:,:,:,:];original_a = np.transpose(original_a, (1,2,3,0))
            # nb.save(nb.Nifti1Image(original_a, affine), '/mnt/camca_NAS/orginal.nii.gz')
            if random.uniform(0,1) < self.noise_add_frequency:
                # noise = np.random.normal(0, 0.005, mvf.shape)
                rand_std = random.uniform(0.005,0.2)
                noise = np.random.normal(0, rand_std, mvf.shape)
                mvf = mvf + noise
                # new_a = mvf[0,:,:,:,:];new_a = np.transpose(new_a, (1,2,3,0))
                # nb.save(nb.Nifti1Image(new_a, affine), '/mnt/camca_NAS/noise.nii.gz')
            mvf = Data_processing.cutoff_intensity(mvf, cutoff_low = self.mvf_cutoff[0], cutoff_high = self.mvf_cutoff[1])
            mvf = Data_processing.normalize_image(mvf, normalize_factor = self.normalize_factor, image_max = self.mvf_cutoff[1], image_min = self.mvf_cutoff[0], invert = False)
            mvf = np.reshape(mvf,(self.how_many_timeframes_together*3, self.mvf_size_3D[0], self.mvf_size_3D[1], self.mvf_size_3D[2]))
            final_data = np.copy(mvf)
                
        else: # on the fly augmentation
            mvf = np.zeros([self.how_many_timeframes_together, self.mvf_size_3D[0], self.mvf_size_3D[1], self.mvf_size_3D[2], 3])
            for iii in range(0,self.how_many_timeframes_together):
                mvf_filename = os.path.join('/workspace/Documents/Data/mvf_aug',patient_class, patient_id, 'aug_0/mvf', str(picked_tf[iii])+'.nii.gz')
                mvf[iii,:,:,:,:] = self.load_mvf(mvf_filename, x_translate = x_translate, y_translate = y_translate, z_rotate_degree = z_rotate_degree, do_augment = do_augment)
                final_data = np.transpose(np.copy(mvf), (0,4,1,2,3))
                final_data = np.reshape(final_data,(self.how_many_timeframes_together*3,self.mvf_size_3D[0], self.mvf_size_3D[1], self.mvf_size_3D[2]))
 
        
        # condition on EF
        EF_data = np.asarray([EF])
        # print('condition on EF: ', EF_data)

        x0_image_data = torch.from_numpy(final_data).float()
        condition_tf_data = torch.from_numpy(condition_tf_data).float()
        condition_tf_normalized_data = torch.from_numpy(condition_tf_normalized_data).float()
        EF_data = torch.from_numpy(EF_data).float()

        # print('in generator, x0 image data shape: ', x0_image_data.shape, ' and condition timeframes data shape: ', condition_tf_data.shape,  ' and  EF data shape: ', EF_data.shape)
        
        return x0_image_data, condition_tf_data, condition_tf_normalized_data,EF_data
    
    def on_epoch_end(self):
        print('now run on_epoch_end function')
        self.index_array = self.generate_index_array()
