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
import Cardiac4DCT_Synth_Diffusion.Data_processing as Data_processing
import Cardiac4DCT_Synth_Diffusion.functions_collection as ff
from ema_pytorch import EMA

def get_info(patient_id, timeframe_info, how_many_timeframes_together):
    row = timeframe_info[timeframe_info['patient_id'] == patient_id]
    if row.shape[0] == 0:
        return None, 0,None,None, None, None, None
    tf_num = row['total_tf_num'].iloc[0]
    es_index = row['es_index'].iloc[0]
    sampled_time_frame_list = ast.literal_eval(row['sampled_time_frame_list'].iloc[0])
    normalized_time_frame_list = ast.literal_eval(row['normalized_time_frame_list_copy'].iloc[0])
    assert len(sampled_time_frame_list) == len(normalized_time_frame_list)
    if how_many_timeframes_together == 5:
        EF = round(row['EF_sampled_in_5tf_by_mvf'].iloc[0],2)
    elif how_many_timeframes_together == 10:
        EF = round(row['EF_sampled_in_10tf_by_mvf'].iloc[0],2)
    return row, tf_num, es_index, sampled_time_frame_list, normalized_time_frame_list, EF

class Dataset_dual_3D(Dataset):
    def __init__(
        self, 

        patient_class_list,
        patient_id_list,
        main_path = '/mnt/camca_NAS/4DCT',
        timeframe_info = None, 

        how_many_timeframes_together = 10,
        
        mvf_size_3D = [40,40,24],
        slice_range = [0,96],
        picked_tf = 'random', #'random' or specific tf or 'ES'
        preset_EF = None,
        condition_on_image = False,
        prepare_seg = False,
        mvf_cutoff = [-20,20],
       
        normalize_factor = 'equation',
        shuffle = False,
        augment = False,
        augment_frequency = None,
        augment_range = [[-10,10], [-10,10]], # translate, rotation
        augment_pre_done = True,
        augment_aug_index = [1,2],
    ):
        super().__init__()
        self.patient_class_list = patient_class_list
        self.patient_id_list = patient_id_list
        self.main_path = main_path
        self.timeframe_info = timeframe_info
        self.how_many_timeframes_together = how_many_timeframes_together

        self.mvf_size_3D = mvf_size_3D
        self.slice_range = slice_range
        self.picked_tf = picked_tf
        self.preset_EF = preset_EF
        self.condition_on_image = condition_on_image
        self.prepare_seg = prepare_seg
        self.normalize_factor = normalize_factor
        self.mvf_cutoff = mvf_cutoff

        self.shuffle = shuffle
        self.augment = augment
        self.augment_frequency = augment_frequency
        self.augment_range = augment_range
        self.augment_pre_done = augment_pre_done
        self.augment_aug_index = augment_aug_index
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
       
        # cutoff and normalize mvf
        mvf = Data_processing.cutoff_intensity(mvf, cutoff_low = self.mvf_cutoff[0], cutoff_high = self.mvf_cutoff[1])
        mvf = Data_processing.normalize_image(mvf, normalize_factor = self.normalize_factor, image_max = self.mvf_cutoff[1], image_min = self.mvf_cutoff[0], invert = False)
        return mvf
        
    def __len__(self):
        return self.num_files  

    def __getitem__(self, index):
        f = self.index_array[index]
        patient_class = self.patient_class_list[f]
        patient_id = self.patient_id_list[f]

        # load the timeframe info
        row, tf_num, es_index,  sampled_time_frame_list, normalized_time_frame_list, EF = get_info(patient_id, self.timeframe_info, self.how_many_timeframes_together)
        if self.preset_EF != None:
            EF = round(self.preset_EF,2)

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
                aug_parameter = np.load(os.path.join(self.main_path, 'example_data/mvf_aug',patient_class, patient_id, 'aug_'+str(aug_index), 'aug_parameter.npy'))
                z_rotate_degree, x_translate, y_translate = aug_parameter[0], int(aug_parameter[1]), int(aug_parameter[2])
  
        # picked timeframe
        picked_tf_normalized = self.picked_tf
        picked_tf = [sampled_time_frame_list[normalized_time_frame_list.index(picked_tf_normalized[iii])] for iii in range(0,self.how_many_timeframes_together)]
        condition_tf_data = np.asarray(picked_tf); condition_tf_normalized_data = np.asarray(picked_tf_normalized)
        
        # load data 
        if on_the_fly == False:   # when we have prepared augmentation beforehand 
            mvf = np.zeros([self.how_many_timeframes_together, 3, self.mvf_size_3D[0], self.mvf_size_3D[1], self.mvf_size_3D[2]])
            for iii in range(0, self.how_many_timeframes_together):
                mvf_filename = os.path.join(self.main_path, 'example_data/mvf_aug',patient_class, patient_id, 'aug_'+str(aug_index), 'mvf_downsampled', str(picked_tf[iii])+'.nii.gz')
                a = nb.load(mvf_filename).get_fdata()
                mvf[iii,:,:,:,:] = np.transpose(a, (3,0,1,2))
            mvf = Data_processing.cutoff_intensity(mvf, cutoff_low = self.mvf_cutoff[0], cutoff_high = self.mvf_cutoff[1])
            mvf = Data_processing.normalize_image(mvf, normalize_factor = self.normalize_factor, image_max = self.mvf_cutoff[1], image_min = self.mvf_cutoff[0], invert = False)
            mvf = np.reshape(mvf,(self.how_many_timeframes_together*3, self.mvf_size_3D[0], self.mvf_size_3D[1], self.mvf_size_3D[2]))
            final_data = np.copy(mvf)
                
    
        else: # on the fly augmentation
            mvf = np.zeros([self.how_many_timeframes_together, self.mvf_size_3D[0], self.mvf_size_3D[1], self.mvf_size_3D[2], 3])
            for iii in range(0,self.how_many_timeframes_together):
                mvf_filename = os.path.join(self.main_path,'example_data/mvf_aug',patient_class, patient_id, 'aug_0/mvf', str(picked_tf[iii])+'.nii.gz')
                mvf[iii,:,:,:,:] = self.load_mvf(mvf_filename, x_translate = x_translate, y_translate = y_translate, z_rotate_degree = z_rotate_degree, do_augment = do_augment)
      
            final_data = np.transpose(np.copy(mvf), (0,4,1,2,3))
            final_data = np.reshape(final_data,(self.how_many_timeframes_together*3,self.mvf_size_3D[0], self.mvf_size_3D[1], self.mvf_size_3D[2]))
         
        # condition on image
        if self.condition_on_image == True:
            if on_the_fly == False:
                condition_img_file = os.path.join(self.main_path, 'example_data/mvf_aug',patient_class, patient_id, 'aug_'+str(aug_index), 'condition_img/0.nii.gz')
                condition_image_data = Data_processing.cutoff_intensity(nb.load(condition_img_file).get_fdata(), cutoff_low = -500, cutoff_high = 1000)
                condition_image_data = Data_processing.normalize_image(condition_image_data, normalize_factor = 'equation', image_max = 1000, image_min = -500,invert = False)
            else:
                image_path = os.path.join(self.main_path, 'example_data/nii-imagse',patient_class, patient_id, 'img-nii-resampled-1.5mm/0.nii.gz')
                condition_image_data = nb.load(image_path).get_fdata()
                if len(condition_image_data.shape) == 4:
                    condition_image_data = condition_image_data[:,:,:,0]
                condition_image_data = Data_processing.crop_or_pad(condition_image_data, [160,160,96], value = np.min(condition_image_data))
                if do_augment == True:
                    condition_image_data = Data_processing.random_move(condition_image_data,x_translate,y_translate,z_rotate_degree,do_augment = do_augment, fill_val = np.min(condition_image_data))
        
                condition_image_data = block_reduce(condition_image_data, (160//self.mvf_size_3D[0],160//self.mvf_size_3D[1], 96//self.mvf_size_3D[2]), func=np.mean)
                condition_image_data = Data_processing.cutoff_intensity(condition_image_data, cutoff_low = -500, cutoff_high = 1000)
                condition_image_data = Data_processing.normalize_image(condition_image_data, normalize_factor = 'equation', image_max = 1000, image_min = -500,invert = False)
        else:
            condition_image_data = np.zeros(self.mvf_size_3D)

        # need segmentation as well
        if self.prepare_seg == True:
            if on_the_fly == False:
                condition_seg_file = os.path.join(self.main_path,'example_data/mvf_aug',patient_class, patient_id, 'aug_'+str(aug_index), 'segmentation/0.nii.gz')
                condition_seg_data = np.round(nb.load(condition_seg_file).get_fdata()).astype(np.float32)

                condition_seg_ori_res = os.path.join(self.main_path,'example_data/mvf_aug',patient_class, patient_id, 'aug_'+str(aug_index), 'segmentation_original_res/0.nii.gz')
                condition_seg_ori_res_data = np.round(nb.load(condition_seg_ori_res).get_fdata()).astype(np.float32)    
            else:
                ValueError('please prepare segmentation data beforehand')
        else:
            condition_seg_data =  np.zeros(self.mvf_size_3D)
            condition_seg_ori_res_data = np.zeros((160,160,96))
        
        # condition on EF
        condition_EF_data = np.asarray([EF])

        x0_image_data = torch.from_numpy(final_data).float()
        condition_tf_data = torch.from_numpy(condition_tf_data).float()
        condition_tf_normalized_data = torch.from_numpy(condition_tf_normalized_data).float()
        condition_image_data = torch.from_numpy(condition_image_data).unsqueeze(0).float()
        condition_seg_data = torch.from_numpy(condition_seg_data).unsqueeze(0).float()
        condition_seg_ori_res_data = torch.from_numpy(condition_seg_ori_res_data).unsqueeze(0).float()
        condition_EF_data = torch.from_numpy(condition_EF_data).float()

        # print('in generator, x0 image data shape: ', x0_image_data.shape,  ' and condition image data shape: ', condition_image_data.shape, ' and condition EF data shape: ', condition_EF_data.shape, ' and condition segmentation data shape: ', condition_seg_data.shape)
        
        return x0_image_data, condition_tf_data, condition_tf_normalized_data, condition_image_data, condition_EF_data, condition_seg_data, condition_seg_ori_res_data
    
    def on_epoch_end(self):
        print('now run on_epoch_end function')
        self.index_array = self.generate_index_array()

