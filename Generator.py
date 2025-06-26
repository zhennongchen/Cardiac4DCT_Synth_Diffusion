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

class VAE_process(object):
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = VAE_model.VAE(
            in_channels= 3, 
            out_channels= 3, 
            spatial_dims = 3,
            emb_channels = 3,
            hid_chs =   [64,128,256], 
            kernel_sizes=[3,3,3], 
            strides =    [1,2,2],)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ema = EMA(self.model)
        self.ema.to(self.device)  
    
    def load_model(self):
        data = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(data['model'])
        self.ema.load_state_dict(data["ema"])

    def VAE_encode(self,x):
        self.load_model()
        x = np.moveaxis(x, -1, 0)
        x = x[np.newaxis,...] # add batch axis
        x0_image_data = torch.from_numpy(x).float()
        pred_latent = self.ema.ema_model.encode(x0_image_data.to(self.device))
        pred_latent = np.copy(pred_latent.detach().cpu().numpy().squeeze())
        return pred_latent
    
def get_info(patient_id, timeframe_info, how_many_timeframes_together):
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

class Dataset_dual_3D(Dataset):
    def __init__(
        self, 
        VAE_process, # False or VAE_process object
        
        patient_class_list,
        patient_id_list,
        mvf_folder,

        how_many_timeframes_together,

        mvf_size_3D = [160,160,96],
        latent_size_3D = [40,40,24],
        slice_range = [0,96],
        picked_tf = 'random', #'random' or specific tf or 'ES'
        preset_EF = None,
        condition_on_image = False,
        condition_on_seg = False,
        mvf_cutoff = [-20,20],
        latent_cutoff = [-30,30],
        normalize_factor = 'equation',
        shuffle = False,
        augment = False,
        augment_frequency = None,
        augment_range = [[-10,10], [-10,10]], # translate, rotation
        augment_pre_done = False,
        augment_aug_index = [1,5],
    ):
        super().__init__()
        self.VAE_process = VAE_process
        self.patient_class_list = patient_class_list
        self.patient_id_list = patient_id_list
        self.mvf_folder = mvf_folder
        self.how_many_timeframes_together = how_many_timeframes_together

        self.mvf_size_3D = mvf_size_3D
        self.latent_size_3D = latent_size_3D
        self.slice_range = slice_range
        self.picked_tf = picked_tf
        self.preset_EF = preset_EF
        self.condition_on_image = condition_on_image
        self.condition_on_seg = condition_on_seg
        self.normalize_factor = normalize_factor
        self.mvf_cutoff = mvf_cutoff
        self.latent_cutoff = latent_cutoff

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
        if self.preset_EF != None:
            EF = round(self.preset_EF,2)

        ### decide whether to do augmentation
        # on_the_fly = False if (self.augment_pre_done == True and self.VAE_process !=False) else True
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
        # print('on the fly:', on_the_fly, ' do augment:', do_augment)
        # print('aug index:', aug_index, ' z_rotate_degree:', z_rotate_degree, ' x_translate:', x_translate, ' y_translate:', y_translate)
           
        # picked timeframe
        if self.how_many_timeframes_together == 1:
            if self.picked_tf == 'random':
                random_index = random.randint(0, len(sampled_time_frame_list)- 1)
                picked_tf = sampled_time_frame_list[random_index]
                picked_tf_normalized = normalized_time_frame_list[random_index]
            elif self.picked_tf == 'ES':
                picked_tf = es_index
                picked_tf_normalized = es_index_normalized
            # elif isinstance(self.picked_tf, int):
            #     picked_tf = self.picked_tf
            #     picked_tf_normalized = normalized_time_frame_list[sampled_time_frame_list.index(picked_tf)]
            elif isinstance(self.picked_tf, float) and self.picked_tf < 1:
                picked_tf_normalized = self.picked_tf
                picked_tf = sampled_time_frame_list[normalized_time_frame_list.index(picked_tf_normalized)]
            condition_tf_data = np.asarray([picked_tf])
            condition_tf_normalized_data = np.asarray([picked_tf_normalized])
        elif self.how_many_timeframes_together > 1:
            assert len(self.picked_tf) == self.how_many_timeframes_together and isinstance(self.picked_tf[0], float) # only normalized timeframes
            picked_tf_normalized = self.picked_tf
            picked_tf = [sampled_time_frame_list[normalized_time_frame_list.index(picked_tf_normalized[iii])] for iii in range(0,self.how_many_timeframes_together)]
            condition_tf_data = np.asarray(picked_tf); condition_tf_normalized_data = np.asarray(picked_tf_normalized)
        # print('picked timeframes: ', picked_tf, ' picked timeframes normalized: ', picked_tf_normalized)
        
        # load data 
        if on_the_fly == False:   # when we have prepared augmentation beforehand 
            # for simple down-sampling
            if self.VAE_process == False:
                mvf = np.zeros([self.how_many_timeframes_together, 3, self.mvf_size_3D[0], self.mvf_size_3D[1], self.mvf_size_3D[2]])
                for iii in range(0, self.how_many_timeframes_together):
                    mvf_filename = os.path.join('/workspace/Documents/Data/mvf_aug',patient_class, patient_id, 'aug_'+str(aug_index), 'mvf_downsampled', str(picked_tf[iii])+'.nii.gz')
                    # print('load prepared mvf file:', mvf_filename) 
                    a = nb.load(mvf_filename).get_fdata()
                    mvf[iii,:,:,:,:] = np.transpose(a, (3,0,1,2))
                mvf = Data_processing.cutoff_intensity(mvf, cutoff_low = self.mvf_cutoff[0], cutoff_high = self.mvf_cutoff[1])
                # print('mvf max and min:', np.max(mvf), np.min(mvf))
                mvf = Data_processing.normalize_image(mvf, normalize_factor = self.normalize_factor, image_max = self.mvf_cutoff[1], image_min = self.mvf_cutoff[0], invert = False)
                mvf = np.reshape(mvf,(self.how_many_timeframes_together*3, self.mvf_size_3D[0], self.mvf_size_3D[1], self.mvf_size_3D[2]))
                final_data = np.copy(mvf)
                
            # for latent
            if self.VAE_process != False:
                latent = np.zeros([self.how_many_timeframes_together, 3, self.latent_size_3D[0], self.latent_size_3D[1], self.latent_size_3D[2]])
                for iii in range(0, self.how_many_timeframes_together):
                    latent_filename = os.path.join('/workspace/Documents/Data/mvf_aug',patient_class, patient_id, 'aug_'+str(aug_index), 'latent', str(picked_tf[iii])+'.nii.gz')
                    # print('load prepared latent file:', latent_filename) 
                    latent[iii,:,:,:,:] = np.transpose(nb.load(latent_filename).get_fdata(), (3,0,1,2))
                latent = Data_processing.cutoff_intensity(latent, cutoff_low = self.latent_cutoff[0], cutoff_high = self.latent_cutoff[1])
                latent = Data_processing.normalize_image(latent, normalize_factor = self.normalize_factor, image_max = self.latent_cutoff[1], image_min = self.latent_cutoff[0], invert = False)
                latent = np.reshape(latent,(self.how_many_timeframes_together*3, self.latent_size_3D[0], self.latent_size_3D[1], self.latent_size_3D[2]))
                final_data = np.copy(latent)
        

        else: # on the fly augmentation
            mvf = np.zeros([self.how_many_timeframes_together, self.mvf_size_3D[0], self.mvf_size_3D[1], self.mvf_size_3D[2], 3])
            for iii in range(0,self.how_many_timeframes_together):
                mvf_filename = os.path.join('/workspace/Documents/Data/mvf_aug',patient_class, patient_id, 'aug_0/mvf', str(picked_tf[iii])+'.nii.gz')
                mvf[iii,:,:,:,:] = self.load_mvf(mvf_filename, x_translate = x_translate, y_translate = y_translate, z_rotate_degree = z_rotate_degree, do_augment = do_augment)
            if self.VAE_process == False:
                final_data = np.transpose(np.copy(mvf), (0,4,1,2,3))
                final_data = np.reshape(final_data,(self.how_many_timeframes_together*3,self.mvf_size_3D[0], self.mvf_size_3D[1], self.mvf_size_3D[2]))
            else:
                latent = np.zeros([self.how_many_timeframes_together, 3, self.latent_size_3D[0], self.latent_size_3D[1], self.latent_size_3D[2]])
                for iii in range(0,self.how_many_timeframes_together):
                    latent[iii,:,:,:,:] = self.VAE_process.VAE_encode(mvf[iii,:,:,:,:])
                latent = Data_processing.cutoff_intensity(latent, cutoff_low = self.latent_cutoff[0], cutoff_high = self.latent_cutoff[1])
                latent = Data_processing.normalize_image(latent, normalize_factor = self.normalize_factor, image_max = self.latent_cutoff[1], image_min = self.latent_cutoff[0], invert = False)
                latent = np.reshape(latent,(self.how_many_timeframes_together*3, self.latent_size_3D[0], self.latent_size_3D[1], self.latent_size_3D[2]))
                final_data = np.copy(latent)

        # condition on image
        if self.condition_on_image == True:
            if on_the_fly == False:
                condition_img_file = os.path.join('/workspace/Documents/Data/mvf_aug',patient_class, patient_id, 'aug_'+str(aug_index), 'condition_img/0.nii.gz')
                # print('load prepared condition image file:', condition_img_file)
                condition_image_data = Data_processing.cutoff_intensity(nb.load(condition_img_file).get_fdata(), cutoff_low = -500, cutoff_high = 1000)
                condition_image_data = Data_processing.normalize_image(condition_image_data, normalize_factor = 'equation', image_max = 1000, image_min = -500,invert = False)
            else:
                image_path = os.path.join('/workspace/Documents/Data/mvf',patient_class, patient_id, 'img-nii-resampled-1.5mm/0.nii.gz')
                condition_image_data = nb.load(image_path).get_fdata()
                if len(condition_image_data.shape) == 4:
                    condition_image_data = condition_image_data[:,:,:,0]
                condition_image_data = Data_processing.crop_or_pad(condition_image_data, [160,160,96], value = np.min(condition_image_data))
                # do augmentation on original image  then down-sample
                if do_augment == True:
                    condition_image_data = Data_processing.random_move(condition_image_data,x_translate,y_translate,z_rotate_degree,do_augment = do_augment, fill_val = np.min(condition_image_data))
                
                if self.VAE_process != False:
                    condition_image_data = block_reduce(condition_image_data, (160//self.latent_size_3D[0],160//self.latent_size_3D[1], 96//self.latent_size_3D[2]), func=np.mean)
                else:
                    condition_image_data = block_reduce(condition_image_data, (160//self.mvf_size_3D[0],160//self.mvf_size_3D[1], 96//self.mvf_size_3D[2]), func=np.mean)

                condition_image_data = Data_processing.cutoff_intensity(condition_image_data, cutoff_low = -500, cutoff_high = 1000)
                condition_image_data = Data_processing.normalize_image(condition_image_data, normalize_factor = 'equation', image_max = 1000, image_min = -500,invert = False)
        else:
            condition_image_data = np.zeros(self.latent_size_3D) if self.VAE_process != False else np.zeros(self.mvf_size_3D)

        # condition on segmentation as well
        if self.condition_on_seg == True:
            if on_the_fly == False:
                condition_seg_file = os.path.join('/workspace/Documents/Data/mvf_aug',patient_class, patient_id, 'aug_'+str(aug_index), 'segmentation/0.nii.gz')
                condition_seg_data = np.round(nb.load(condition_seg_file).get_fdata()).astype(np.float32)

                condition_seg_ori_res = os.path.join('/workspace/Documents/Data/mvf_aug',patient_class, patient_id, 'aug_'+str(aug_index), 'segmentation_original_res/0.nii.gz')
                condition_seg_ori_res_data = np.round(nb.load(condition_seg_ori_res).get_fdata()).astype(np.float32)    
            else:
                ValueError('please prepare segmentation data beforehand')
        else:
            condition_seg_data = np.zeros(self.latent_size_3D) if self.VAE_process != False else np.zeros(self.mvf_size_3D)
            condition_seg_ori_res_data = np.zeros((160,160,96))
        
        # condition on EF
        condition_EF_data = np.asarray([EF])
        # print('condition on EF: ', condition_EF_data)

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

# # load all data
# def load_all_mvf_data(patient_class_list, patient_id_list, data_folder, image_size_3D):
#     train_dataset = np.zeros([len(patient_id_list),image_size_3D[0], image_size_3D[1], image_size_3D[2], 3]) 
#     for i in range(0,len(patient_id_list)):
#         path = os.path.join(data_folder,patient_class_list[i], patient_id_list[i], 'voxel_final')
#         # all files
#         files = ff.find_all_target_files(['*.nii.gz'],path)
#         final_files = np.copy(files)
#         for f in files:
#             if 'moved' in f or 'original' in f:
#                 # remove it from the numpy array
#                 final_files = np.delete(final_files, np.where(final_files == f))
#         files = ff.sort_timeframe(final_files,2)

#         # only ES
#         patient_id = patient_id_list[i]
#         timeframe_info = pd.read_excel('/mnt/camca_NAS/4DCT/Patient_lists/patient_list_final_selection_timeframes.xlsx')
#         row = timeframe_info[timeframe_info['patient_id'] == patient_id]
#         es_index = row['es_index'].iloc[0]
#         picked_tf = es_index

#         img = nb.load(files[picked_tf]).get_fdata()
#         train_dataset[i,:,:,:,:] = img
#     return train_dataset

# def load_all_img_data(patient_class_list, patient_id_list, data_folder, image_size_3D):
#     train_dataset = np.zeros([len(patient_id_list),image_size_3D[0], image_size_3D[1], image_size_3D[2]]) 
#     for i in range(0,len(patient_id_list)):
#         path = os.path.join(data_folder,patient_class_list[i], patient_id_list[i], 'img-nii-resampled-1.5mm/0.nii.gz')
#         img = nb.load(path).get_fdata()
#         if len(img.shape) == 4:
#             img = img[:,:,:,0]
#         img = Data_processing.crop_or_pad(img, [160,160,96], value = np.min(img))
#         train_dataset[i,:,:,:] = img
#     return train_dataset


