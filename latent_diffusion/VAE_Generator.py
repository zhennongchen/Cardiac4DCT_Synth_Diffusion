# dataset classes

import os
import numpy as np
import nibabel as nb
import random
from scipy import ndimage
from skimage.measure import block_reduce

import torch
from torch.utils.data import Dataset
import Diffusion_motion_field.Data_processing as Data_processing
import Diffusion_motion_field.functions_collection as ff

# random function
def random_move(img,x_translate,y_translate, z_rotate_degree, do_augment = False):
    if do_augment == False:
        return img
    else:
        if img.ndim == 3:
            img_new = np.copy(img)
            img_new = Data_processing.rotate_image(img_new,[0,0,z_rotate_degree],3)
            img_new = Data_processing.translate_image(img_new, [x_translate,y_translate,0])
        
            return img_new

        elif img.ndim == 4:
            img_new = np.zeros(img.shape)
            for z in range(0,3):
                m = np.copy(img)[...,z]
                m = Data_processing.rotate_image(m,[0,0,z_rotate_degree],3)
                m = Data_processing.translate_image(m, [x_translate,y_translate,0])
                img_new[...,z] = m
            return img_new
        else:
            ValueError('the input image is not 3D or 4D, please check the input image')


class Dataset_dual_3D(Dataset):
    def __init__(
        self,
        patient_class_list,
        patient_id_list,
        picked_tf = 'random',
        mvf_folder = None,
        latent_space_folder = False,

        image_size_3D = None,
        slice_range = None,
        normalize_factor = 'equation',
        maximum_cutoff = 20,
        minimum_cutoff = -20,
        shuffle = None,
        augment = None,
        augment_frequency = None,
        pre_done_aug = False,
    ):
        super().__init__()
        self.patient_class_list = patient_class_list
        self.patient_id_list = patient_id_list
        self.picked_tf = picked_tf
        self.mvf_folder = mvf_folder
        self.latent_space_folder = latent_space_folder

        self.image_size_3D = image_size_3D
        self.slice_range = slice_range
        self.normalize_factor = normalize_factor
        self.maximum_cutoff = maximum_cutoff
        self.minimum_cutoff = minimum_cutoff

        self.shuffle = shuffle
        self.augment = augment
        self.augment_frequency = augment_frequency
        self.pre_done_aug = pre_done_aug
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
    
    def load_mvf(self, filename, do_augment = False, x_translate = 0, y_translate = 0, z_rotate_degree = 0):
        ii = nb.load(filename)
        affine = nb.load(filename).affine
        ii = ii.get_fdata()
        downsampled_ii = np.copy(ii)

        downsampled_ii = downsampled_ii[:,:,self.slice_range[0]:self.slice_range[1],:] 
    
        if do_augment == True:
            downsampled_ii = random_move(downsampled_ii,x_translate,y_translate,z_rotate_degree,do_augment = do_augment)
   
        downsampled_ii = Data_processing.cutoff_intensity(downsampled_ii, cutoff_low = self.minimum_cutoff, cutoff_high = self.maximum_cutoff)
        downsampled_ii = Data_processing.normalize_image(downsampled_ii, normalize_factor = self.normalize_factor, image_max = self.maximum_cutoff, image_min = self.minimum_cutoff ,invert = False)
        # print('filename: ', filename, 'augmentation: ', do_augment, 'max and min', np.max(downsampled_ii), np.min(downsampled_ii))
    
        return downsampled_ii
        
    def __len__(self):
        return self.num_files 

    def __getitem__(self, index):
        # print('in this geiitem, self.index_array is: ', self.index_array)
        f = self.index_array[index]
        # print('index is: ', index, ' now we pick file ', f)
        patient_class = self.patient_class_list[f]
        patient_id = self.patient_id_list[f]

        path = os.path.join(self.mvf_folder ,patient_class, patient_id,'voxel_final')
        # print('patient class: ', patient_class, 'patient id: ', patient_id, ' path: ', path)

        # find out all files and time frames
        files = ff.find_all_target_files(['*.nii.gz'],path)
        final_files = np.copy(files)
        for f in files:
            if 'moved' in f or 'original' in f:
                # remove it from the numpy array
                final_files = np.delete(final_files, np.where(final_files == f))
        files = ff.sort_timeframe(final_files,2)
        tf_num = len(files)
        # print('total tf_num is :', tf_num)

        ### do augmentation
        do_augment = False; x_translate = 0; y_translate = 0; z_rotate_degree = 0
        if self.augment == True and random.uniform(0,1) < self.augment_frequency:
            z_rotate_degree = int(random.uniform(-5,5))
            x_translate = int(random.uniform(-8,8))
            y_translate = int(random.uniform(-8,8))
            do_augment = True
            # print('ok we need to do augmentation!!!!!')
        aug_index = int(round(random.uniform(1,5))) if do_augment == True else 0
        
        if self.picked_tf == 'random':
            picked_tf = random.randint(0, len(files)- 1)
        else:
            picked_tf = self.picked_tf
        # print('picked timeframes: ', picked_tf, 'aug_index: ', aug_index)
        x0_filename = files[picked_tf]
        if self.pre_done_aug == False:
            x0_data = self.load_mvf(x0_filename, do_augment = do_augment, x_translate = x_translate, y_translate = y_translate, z_rotate_degree = z_rotate_degree)
        else:
            x0_filename = os.path.join('/mnt/camca_NAS/4DCT/mvf_aug/',patient_class,patient_id,'aug_'+str(aug_index),'mvf',str(picked_tf)+'.nii.gz')
            x0_data = self.load_mvf(x0_filename, do_augment = False, x_translate = 0, y_translate = 0, z_rotate_degree = 0)
        
        final_data = np.copy(x0_data)
        final_data = np.moveaxis(final_data, -1, 0)
        # print('final data shape: ', final_data.shape, ' final data max and min: ', np.max(final_data), np.min(final_data))
        
        x0_image_data = torch.from_numpy(final_data).float()

        if self.latent_space_folder == False:
            return x0_image_data, picked_tf
        else:
            latent_path = os.path.join(self.latent_space_folder,'pred_latent_tf'+str(picked_tf)+'.nii.gz')
            latent_space = nb.load(latent_path).get_fdata()
            latent_space = np.moveaxis(latent_space, -1, 0)
            # print('latent space shape: ', latent_space.shape)
            latent_image_data = torch.from_numpy(latent_space).float()
            return x0_image_data,  picked_tf, latent_image_data
            

    
    def on_epoch_end(self):
        print('now run on_epoch_end function')
        self.index_array = self.generate_index_array()
    