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

def random_rotate(i, z_rotate_degree = None, z_rotate_range = [0,0], fill_val = None, order = 0):
    # only do rotate according to z (in-plane rotation)
    if z_rotate_degree is None:
        z_rotate_degree = round(random.uniform(z_rotate_range[0], z_rotate_range[1]),3)

    if fill_val is None:
        fill_val = np.min(i) 
    
    if z_rotate_degree == 0:
        return i, z_rotate_degree
    else:
        if i.ndim == 3:
            return Data_processing.rotate_image(np.copy(i), [0,0,z_rotate_degree], order = order, fill_val = fill_val, ), z_rotate_degree
        if i.ndim == 2:
            return Data_processing.rotate_image(np.copy(i), z_rotate_degree, order = order, fill_val = fill_val, ), z_rotate_degree


def random_translate(i, x_translate = None,  y_translate = None, translate_range = [-10,10]):
    # only do translate according to x and y
    if x_translate is None or y_translate is None:
        x_translate = int(random.uniform(translate_range[0], translate_range[1]))
        y_translate = int(random.uniform(translate_range[0], translate_range[1]))
    if i.ndim == 3:
        return Data_processing.translate_image(np.copy(i), [x_translate,y_translate,0]), x_translate,y_translate
    if i.ndim == 2:
        return Data_processing.translate_image(np.copy(i), [x_translate, y_translate]), x_translate,y_translate
    

class Dataset_3D(Dataset):
    def __init__(
        self,
        patient_class_list,
        patient_id_list,
        image_folder = '/mnt/camca_NAS/4DCT',
        have_manual_seg = True,
        img_size_3D = [160,160,96],
        picked_tf = 'random', #'random' or specific tf or 'ES'
        defined_input_path = None,
        relabel_LVOT = True,
        shuffle = False,
        augment = False,
        augment_frequency = None,
    ):
        super().__init__()
        self.patient_class_list = patient_class_list
        self.patient_id_list = patient_id_list
        self.image_folder = image_folder
        self.have_manual_seg = have_manual_seg
        self.img_size_3D = img_size_3D
        self.picked_tf = picked_tf
        self.defined_input_path = defined_input_path
        self.relabel_LVOT = relabel_LVOT
        self.shuffle = shuffle
        self.augment = augment
        self.augment_frequency = augment_frequency

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
        
    def __len__(self):
        return self.num_files 

    def __getitem__(self, index):
        # print('in this geiitem, self.index_array is: ', self.index_array)
        f = self.index_array[index]
        # print('index is: ', index, ' now we pick file ', f)
        patient_class = self.patient_class_list[f]
        patient_id = self.patient_id_list[f]

        # find time frames
        nii_files = ff.sort_timeframe(ff.find_all_target_files(['*.nii.gz'],os.path.join(self.image_folder,'nii-images',patient_class, patient_id, 'img-nii-resampled-1.5mm')),2)
        # uc seg
        seg_files = ff.sort_timeframe(ff.find_all_target_files(['*.nii.gz'],os.path.join(self.image_folder,'predicted_seg',patient_class, patient_id, 'seg-pred-0.625-4classes-connected-retouch-resampled-1.5mm')),2,'_')
        # mgh seg
        # seg_files = ff.sort_timeframe(ff.find_all_target_files(['*.nii.gz'],os.path.join(self.image_folder,'predicted_seg',patient_class, patient_id)),2,'_')
        tf_num = len(nii_files)

        # picked timeframe
        if self.picked_tf == 'random': 
            tf = random.randint(0, tf_num - 1)
        else: # input a particular time frame
            tf = self.picked_tf

        # load img and seg data
        if self.defined_input_path == None:
            img_file = nii_files[tf]#os.path.join('/mnt/camca_NAS/4DCT','nii-images',patient_class, patient_id, 'img-nii-resampled-1.5mm',str(tf)+'.nii.gz')
        else:
            img_file = self.defined_input_path
        seg_file = seg_files[tf] if self.have_manual_seg == True else None
        # print('picked tf: ', tf, ' img file: ', img_file, ' seg file: ', seg_file)

        # load image
        img = nb.load(img_file).get_fdata()
        if img.ndim == 4:
            img = img[:,:,:,0]
        # load seg
        if seg_file != None:
            seg = nb.load(seg_file).get_fdata()
            if seg.ndim == 4:
                seg = seg[:,:,:,0]
            seg = np.round(seg).astype(int)
            if self.relabel_LVOT == True:
                seg[seg==4] = 3
        else:
            seg = np.zeros_like(img).astype(int)
        # assert shape same
        # assert img.shape == seg.shape

        # do crop
        img = Data_processing.crop_or_pad(img, self.img_size_3D, value = np.min(img))
        seg = Data_processing.crop_or_pad(seg, self.img_size_3D, value = 0)

        ### do augmentation
        if self.augment == True  and np.random.uniform(0,1)  < self.augment_frequency:       
            img, z_rotate_degree = random_rotate(img, order = 0, z_rotate_range = [-10,10])
            seg,_ = random_rotate(seg, z_rotate_degree = z_rotate_degree, order = 0, fill_val = 0)
            # print('did rotate augmentation, z_rotate_degree: ', z_rotate_degree)

        if self.augment == True and np.random.uniform(0,1)  < self.augment_frequency:
            img, x_translate, y_translate = random_translate(img, translate_range = [-15,15])
            seg,_,_ = random_translate(seg, x_translate = x_translate, y_translate = y_translate)
            # print('did translate augmentation, x_translate: ', x_translate, ' y_translate: ', y_translate)

        # do normalization
        img = img / 1000

        # add channel dimension
        img = img[np.newaxis,:,:,:]
        seg = seg[np.newaxis,:,:,:]
    

        img_data = torch.from_numpy(img).float()
        seg_data = torch.from_numpy(seg).float()
        # print('img_data shape: ', img_data.shape, ' seg_data shape: ', seg_data.shape, ' seg unique: ', np.unique(seg_data.numpy()))

        return img_data, seg_data
    
    
    def on_epoch_end(self):
        print('now run on_epoch_end function')
        self.index_array = self.generate_index_array()