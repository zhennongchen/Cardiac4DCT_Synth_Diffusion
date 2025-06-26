import sys 
sys.path.append('/workspace/Documents')
import os
import torch
import numpy as np
import nibabel as nb
import Diffusion_motion_field.segmentation.model as seg_model
import Diffusion_motion_field.functions_collection as ff
import Diffusion_motion_field.Build_lists.Build_list as Build_list
import Diffusion_motion_field.segmentation.Generator as Generator

main_path = '/mnt/camca_NAS/4DCT'

####################### 
trial_name = 'seg_3D'
num_classes = 4 # 4 classes: background, LV, LA, LVOT
post_processing = True


epoch = 352
trained_model_filename = os.path.join(main_path, 'models', trial_name, 'models/model-' + str(epoch)+ '.pt')
save_folder = os.path.join(main_path,'mgh_data/predicted_seg')
main_folder = '/mnt/camca_NAS/4DCT/mgh_data'
# save_folder = os.path.join(main_path, 'models', trial_name, 'pred_seg'); os.makedirs(save_folder, exist_ok=True)

img_size_3D = [160,160,96]
#######################
# # define train
# data_sheet = os.path.join(main_path,'Patient_lists/patient_list_MVF_diffusion_train_test_filtered.xlsx')
data_sheet = os.path.join(main_path,'Patient_lists/mgh/patient_list_train_test_for_seg.xlsx')
b = Build_list.Build(data_sheet)
patient_class_list, patient_id_list,_ = b.__build__(batch_list = [0])

# build model
model = seg_model.Unet3D(
    init_dim = 16,
    channels = 1,
    num_classes = num_classes,
    dim_mults = (2,4,8,16),
    full_attn = (None,None, None, None),
    act = 'LeakyReLU',)


# main
for i in range(0,patient_class_list.shape[0]):
    
    patient_class = patient_class_list[i]
    patient_id = patient_id_list[i]

    print(patient_class, patient_id)

    # get the number of time frames
    # files = ff.find_all_target_files(['*.nii.gz'],os.path.join('/mnt/camca_NAS/4DCT','nii-images',patient_class, patient_id, 'img-nii-resampled-1.5mm'))
    files = ff.find_all_target_files(['*.nii.gz'],os.path.join('/mnt/camca_NAS/4DCT/mgh_data','nii-images',patient_class, patient_id, 'img-nii-resampled-1.5mm'))
    tf_num = len(files)
    print('num of tf:', tf_num)

    ff.make_folder([os.path.join(save_folder, patient_class), os.path.join(save_folder, patient_class, patient_id)])

    for tf in range(0,tf_num):
        print('tf:', tf)

        save_folder_case = os.path.join(save_folder,patient_class, patient_id)#, 'epoch' + str(epoch))
        os.makedirs(save_folder_case, exist_ok=True)
        save_file = os.path.join(save_folder_case, 'pred_s_' + str(tf) + '.nii.gz')

        if os.path.isfile(save_file) == False:
            generator = Generator.Dataset_3D(
                np.asarray([patient_class]),
                np.asarray([patient_id]),
                image_folder = main_folder,
                have_manual_seg = False,
                img_size_3D = img_size_3D,
                picked_tf = tf, #'random' or specific tf or 'ES'
                relabel_LVOT = True,)

            # sample:
            sampler = seg_model.Sampler(
                model,
                generator,
                image_size = img_size_3D,
                batch_size = 1)

            sampler.sample(trained_model_filename, save_file, patient_class, patient_id, picked_tf = tf, reshape_pred = True, save_gt_and_img=False, main_folder = main_folder)
        if post_processing == True:
            print('doing post processing')
            original_a = nb.load(save_file).get_fdata(); original_a = np.round(original_a).astype(int)
            a = np.copy(original_a)
            a[a != 1] = 0
            if np.sum(a) == 0:
                new_image = original_a
            else:
                new_image,need_to_remove = ff.remove_scatter3D(a,1)
                # print('need to remove:',need_to_remove)
                new_image[original_a == 2] = 2; new_image[original_a == 3] = 3
                nb.save(nb.Nifti1Image(new_image, nb.load(save_file).affine, nb.load(save_file).header), save_file)
            