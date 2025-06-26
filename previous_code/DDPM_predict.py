import sys 
sys.path.append('/workspace/Documents')
import os
import torch
import numpy as np
import Diffusion_motion_field.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion_3D as ddpm_3D
import Diffusion_motion_field.functions_collection as ff
import Diffusion_motion_field.Build_lists.Build_list as Build_list
import Diffusion_motion_field.Generator as Generator
main_path = '/mnt/camca_NAS/4DCT'
from ema_pytorch import EMA


###########
trial_name = 'MVF_DDPM_down_tf1_imgcon_onecase_noaug'
epoch = 14000
trained_model_filename = os.path.join(main_path, 'models', trial_name, 'models/model-' + str(epoch)+ '.pt')
save_folder = os.path.join(main_path, 'models', trial_name, 'pred_mvf'); os.makedirs(save_folder, exist_ok=True)

slice_range = [0,96]
image_size_3D = [80,80,slice_range[1]-slice_range[0]]

how_many_timeframes_together = 1
condition_on_image = True

objective = 'pred_noise'
timesteps = 1000
sampling_timesteps = 250
eta = 0. # usually use 1.
clip_range = [-1,1]

###########
data_sheet = os.path.join(main_path,'Patient_lists/patient_list_MVF_diffusion_train_test.xlsx')
b = Build_list.Build(data_sheet)
patient_class_list, patient_id_list,_ = b.__build__(batch_list = [0])

model = ddpm_3D.Unet3D_tfcondition(
    init_dim = 64,
    channels = 3 *  how_many_timeframes_together, 
    out_dim = 3 * how_many_timeframes_together,
    conditional_timeframe_input_dim = how_many_timeframes_together,
    conditional_diffusion_timeframe = True,
    conditional_diffusion_image = condition_on_image,
    dim_mults = (1, 2, 4, 8),
    flash_attn = False, 
    full_attn = (None, None, None,False),
)


diffusion_model = ddpm_3D.GaussianDiffusion3D(
    model,
    image_size_3D = image_size_3D,
    timesteps = timesteps,           # number of steps
    sampling_timesteps = sampling_timesteps,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    ddim_sampling_eta = eta,
    force_ddim = False,
    auto_normalize=False,
    objective = objective,
    clip_or_not = True, 
    clip_range = clip_range, 
)


for i in range(0,1):#patient_class_list.shape[0]):
    
    patient_class = patient_class_list[i]
    patient_id = patient_id_list[i]

    img_file = os.path.join('/mnt/camca_NAS/4DCT','nii-images',patient_class, patient_id, 'img-nii-resampled-1.5mm/0.nii.gz')
    print(i, patient_class, patient_id)

    # get the number of time frames
    files = ff.find_all_target_files(['*.nii.gz'],os.path.dirname(img_file))
    final_files = np.copy(files)
    for f in files:
        if 'moved' in f or 'original' in f:
            # remove it from the numpy array
            final_files = np.delete(final_files, np.where(final_files == f))
    files = ff.sort_timeframe(final_files,2)
    tf_num = len(files)

    ff.make_folder([os.path.join(save_folder, patient_class), os.path.join(save_folder, patient_class, patient_id)])

    save_folder_case = os.path.join(save_folder,patient_class, patient_id, 'epoch' + str(epoch)); os.makedirs(save_folder_case, exist_ok=True)

    if 1==1:#os.path.isfile(os.path.join(save_folder_case, 'pred-epoch-' + str(epoch) + '.nii.gz')) == 0:
        generator = Generator.Dataset_dual_3D(
            patient_class_list = np.asarray([patient_class]),
            patient_id_list = np.asarray([patient_id]),
            mvf_folder = '/mnt/camca_NAS/4DCT/mvf_warp0_onecase',
            how_many_timeframes_together = how_many_timeframes_together,
            image_size_3D = image_size_3D,
            slice_range = slice_range,
            normalize_factor = 'equation',
            maximum_cutoff = 20,
            minimum_cutoff = -20,
            condition_on_image = condition_on_image,
            shuffle = False,)

        # sample:
        sampler = ddpm_3D.Sampler(
            diffusion_model,
            generator,
            batch_size = 1,)

        save_file = os.path.join(save_folder_case, 'pred-epoch-' + str(epoch) + '.nii.gz')

        # pre-define the time frames
        # picked_tf = ff.pick_random_from_segments(tf_num - 1)
        # picked_tf[0] = 0
        picked_tf = [4]
        timeframes_data = []
        for picked_tf_n in picked_tf:
            A = picked_tf_n / (tf_num - 1)
            A = round(A * 20) / 20
            timeframes_data.append(A)
        input_timeframe = np.asarray(timeframes_data); input_timeframe = np.reshape(input_timeframe, (1,how_many_timeframes_together))
        print(picked_tf, input_timeframe, tf_num)


        sampler.sample_3D_w_trained_model(trained_model_filename=trained_model_filename,  slice_range = slice_range,
                save_file = save_file, image_file = img_file, input_timeframe = input_timeframe)
  