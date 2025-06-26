import sys 
sys.path.append('/workspace/Documents')
import os
import torch
import torch.nn.functional as F
import ast
import pandas as pd
import numpy as np
import nibabel as nb
from scipy.ndimage import zoom
from skimage.measure import block_reduce
from ema_pytorch import EMA
import Diffusion_motion_field.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion_3D as ddpm_3D
import Diffusion_motion_field.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_EDM as edm
import Diffusion_motion_field.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_EDM_warp as edm_warp
import Diffusion_motion_field.functions_collection as ff
import Diffusion_motion_field.Build_lists.Build_list as Build_list
import Diffusion_motion_field.EF_predictor.model as ef_model
import Diffusion_motion_field.Generator as Generator
import Diffusion_motion_field.Data_processing as Data_processing
main_path = '/mnt/camca_NAS/4DCT'


###########
trial_name = 'MVF_EDM_down_10tf_imgcon_EFcon_warp_orires'
epoch = 2160
trained_model_filename = os.path.join(main_path, 'models', trial_name, 'models/model-' + str(epoch)+ '.pt')
save_folder = os.path.join(main_path, 'models', trial_name, 'pred_mvf'); os.makedirs(save_folder, exist_ok=True)

latent = True if 'latent' in trial_name else False
how_many_timeframes_together = 10
picked_tf = 'ES' if how_many_timeframes_together == 1 else [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

mvf_size_3D = [160,160,96] if latent else [40,40,24] # downsampled
mvf_slice_range = [0,96]
latent_size_3D = [40,40,24]; latent_slice_range = [0,24]
mvf_folder = '/workspace/Documents/Data/mvf'#'/mnt/camca_NAS/4DCT/mvf_warp0_onecase'
VAE_model_path = os.path.join(main_path, 'models/VAE_embed3/models/model-54.pt')
cutoff_min = -30 if latent else -20
cutoff_max = 30 if latent else 20

downsample_list = (True,False,True,False)  if latent else (True, True, False, False) # default is (True, True, True, False) 

conditional_diffusion_timeframe = False
conditional_diffusion_image = True
conditional_diffusion_EF = True if 'EFcon' in trial_name else False
conditional_diffusion_seg = False if 'warp' in trial_name else False

###########
# data_sheet = os.path.join(main_path,'Patient_lists/uc/patient_list_MVF_diffusion_train_test_filtered_at_least_10tf.xlsx')
data_sheet = os.path.join(main_path,'Patient_lists/mgh/patient_list_MVF_diffusion_train_test.xlsx')
b = Build_list.Build(data_sheet)
patient_class_list, patient_id_list,_ = b.__build__(batch_list = [5])
# patient_class_list = patient_class_list[0:1]
# patient_id_list = patient_id_list[0:1]

# define diffusion model
model = ddpm_3D.Unet3D_tfcondition(
    init_dim = 64,
    channels = 3 * how_many_timeframes_together,
    out_dim = 3 * how_many_timeframes_together,
    # conditional_timeframe_input_dim = None,
    # conditional_diffusion_timeframe = conditional_diffusion_timeframe,
    conditional_diffusion_image = conditional_diffusion_image,
    conditional_diffusion_EF = conditional_diffusion_EF,
    conditional_diffusion_seg = conditional_diffusion_seg,
    dim_mults = (1, 2, 4, 8),
    downsample_list = downsample_list,
    upsample_list = (downsample_list[2], downsample_list[1], downsample_list[0], False),
    flash_attn = False, 
    full_attn = (None, None, False ,False), # (None, None, None,False),
)

diffusion_model = edm.EDM(
    model,
    image_size = latent_size_3D if latent else mvf_size_3D,
    num_sample_steps = 50,
    clip_or_not = True,
    clip_range = [-1,1],)

# define VAE model
vae_model = Generator.VAE_process(model_path = VAE_model_path) if latent else False
# define EF predict model
if 'EFpredict' in trial_name:
    # define EF predictor
    model = ef_model.CNN_EF_predictor_LSTM(init_dim = 16,channels = 1,dim_mults = (2,4,8),full_attn = (None,None, None),act = 'LeakyReLU',)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ema = EMA(model);ema.to(device)  
    data = torch.load('/mnt/camca_NAS/4DCT/models/EF_predictor_LSTM_noisy_trial2/models/model-196.pt', map_location=device)
    model.load_state_dict(data['model']); ema.load_state_dict(data["ema"])

result,result_EFpredictor = [],[]
for i in range(0,patient_class_list.shape[0]):
    
    patient_class = patient_class_list[i]
    patient_id = patient_id_list[i]
    print('patient_class:', patient_class, 'patient_id:', patient_id)

    ff.make_folder([os.path.join(save_folder, patient_class), os.path.join(save_folder, patient_class, patient_id)])

    for round_test in range(0,4):

        # get EF
        timeframe_info = pd.read_excel('/mnt/camca_NAS/4DCT/Patient_lists/mgh/patient_list_final_selection_timeframes.xlsx')
        row = timeframe_info[timeframe_info['patient_id'] == patient_id]
        preset_EF = round(row['EF_sampled_in_5tf_by_mvf'].iloc[0],2) if how_many_timeframes_together == 5 else round(row['EF_sampled_in_10tf_by_mvf'].iloc[0],2)
        # print('original EF:', preset_EF)
        sampled_time_frame_list = ast.literal_eval(row['sampled_time_frame_list'].iloc[0])
        normalized_time_frame_list = ast.literal_eval(row['normalized_time_frame_list_copy'].iloc[0])

        if round_test > 0 and conditional_diffusion_EF:
            # randomly get a new EF from 0.1 to 0.8
            # preset_EF = round(np.random.uniform(0.1,0.80),2)
            ef_condition_sheet = pd.read_excel('/mnt/camca_NAS/4DCT/Patient_lists/mgh/random_EF_conditions.xlsx')
            row = ef_condition_sheet[ef_condition_sheet['patient_id'] == patient_id]
            preset_EF = row['EF_condition'+str(round_test)].iloc[0]
            preset_EF = round(preset_EF,2)
        print('EF:', preset_EF)

        save_folder_case = os.path.join(save_folder,patient_class, patient_id, 'epoch' + str(epoch)+'_'+ str(round_test))
        os.makedirs(save_folder_case, exist_ok=True)

        with open(os.path.join(save_folder_case, 'EF.txt'), 'w') as f:
                f.write(str(preset_EF))

        generator = Generator.Dataset_dual_3D(
            VAE_process = vae_model if latent else False,

            patient_class_list = np.asarray([patient_class]),
            patient_id_list = np.asarray([patient_id]),
            mvf_folder = mvf_folder,
            how_many_timeframes_together = how_many_timeframes_together,
            mvf_size_3D = mvf_size_3D,
            latent_size_3D = latent_size_3D,
            slice_range = mvf_slice_range,
            preset_EF = preset_EF,
                        
            picked_tf = picked_tf,
            condition_on_image = True,
            condition_on_seg = True,
            augment_pre_done = True)


        if os.path.isfile(os.path.join(save_folder_case, 'pred_mvf.nii.gz'))== 1:
            print('already done')
            pred_mvf = nb.load(os.path.join(save_folder_case, 'pred_mvf.nii.gz')).get_fdata()
            pred_mvf = Data_processing.normalize_image(pred_mvf,normalize_factor = 'equation', image_max = cutoff_max, image_min =cutoff_min)
            for ii in range(0,len(generator)):
                _,_,_,_,_,_, condition_seg_data = generator[ii]
            pred_mvf_torch = torch.from_numpy(np.transpose(pred_mvf, (3, 0, 1, 2))).unsqueeze(0).float().to('cuda')
            seg_img_torch = condition_seg_data.unsqueeze(0).to('cuda')
            denormalize_mvf = edm_warp.DeNormalizeMVF(image_min=cutoff_min, image_max=cutoff_max)
            EF_pred,volumes = edm_warp.warp_loss(pred_mvf_torch, seg_img_torch, denormalize_mvf)
            EF_pred = EF_pred.cpu().detach().numpy()[0][0] if isinstance(EF_pred, torch.Tensor) else EF_pred
            print('EF pred', EF_pred)
            pred_mvf_numpy = np.transpose(pred_mvf,(3,0,1,2)) 
            save_file = os.path.join(save_folder_case, 'pred-epoch-' + str(epoch) + '.nii.gz')
         
        else:                
            # sample:
            count = 0
            low_end = [0,0,0.395, 0.500]
            high_end = [0,0.400, 0.50, 0.90]
            while True:
                sampler = edm.Sampler(diffusion_model,generator,batch_size = 1,image_size = latent_size_3D if latent else mvf_size_3D,)

                save_file = os.path.join(save_folder_case, 'pred-epoch-' + str(epoch) + '.nii.gz')

                pred_mvf_torch, EF_pred, pred_mvf_numpy = sampler.sample_3D_w_trained_model(trained_model_filename=trained_model_filename,cutoff_min = cutoff_min, cutoff_max = cutoff_max,
                            save_file = save_file,  patient_class = patient_class, patient_id = patient_id)
                
                EF_pred = EF_pred.cpu().detach().numpy()[0][0] if isinstance(EF_pred, torch.Tensor) else EF_pred

                if 'warp' not in trial_name: 
                    break
                else:
                    if round_test == 0: # original
                        break
                    else:
                        in_range = (EF_pred >= low_end[round_test] and EF_pred <= high_end[round_test]) # make it in the correct EF range
                        if in_range or count >= 10:
                            break
                        else:
                            count += 1

        result.append([patient_class, patient_id, round_test, preset_EF, EF_pred])
        df = pd.DataFrame(result, columns=['patient_class', 'patient_id', 'round_test', 'preset_EF', 'pred_EF'])
        df.to_excel(os.path.join(os.path.dirname(save_folder), 'EF_from_warploss_results_epoch' + str(epoch) + '.xlsx'), index=False)

        if 'EFpredict' in trial_name:
            pred_EF,_,_,_ = ema.ema_model(pred_mvf_torch)
            pred_EF = pred_EF.cpu().detach().numpy()
            print('pred_EF from EFpredict:', pred_EF[0][0])
            result_EFpredictor.append([patient_class, patient_id, round_test, preset_EF, pred_EF[0][0]])
            df_EFpredictor = pd.DataFrame(result_EFpredictor, columns=['patient_class', 'patient_id', 'round_test', 'preset_EF', 'pred_EF'])
            df_EFpredictor.to_excel(os.path.join(os.path.dirname(save_folder), 'EF_from_EFpredictor_results_epoch' + str(epoch) + '.xlsx'), index=False)

        # save in original resolution
        if os.path.isfile(os.path.join(save_folder_case, 'pred_tf'+ str(sampled_time_frame_list[-1])+'_x.nii.gz')) == 0:
            image_file = os.path.join('/mnt/camca_NAS/4DCT','mgh_data/nii-images',patient_class, patient_id, 'img-nii-resampled-1.5mm/0.nii.gz')
            affine = nb.load(image_file).affine
            for ii in range(len(sampled_time_frame_list)):
                segment_range = [3*ii, 3*(ii+1)]
                mvf1 = pred_mvf_numpy[3*ii:3*(ii+1),...]; mvf1 = np.moveaxis(mvf1, 0, -1)
                if latent == False:
                    mvf1 = zoom(mvf1, (4,4,4,1), order=1)
                nb.save(nb.Nifti1Image(mvf1[:,:,:,0], affine), os.path.join(os.path.dirname(save_file), 'pred_tf'+str(sampled_time_frame_list[ii])+'_x.nii.gz'))
                nb.save(nb.Nifti1Image(mvf1[:,:,:,1], affine), os.path.join(os.path.dirname(save_file), 'pred_tf'+str(sampled_time_frame_list[ii])+'_y.nii.gz'))
                nb.save(nb.Nifti1Image(mvf1[:,:,:,2], affine), os.path.join(os.path.dirname(save_file), 'pred_tf'+str(sampled_time_frame_list[ii])+'_z.nii.gz'))
                if latent == True:
                    nb.save(nb.Nifti1Image(mvf1, affine), os.path.join(os.path.dirname(save_file), 'pred_latent_tf'+str(sampled_time_frame_list[ii])+'.nii.gz'))
                # else:
                    #     nb.save(nb.Nifti1Image(mvf1, affine), os.path.join(os.path.dirname(save_file), 'pred_tf'+str(sampled_time_frame_list[ii])+'.nii.gz'))

                if round_test ==0:
                    if latent == False:
                        gt = os.path.join('/mnt/camca_NAS/4DCT/mvf_warp0_onecase',patient_class,patient_id, 'voxel_final', str(sampled_time_frame_list[ii]) + '.nii.gz')
                    else:
                        gt = os.path.join('/mnt/camca_NAS/4DCT','models/VAE_embed3/pred_mvf',patient_class,patient_id, 'epoch100', 'pred_latent_tf'+str(sampled_time_frame_list[ii])+'.nii.gz')
                    gt = nb.load(gt).get_fdata()
                    gt = block_reduce(gt, (4,4,4,1), func=np.mean)
                    gt = zoom(gt, (4,4,4,1), order=1)
                    nb.save(nb.Nifti1Image(gt[:,:,:,0], affine), os.path.join(os.path.dirname(save_file), 'gt_tf'+str(sampled_time_frame_list[ii])+'_x.nii.gz'))
                    nb.save(nb.Nifti1Image(gt[:,:,:,1], affine), os.path.join(os.path.dirname(save_file), 'gt_tf'+str(sampled_time_frame_list[ii])+'_y.nii.gz'))
                    nb.save(nb.Nifti1Image(gt[:,:,:,2], affine), os.path.join(os.path.dirname(save_file), 'gt_tf'+str(sampled_time_frame_list[ii])+'_z.nii.gz'))