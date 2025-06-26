import sys 
sys.path.append('/workspace/Documents')
import os
import torch
import numpy as np
import nibabel as nb
import pandas as pd
import ast
import Diffusion_motion_field.EF_predictor.model as ef_model
import Diffusion_motion_field.functions_collection as ff
import Diffusion_motion_field.Build_lists.Build_list as Build_list
import Diffusion_motion_field.EF_predictor.Generator as Generator
from ema_pytorch import EMA
main_path = '/mnt/camca_NAS/4DCT'
 
###########
trial_name = 'EF_predictor_temporalConv_noisy'

epoch = 596
trained_model_filename = os.path.join(main_path, 'models', trial_name, 'models/model-' + str(epoch)+ '.pt')

##########
# data_sheet = os.path.join(main_path,'Patient_lists/uc/patient_list_MVF_diffusion_train_test_filtered_at_least_10tf.xlsx')
data_sheet = os.path.join(main_path,'Patient_lists/mgh/patient_list_MVF_diffusion_train_test.xlsx')
b = Build_list.Build(data_sheet)
patient_class_list, patient_id_list,_ = b.__build__(batch_list = [5])

# define predictor 
model = ef_model.CNN_EF_predictor_temporalConv(
    init_dim = 16,
    channels = 1,
    dim_mults = (2,4,8),
    full_attn = (None,None, None),
    act = 'LeakyReLU',)

# load
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ema = EMA(model)
ema.to(device)  
data = torch.load(trained_model_filename, map_location=device)
model.load_state_dict(data['model'])
ema.load_state_dict(data["ema"])

# main
result = []
for i in range(0,patient_class_list.shape[0]):
    
    patient_class = patient_class_list[i]
    patient_id = patient_id_list[i]
    print('patient_class:', patient_class, 'patient_id:', patient_id)

    # get EF
    timeframe_info = pd.read_excel('/mnt/camca_NAS/4DCT/Patient_lists/mgh/patient_list_final_selection_timeframes.xlsx')
    row = timeframe_info[timeframe_info['patient_id'] == patient_id]
    preset_EF = round(row['EF_sampled_in_10tf_by_mvf'].iloc[0],2)
    print('preset_EF:', preset_EF)
    sampled_time_frame_list = ast.literal_eval(row['sampled_time_frame_list'].iloc[0])
    normalized_time_frame_list = ast.literal_eval(row['normalized_time_frame_list_copy'].iloc[0])

    generator_test = Generator.Dataset_MVF(
        np.array([patient_class]),
        np.array([patient_id]),
        mvf_folder = '/workspace/Documents/Data/mvf',
        mvf_size_3D = [40,40,24],
        picked_tf = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        mvf_cutoff = [-20,20],
        shuffle =False,
        augment = False,
        noise_add_frequency = -0.1,
        augment_pre_done = True)

    # no noise version
    for ii in range(len(generator_test)):
        img, _, _, ef = generator_test[ii]
        # add batch size to img
        img = img.unsqueeze(0)
        predict_EF,_,_,_ = ema.ema_model(img.to(device))
        predict_EF = predict_EF.cpu().detach().numpy()[0][0]
    
    # add noise
    predict_EF_noise_list = []
    for t in range(0,3):
        generator_test_noise = generator_test
        generator_test_noise.noise_add_frequency = 1.1

        for ii in range(len(generator_test_noise)):
            img, _, _, ef = generator_test_noise[ii]
            # add batch size to img
            img = img.unsqueeze(0)
            predict_EF_noise,_,_,_ = ema.ema_model(img.to(device))
            predict_EF_noise = predict_EF_noise.cpu().detach().numpy()[0][0]
            predict_EF_noise_list.append(predict_EF_noise)
    
    a = [patient_class, patient_id, preset_EF, predict_EF] + predict_EF_noise_list
    result.append(a)

    df = pd.DataFrame(result, columns = ['patient_class', 'patient_id', 'preset_EF', 'predict_EF'] + ['predict_EF_noise_' + str(i) for i in range(1,4)])
    df.to_excel(os.path.join(main_path, 'models', trial_name, 'predict_EF_noise_mgh_epoch_' + str(epoch) + '.xlsx'), index = False)
            

   
     


         