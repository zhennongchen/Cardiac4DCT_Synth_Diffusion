from math import sqrt
from random import random
import torch
from torch import nn, einsum
import torch.nn.functional as F

from tqdm import tqdm
from einops import rearrange, repeat, reduce
from scipy.ndimage import zoom
from skimage.measure import block_reduce

from Diffusion_motion_field.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion_3D import *
from Diffusion_motion_field.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_EDM import *
from Diffusion_motion_field.denoising_diffusion_pytorch.denoising_diffusion_pytorch.version import __version__

import Diffusion_motion_field.functions_collection as ff
import Diffusion_motion_field.Data_processing as Data_processing

# helpers

def exists(val):
    return val is not None

def default(val, d):
    if exists(val): 
        return val
    return d() if callable(d) else d

# tensor helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def generate_random_ef(batch_size, low_end, high_end, device ):
    ef_np = np.random.uniform(low=low_end, high=high_end, size=(batch_size, 1))

    # Step 2: 保留两位小数（四舍五入）
    ef_np = np.round(ef_np, 2)

    # Step 3: 转成 torch tensor，并转 float32
    ef_tensor = torch.from_numpy(ef_np).float()

    # Step 4: 送到指定设备
    ef_tensor = ef_tensor.to(device)

    return ef_tensor

# turn pred_mvf into voxels unit
class DeNormalizeMVF(nn.Module):
    def __init__(self, image_min=-20.0, image_max=20.0):
        super().__init__()
        self.register_buffer('image_min', torch.tensor(image_min))
        self.register_buffer('image_max', torch.tensor(image_max))

    def forward(self, x_norm):
        # x_norm ∈ [-1, 1]
        scale = (self.image_max - self.image_min) / 2
        shift = (self.image_max + self.image_min) / 2
        return x_norm * scale + shift  # output is voxel-space MVF

def warp_segmentation_from_mvf(seg_t0, mvf_voxel):
    """
    seg_t0:    [B, 1, X, Y, Z] → will be permuted to [B, 1, Z, Y,X]
    mvf_voxel: [B, 3, X, Y, Z] → will be permuted to [B, 3, Z, Y,X]
    returns:
        warped_seg: [B, 1, X, Y, Z]
    """
    # Step 0: permute to [B, 1, Z, Y,X]
    seg_t0 = seg_t0.permute(0, 1, 4, 3, 2).contiguous()
    mvf_voxel = mvf_voxel.permute(0, 1, 4, 3, 2).contiguous()

    B, _, D, H, W = seg_t0.shape  # Z, Y, X
    device = seg_t0.device

    # Step 1: normalize MVF to [-1, 1]
    mvf_norm = torch.zeros_like(mvf_voxel)
    mvf_norm[:, 0] = mvf_voxel[:, 0] * 2 / (W - 1)  # dx
    mvf_norm[:, 1] = mvf_voxel[:, 1] * 2 / (H - 1)  # dy
    mvf_norm[:, 2] = mvf_voxel[:, 2] * 2 / (D - 1)  # dz

    # Step 2: create identity grid in Z, Y, X order
    grid_z = torch.linspace(-1, 1, D, device=device)
    grid_y = torch.linspace(-1, 1, H, device=device)
    grid_x = torch.linspace(-1, 1, W, device=device)
    meshz, meshy, meshx = torch.meshgrid(grid_z, grid_y, grid_x, indexing='ij')  # [D, H, W]
    identity_grid = torch.stack((meshx, meshy, meshz), dim=-1)  # [D, H, W, 3]
    identity_grid = identity_grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)  # [B, D, H, W, 3]

    # Step 3: add displacement
    displacement_grid = identity_grid + mvf_norm.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]

    # Step 4: warp
    warped = F.grid_sample(
        seg_t0, displacement_grid,
        mode='bilinear', padding_mode='border', align_corners=True
    )  # [B, 1, D, H, W]

    # Step 5: permute back to [B, 1, X, Y, Z]
    warped = warped.permute(0, 1, 4,3, 2).contiguous()

    return warped  # [B, 1, X, Y, Z]

def warp_loss(denoised_image, initial_input, denormalize_mvf):
    B,T,X,Y,Z = denoised_image.shape
    # get segmentation at ED
    seg_img_torch_set = initial_input[:,-1,:,:,:].unsqueeze(1)
    seg_img_torch_set_upsample = F.interpolate(seg_img_torch_set, size=(4*X, 4*Y, 4*Z), mode='trilinear', align_corners=True)

    # warp
    denoised_image = denormalize_mvf(denoised_image)  # denormalize to voxel space
    volume_list = []
    for t in range(10):
        pred_mvf_voxel_set_t = denoised_image[:, 3*t:3*(t+1),...]
        pred_mvf_voxel_set_t_upsample = F.interpolate(pred_mvf_voxel_set_t, size=(4*X, 4*Y, 4*Z), mode='trilinear', align_corners=True)
        warped_seg_t = warp_segmentation_from_mvf(seg_img_torch_set_upsample, pred_mvf_voxel_set_t_upsample)
        # print('warped_seg_t shape: ', warped_seg_t.shape)   
        volume_t = warped_seg_t.sum(dim=(1,2,3,4))
        volume_list.append(volume_t)
    volumes = torch.stack(volume_list, dim = 1)
    # print('volumes value: ', volumes)
    v0 = volumes[:,0]
    vmin = volumes.min(dim=1).values
    EF_pred = (v0-vmin) / (v0 )
    EF_pred = EF_pred.unsqueeze(1)  
    return EF_pred, volumes


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        discriminator_model,
        generator_train,
        generator_val,
        EF_loss_weight,
        adversarial_loss_weight, 
        train_batch_size,

        *,
        train_num_steps = 10000, # total training epochs
        results_folder = None,
        train_lr = 1e-4,
        train_lr_decay_every = 100, 
        save_models_every = 1,
        validation_every = 1,
        
        ema_update_every = 10,
        ema_decay = 0.95,
        adam_betas = (0.9, 0.99),

        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        max_grad_norm = 1., 
         
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model
        self.model = diffusion_model   # it's not just the model architecture, but the actual model with loss calculation
        self.D_model = discriminator_model
        self.conditional_diffusion_image = self.model.conditional_diffusion_image
        self.conditional_diffusion_EF = self.model.conditional_diffusion_EF
        self.conditional_diffusion_seg = self.model.conditional_diffusion_seg
        print('conditional_image: ', self.conditional_diffusion_image, ' condition_EF: ', self.conditional_diffusion_EF, ' condition_seg: ', self.conditional_diffusion_seg)
        self.channels = diffusion_model.channels
        self.EF_loss_weight = EF_loss_weight
        self.adversarial_loss_weight = adversarial_loss_weight

        # sampling and training hyperparameters
        self.batch_size = train_batch_size
        self.train_num_steps = train_num_steps

        # dataset and dataloader
        self.ds = generator_train
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())
        self.dl = self.accelerator.prepare(dl)

        self.ds_val = generator_val
        dl_val = DataLoader(self.ds_val, batch_size = train_batch_size, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())
        self.dl_val = self.accelerator.prepare(dl_val)

        # optimizer
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)
        self.opt_D = Adam(discriminator_model.parameters(), lr = train_lr, betas = adam_betas)
        self.scheduler = StepLR(self.opt, step_size = 1, gamma=0.95)
        self.train_lr_decay_every = train_lr_decay_every
        self.save_models_every = save_models_every
        self.validation_every = validation_every

        self.max_grad_norm = max_grad_norm

        # for logging results in a folder periodically
        # EMA:
        # The purpose of using an EMA is to stabilize and improve the performance of a model during training. It achieves this by maintaining a smoothed version of the model's parameters, which reduces the impact of noise or fluctuations in the training process.
        #Typically, during training, you will update both the original model and the EMA model, but when you want to evaluate or make predictions, you would use the EMA model because it provides a more stable representation of the model's knowledge. This is especially useful in tasks like generative modeling, where you want to generate high-quality samples from the model.
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = results_folder
        ff.make_folder([self.results_folder])

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        self.D_model, self.opt_D = self.accelerator.prepare(self.D_model, self.opt_D)

        self.bce_loss = torch.nn.BCELoss()

    @property
    def device(self):
        return self.accelerator.device

    def save(self, stepNum):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'decay_steps': self.scheduler.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__}
        
        torch.save(data, os.path.join(self.results_folder, 'model-' + str(stepNum) + '.pt'))

    def save_discriminator(self, stepNum):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'D_model': self.accelerator.get_state_dict(self.D_model),
            'D_opt': self.opt_D.state_dict(),}

        torch.save(data, os.path.join(self.results_folder, 'discriminator-' + str(stepNum) + '.pt'))

    def load_model(self, trained_model_filename):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(trained_model_filename, map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        self.scheduler.load_state_dict(data['decay_steps'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])


    def train(self, pre_trained_model = None ,start_step = None):
        accelerator = self.accelerator
        device = accelerator.device

        # load pre-trained
        if pre_trained_model is not None:
            self.load_model(pre_trained_model)
            print('model loaded from ', pre_trained_model)

        if start_step is not None:
            self.step = start_step

        self.scheduler.step_size = 1
        val_loss = np.inf; val_diffusion_loss = np.inf; val_EF_loss_factual = np.inf; val_EF_loss_counter = np.inf; val_adversarial_loss = np.inf
        training_log = []

        denormalize_mvf = DeNormalizeMVF(image_min=-20.0, image_max=20.0)
        
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
            # print('11111 save_models_every: ', self.save_models_every, 'validation_every: ', self.validation_every)
            
            while self.step < self.train_num_steps:
                print('training epoch: ', self.step + 1)
                print('learning rate: ', self.scheduler.get_last_lr()[0])

                average_D_loss = []; average_loss = []; average_diffusion_loss = []; average_EF_loss_factual = []; average_EF_loss_counter = []; average_adversarial_loss = []
                count = 0
                # load data
                for batch in self.dl:
                    # prepare data
                    batch_x0_image, _,_, batch_condition_image, batch_condition_EF, batch_condition_seg, batch_condition_seg_orires = batch
                    data_x0 , data_condition_image, data_condition_EF, data_condition_seg , data_condition_seg_orires = batch_x0_image.to(device), batch_condition_image.to(device), batch_condition_EF.to(device), batch_condition_seg.to(device), batch_condition_seg_orires.to(device)

                    if not self.conditional_diffusion_image:
                        data_condition_image = None
                    if not self.conditional_diffusion_EF:
                        data_condition_EF = None
                    if not self.conditional_diffusion_seg:
                        data_condition_seg = None
                
                    ######## train Discriminator
                    for p in self.model.parameters():  # freeze G
                        p.requires_grad = False
                    for p in self.D_model.parameters():  # unfreeze D
                        p.requires_grad = True
                    self.opt_D.zero_grad()
                    with self.accelerator.autocast():
                        with torch.no_grad():
                            # counterfactual generation
                            condition_EF_random = generate_random_ef(batch_size = data_condition_EF.shape[0], low_end = 0.05, high_end = 0.85, device = device)
                            _,denoised_image,_ = self.model(data_x0, condition_image = data_condition_image, condition_EF = condition_EF_random, condition_seg = data_condition_seg)
                        real_mvf = torch.clone(data_x0)
                        D_real,_,_,_ = self.D_model(real_mvf)
                        D_fake,_,_,_ = self.D_model(denoised_image.detach())
                        # print('range of real_mvf: ', torch.min(real_mvf), torch.max(real_mvf), ' shape of real_mvf: ', real_mvf.shape)
                        # print('ranage of denoised_image: ', torch.min(denoised_image), torch.max(denoised_image), ' shape of denoised_image: ', denoised_image.shape)
                        loss_D_real = self.bce_loss(D_real, torch.ones_like(D_real))
                        loss_D_fake = self.bce_loss(D_fake, torch.zeros_like(D_fake))
                        loss_D = (loss_D_real + loss_D_fake) / 2

                    average_D_loss.append(loss_D.item())
                    self.accelerator.backward(loss_D)
                    self.opt_D.step()

                    ######## train diffusion
                    for p in self.model.parameters():  # unfreeze G
                        p.requires_grad = True
                    for p in self.D_model.parameters():  # freeze D
                        p.requires_grad = False
                    self.opt.zero_grad()
                    with self.accelerator.autocast():
                        # factual generation
                        diffusion_loss,denoised_image, _ = self.model(data_x0,  condition_image = data_condition_image, condition_EF = data_condition_EF, condition_seg = data_condition_seg)
                        # print('diffusion_loss value: ', diffusion_loss.item())
                        EF_pred_factual, _ = warp_loss(denoised_image, data_condition_seg_orires, denormalize_mvf)
                        EF_loss_factual = F.mse_loss(EF_pred_factual, data_condition_EF)

                        # counterfactual generation
                        condition_EF_random = generate_random_ef(batch_size = data_condition_EF.shape[0], low_end = 0.05, high_end = 0.85, device = device)
                        # print('condition_EF_random: ', condition_EF_random)
                        _,denoised_image,_ = self.model(data_x0, condition_image = data_condition_image, condition_EF = condition_EF_random, condition_seg = data_condition_seg)
                        
                        # print('shape of data_condition_seg_orires: ', data_condition_seg_orires.shape, ' denoised_image shape: ', denoised_image.shape)
                        EF_pred, volumes = warp_loss(denoised_image, data_condition_seg_orires, denormalize_mvf)
                        # calculate mse loss in EF
                        EF_loss_counter = F.mse_loss(EF_pred, condition_EF_random)

                        # adversarial loss
                        D_fake_for_G,_,_,_ = self.D_model(denoised_image)
                        adv_loss = self.bce_loss(D_fake_for_G, torch.ones_like(D_fake_for_G))

                        loss = diffusion_loss + self.EF_loss_weight * (EF_loss_factual + EF_loss_counter) + self.adversarial_loss_weight * adv_loss
                        
                    average_loss.append(loss.item()); average_diffusion_loss.append(diffusion_loss.item()); average_EF_loss_factual.append(EF_loss_factual.item()); average_EF_loss_counter.append(EF_loss_counter.item()); average_adversarial_loss.append(adv_loss.item())

                    self.accelerator.backward(loss)
                    self.opt.step()

                average_D_loss, average_loss, average_diffusion_loss, average_EF_loss_factual, average_EF_loss_counter, average_adversarial_loss = sum(average_D_loss) / len(average_D_loss), sum(average_loss) / len(average_loss), sum(average_diffusion_loss) / len(average_diffusion_loss), sum(average_EF_loss_factual) / len(average_EF_loss_factual), sum(average_EF_loss_counter) / len(average_EF_loss_counter), sum(average_adversarial_loss) / len(average_adversarial_loss)
                print('average Discriminator loss: ', average_D_loss, ' average loss: ', average_loss, ' average diffusion loss: ', average_diffusion_loss, ' average EF loss factual: ', average_EF_loss_factual, ' average EF loss counter: ', average_EF_loss_counter, ' average adversarial loss: ', average_adversarial_loss)
                pbar.set_description(f'average loss: {average_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
               
                self.step += 1

                # save the model
                if self.step !=0 and divisible_by(self.step, self.save_models_every):
                   self.save(self.step)
                   self.save_discriminator(self.step)
                
                if self.step !=0 and divisible_by(self.step, self.train_lr_decay_every):
                    self.scheduler.step()
                    
                self.ema.update()

                # do the validation if necessary
                if self.step !=0 and divisible_by(self.step, self.validation_every):
                    print('validation at step: ', self.step)
                    self.model.eval()
                    with torch.no_grad():
                        val_loss = []; val_diffusion_loss = []; val_EF_loss_factual = []; val_EF_loss_counter = []; val_adversarial_loss = []
                        for batch in self.dl_val:
                            batch_x0_image, _,_, batch_condition_image, batch_condition_EF , batch_condition_seg, batch_condition_seg_orires = batch
                            data_x0 , data_condition_image, data_condition_EF, data_condition_seg, data_condition_seg_orires = batch_x0_image.to(device), batch_condition_image.to(device), batch_condition_EF.to(device), batch_condition_seg.to(device), batch_condition_seg_orires.to(device)
                            if not self.conditional_diffusion_image:
                                data_condition_image = None
                            if not self.conditional_diffusion_EF:
                                data_condition_EF = None
                            if not self.conditional_diffusion_seg:
                                data_condition_seg = None

                            with self.accelerator.autocast():
                               # factual generation
                                diffusion_loss,denoised_image,_ = self.model(data_x0,  condition_image = data_condition_image, condition_EF = data_condition_EF, condition_seg = data_condition_seg)
                                EF_pred_factual, _ = warp_loss(denoised_image, data_condition_seg_orires, denormalize_mvf)
                                EF_loss_factual = F.mse_loss(EF_pred_factual, data_condition_EF)

                                # counterfactual generation
                                condition_EF_random = generate_random_ef(batch_size = data_condition_EF.shape[0], low_end = 0.05, high_end = 0.85, device = device)
                                _,denoised_image,_ = self.model(data_x0, condition_image = data_condition_image, condition_EF = condition_EF_random, condition_seg = data_condition_seg)
                            
                                EF_pred, volumes = warp_loss(denoised_image, data_condition_seg_orires, denormalize_mvf)
                                # # calculate mse loss in EF
                                EF_loss_counter = F.mse_loss(EF_pred, condition_EF_random)

                                D_fake_for_G,_,_,_ = self.D_model(denoised_image)
                                adv_loss = self.bce_loss(D_fake_for_G, torch.ones_like(D_fake_for_G))

                                loss = diffusion_loss + self.EF_loss_weight * (EF_loss_factual + EF_loss_counter) + self.adversarial_loss_weight * adv_loss

                            val_loss.append(loss.item()); val_diffusion_loss.append(diffusion_loss.item()); val_EF_loss_factual.append(EF_loss_factual.item()); val_EF_loss_counter.append(EF_loss_counter.item()); val_adversarial_loss.append(adv_loss.item())
                        val_loss, val_diffusion_loss, val_EF_loss_factual, val_EF_loss_counter, val_adversarial_loss = sum(val_loss) / len(val_loss), sum(val_diffusion_loss) / len(val_diffusion_loss), sum(val_EF_loss_factual) / len(val_EF_loss_factual), sum(val_EF_loss_counter) / len(val_EF_loss_counter), sum(val_adversarial_loss) / len(val_adversarial_loss)
                        print('validation loss: ', val_loss, ' validation diffusion loss: ', val_diffusion_loss ,' validation EF loss factual: ', val_EF_loss_factual, ' validation EF loss counter: ', val_EF_loss_counter, ' validation adversarial loss: ', val_adversarial_loss)
                    self.model.train(True)

                # save the training log
                training_log.append([self.step, self.scheduler.get_last_lr()[0], average_D_loss, average_loss, average_diffusion_loss, average_EF_loss_factual, average_EF_loss_counter, average_adversarial_loss, val_loss, val_diffusion_loss, val_EF_loss_factual, val_EF_loss_counter, val_adversarial_loss])
                df = pd.DataFrame(training_log,columns = ['iteration', 'learning_rate','discriminator_loss', 'average_loss', 'diffusion_loss', 'EF_loss_factual', 'EF_loss_counter', 'adversarial_loss', 'val_loss', 'val_diffusion_loss', 'val_EF_loss_factual', 'val_EF_loss_counter', 'val_adversarial_loss'])
                log_folder = os.path.join(os.path.dirname(self.results_folder),'log');ff.make_folder([log_folder])
                df.to_excel(os.path.join(log_folder, 'training_log.xlsx'),index=False)

                # at the end of each epoch, call on_epoch_end
                self.ds.on_epoch_end()
                self.ds_val.on_epoch_end()
                pbar.update(1)

        accelerator.print('training complete')


# # Sampling class
# class Sampler(object):
#     def __init__(
#         self,
#         diffusion_model,
#         generator,
#         batch_size,
#         image_size = None,
#         device = 'cuda',

#     ):
#         super().__init__()

#         # model
#         self.model = diffusion_model  
#         if device == 'cuda':
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         if device == 'cpu':
#             self.device = torch.device("cpu")

#         self.conditional_diffusion_image = self.model.conditional_diffusion_image
#         self.conditional_diffusion_EF = self.model.conditional_diffusion_EF
#         self.conditional_diffusion_seg = self.model.conditional_diffusion_seg

#         self.channels = diffusion_model.channels
#         if image_size is None:
#             self.image_size = self.model.image_size
#         else:
#             self.image_size = image_size
#         self.batch_size = batch_size

#         # dataset and dataloader

#         self.generator = generator
#         dl = DataLoader(self.generator, batch_size = self.batch_size, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())     
#         self.how_many_timeframes_together = self.generator.how_many_timeframes_together

#         self.dl = dl
#         self.cycle_dl = cycle(dl)
 
#         # EMA:
#         self.ema = EMA(diffusion_model)
#         self.ema.to(self.device)

#     def load_model(self, trained_model_filename):

#         data = torch.load(trained_model_filename, map_location=self.device)

#         self.model.load_state_dict(data['model'])

#         self.step = data['step']

#         self.ema.load_state_dict(data["ema"])


#     def sample_3D_w_trained_model(self, trained_model_filename, cutoff_max = None, cutoff_min = None,
#                 save_file = None, input_timeframe = None,picked_tf = None, patient_class = None, patient_id = None, save_gt = False):
  
#         self.load_model(trained_model_filename) 
        
#         device = self.device

#         self.ema.ema_model.eval()
#         # check whether model is on GPU:
#         # print('model device: ', next(self.ema.ema_model.parameters()).device)

#         # image info (if needed)
#         # image_file = os.path.join('/mnt/camca_NAS/4DCT','models/VAE_embed3/pred_mvf',patient_class,patient_id, 'epoch100', 'pred_latent_tf0.nii.gz')
#         image_file = os.path.join('/mnt/camca_NAS/4DCT/','nii-images',patient_class, patient_id, 'img-nii-resampled-1.5mm/0.nii.gz')
#         affine = nb.load(image_file).affine
#         # img_info = nb.load(image_file).get_fdata()
#         # scale_factors = (0.5,0.5,1) 
#         # downsampled_img_info = zoom(img_info, scale_factors, order=1)[:,:,slice_range[0]:slice_range[1]]

#         # start to run
#         with torch.inference_mode():
#             datas = next(self.cycle_dl)
#             batch_x0_image, batch_condition_tf, batch_condition_tf_normalized, batch_condition_image, batch_condition_EF, batch_condition_seg = datas
#             data_x0, data_condition_image, data_condition_EF, data_condition_seg = batch_x0_image.to(device), batch_condition_image.to(device), batch_condition_EF.to(device), batch_condition_seg.to(device)

#             if not self.conditional_diffusion_image: 
#                 data_condition_image = None
#             if not self.conditional_diffusion_EF:
#                 data_condition_EF = None
#             if not self.conditional_diffusion_seg:
#                 data_condition_seg = None
            
#             print('data_condition_EF: ', data_condition_EF, ' data_condition_seg shape: ', data_condition_seg.shape)
#             pred_mvf = self.ema.ema_model.sample( condition_image = data_condition_image, condition_EF = data_condition_EF, condition_seg = data_condition_seg, batch_size = self.batch_size)
        
#         pred_mvf_saved = torch.clone(pred_mvf)

#         # apply to segmentation
#         B,_,X,Y,Z = pred_mvf.shape
#         denormalize_mvf = DeNormalizeMVF(image_min=cutoff_min, image_max=cutoff_max)
#         # EF_pred, volumes = warp_loss(pred_mvf, data_condition_seg, denormalize_mvf)
#         # print('EF predicted: ', EF_pred)
#         # print('volumes: ', volumes)
#         pred_mvf_voxels = denormalize_mvf(torch.clone(pred_mvf))  # denormalize to voxel space
#         volume_list = []
#         data_condition_seg_upsample = F.interpolate(data_condition_seg, size=(4*X, 4*Y, 4*Z), mode='trilinear', align_corners=True)
#         for t in range(10):
#             pred_mvf_voxel_set_t = pred_mvf_voxels[:, 3*t:3*(t+1),...]
#             pred_mvf_voxel_set_t_upsample = F.interpolate(pred_mvf_voxel_set_t, size=(4*X, 4*Y, 4*Z), mode='trilinear', align_corners=True)
#             warped_seg_t = warp_segmentation_from_mvf(data_condition_seg_upsample, pred_mvf_voxel_set_t_upsample)
#             volume_t = warped_seg_t.sum(dim=(1,2,3,4))
#             volume_list.append(volume_t)
#         print('volume_t: ', volume_list)
#         volumes = torch.stack(volume_list, dim = 1)
#         v0 = volumes[:,0]
#         vmin = volumes.min(dim=1).values
#         EF_pred = (v0-vmin) / (v0 )
#         print('EF predicted: ', EF_pred)

#         # # use scipy
#         # pred_mvf_voxels_numpy = pred_mvf_voxels.detach().cpu().numpy()
#         # data_condition_seg_numpy = data_condition_seg.detach().cpu().numpy().squeeze()
#         # volume_list = []
#         # for t in range(10):
#         #     pred_mvf_voxel_set_t_numpy = np.transpose(pred_mvf_voxels_numpy[0, 3*t:3*(t+1),...],(1,2,3,0))
#         #     warped_seg_t_numpy =  Data_processing.apply_deformation_field_numpy(np.copy(data_condition_seg_numpy), pred_mvf_voxel_set_t_numpy, order = 0)
#         #     volume_t = np.sum(warped_seg_t_numpy == 1)
#         #     volume_list.append(volume_t)
#         # volume_list = np.asarray(volume_list)
#         # EF_pred_scipy = (volume_list[0] - np.min(volume_list)) / volume_list[0]
#         # print('EF predicted scipy: ', EF_pred_scipy)
#         # Data_processing.apply_deformation_field_numpy(np.copy(seg_img), mvf, order = 0)

#         pred_mvf = pred_mvf.detach().cpu().numpy().squeeze()
#         pred_mvf = Data_processing.normalize_image(pred_mvf, normalize_factor = 'equation', image_max = cutoff_max, image_min = cutoff_min, invert = True)
#         print(pred_mvf.shape, np.min(pred_mvf), np.max(pred_mvf))

#         # save
#         # if input_timeframe is None:
#         #     picked_tf = np.int16(np.reshape(batch_condition_tf.numpy().squeeze(),-1))
#         # else:
#         #     picked_tf = picked_tf
#         # print('picked_tf: ', picked_tf)

#         # for ii in range(len(picked_tf)):
#         #     segment_range = [3*ii, 3*(ii+1)]
#         #     mvf1 = pred_mvf[segment_range[0]:segment_range[1],...]; mvf1 = np.moveaxis(mvf1, 0, -1)
#         #     if self.generator.VAE_process == False:
#         #         mvf1 = zoom(mvf1, (4,4,4,1), order=1)
#         #     nb.save(nb.Nifti1Image(mvf1[:,:,:,0], affine), os.path.join(os.path.dirname(save_file), 'pred_tf'+str(picked_tf[ii])+'_x.nii.gz'))
#         #     nb.save(nb.Nifti1Image(mvf1[:,:,:,1], affine), os.path.join(os.path.dirname(save_file), 'pred_tf'+str(picked_tf[ii])+'_y.nii.gz'))
#         #     nb.save(nb.Nifti1Image(mvf1[:,:,:,2], affine), os.path.join(os.path.dirname(save_file), 'pred_tf'+str(picked_tf[ii])+'_z.nii.gz'))
#         #     if self.generator.VAE_process != False:
#         #         nb.save(nb.Nifti1Image(mvf1, affine), os.path.join(os.path.dirname(save_file), 'pred_latent_tf'+str(picked_tf[ii])+'.nii.gz'))
#         #     # else:
#         #     #     nb.save(nb.Nifti1Image(mvf1, affine), os.path.join(os.path.dirname(save_file), 'pred_tf'+str(picked_tf[ii])+'.nii.gz'))

#         #     if save_gt ==True:
#         #         if self.generator.VAE_process == False:
#         #             gt = os.path.join('/mnt/camca_NAS/4DCT/mvf_warp0_onecase',patient_class,patient_id, 'voxel_final', str(picked_tf[ii]) + '.nii.gz')
#         #         else:
#         #             gt = os.path.join('/mnt/camca_NAS/4DCT','models/VAE_embed3/pred_mvf',patient_class,patient_id, 'epoch100', 'pred_latent_tf'+str(picked_tf[ii])+'.nii.gz')
#         #         # print('gt range: ', nb.load(gt).get_fdata().shape, np.min(nb.load(gt).get_fdata()), np.max(nb.load(gt).get_fdata()))
#         #         gt = nb.load(gt).get_fdata()
#         #         gt = block_reduce(gt, (4,4,4,1), func=np.mean)
#         #         gt = zoom(gt, (4,4,4,1), order=1)
#         #         nb.save(nb.Nifti1Image(gt[:,:,:,0], affine), os.path.join(os.path.dirname(save_file), 'gt_tf'+str(picked_tf[ii])+'_x.nii.gz'))
#         #         nb.save(nb.Nifti1Image(gt[:,:,:,1], affine), os.path.join(os.path.dirname(save_file), 'gt_tf'+str(picked_tf[ii])+'_y.nii.gz'))
#         #         nb.save(nb.Nifti1Image(gt[:,:,:,2], affine), os.path.join(os.path.dirname(save_file), 'gt_tf'+str(picked_tf[ii])+'_z.nii.gz'))

#         return pred_mvf_saved

    
 

