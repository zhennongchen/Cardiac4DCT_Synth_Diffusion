import sys 
sys.path.append('/workspace/Documents')
from pathlib import Path 
import os
import numpy as np
import pandas as pd
import nibabel as nb
from tqdm.auto import tqdm
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from ema_pytorch import EMA
from accelerate import Accelerator
from Diffusion_motion_field.denoising_diffusion_pytorch.denoising_diffusion_pytorch.version import __version__

from Diffusion_motion_field.latent_diffusion.utils.conv_blocks import DownBlock, UpBlock, BasicBlock, BasicResBlock, UnetResBlock, UnetBasicBlock
from Diffusion_motion_field.latent_diffusion.loss.gan_losses import hinge_d_loss
from Diffusion_motion_field.latent_diffusion.loss.perceivers import LPIPS
# from medical_diffusion.models.model_base import BasicModel, VeryBasicModel

from pytorch_msssim import SSIM, ssim
import Diffusion_motion_field.functions_collection as ff
import Diffusion_motion_field.Data_processing as Data_processing

def cycle(dl):
    while True:
        for data in dl:
            yield data
def exists(x):
    return x is not None

def default(val, d):
    if exists(val): 
        return val
    return d() if callable(d) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0


class DiagonalGaussianDistribution(nn.Module):

    def forward(self, x):
        mean, logvar = torch.chunk(x, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        sample = torch.randn(mean.shape, generator=None, device=x.device)
        z = mean + std * sample

        batch_size = x.shape[0]
        var = torch.exp(logvar)
        kl = 0.5 * torch.sum(torch.pow(mean, 2) + var - 1.0 - logvar)/batch_size

        return z, kl 

class VAE(nn.Module):
    def __init__(
        self,
        in_channels= 1, 
        out_channels= 1, 
        spatial_dims = 3,
        emb_channels = 4,
        hid_chs =   [64,128,256,512], #[ 64, 128,  256, 512],
        kernel_sizes=[3,3,3,3], #[ 3,  3,   3,   3],
        strides =    [1,2,2,3],#[ 1,  2,   2,   2],
        norm_name = ("GROUP", {'num_groups':8, "affine": True}),
        act_name=("RELU", {}),#("Swish", {}),
        dropout=None,
        use_res_block=True,
        deep_supervision=False,
        learnable_interpolation=False,
        use_attention='none',
        embedding_loss_weight=1e-6,
        perceiver = LPIPS, 
        perceiver_kwargs = {},
        perceptual_loss_weight = 1.0,
        
        loss = torch.nn.L1Loss,
        loss_kwargs={'reduction': 'none'},

        sample_every_n_steps = 1000):

        super().__init__()
        self.sample_every_n_steps=sample_every_n_steps
        # self.loss_fct = loss(**loss_kwargs)
        self.ssim_fct = SSIM(data_range=1, size_average=False, channel=out_channels, spatial_dims=spatial_dims, nonnegative_ssim=True)
        self.embedding_loss_weight = embedding_loss_weight
        self.perceiver = perceiver(**perceiver_kwargs).eval() if perceiver is not None else None 
        self.perceptual_loss_weight = perceptual_loss_weight
        use_attention = use_attention if isinstance(use_attention, list) else [use_attention]*len(strides) 
        self.depth = len(strides)
        self.deep_supervision = deep_supervision
        downsample_kernel_sizes = kernel_sizes
        upsample_kernel_sizes = strides 

        # -------- Loss-Reg---------
        # self.logvar = nn.Parameter(torch.zeros(size=()) )

        # ----------- In-Convolution ------------
        ConvBlock = UnetResBlock if use_res_block else UnetBasicBlock
        self.inc = ConvBlock(
            spatial_dims, 
            in_channels, 
            hid_chs[0], 
            kernel_size=kernel_sizes[0], 
            stride=strides[0],
            act_name=act_name, 
            norm_name=norm_name,
            emb_channels=None
        )

        # ----------- Encoder ----------------
        self.encoders = nn.ModuleList([
            DownBlock(
                spatial_dims = spatial_dims, 
                in_channels = hid_chs[i-1], 
                out_channels = hid_chs[i], 
                kernel_size = kernel_sizes[i], 
                stride = strides[i],
                downsample_kernel_size = downsample_kernel_sizes[i],
                norm_name = norm_name,
                act_name = act_name,
                dropout = dropout,
                use_res_block = use_res_block,
                learnable_interpolation = learnable_interpolation,
                use_attention = use_attention[i],
                emb_channels = None
            )
            for i in range(1, self.depth)
        ])

        # ----------- Out-Encoder ------------
        self.out_enc = nn.Sequential(
            BasicBlock(spatial_dims, hid_chs[-1], 2*emb_channels, 3),
            BasicBlock(spatial_dims, 2*emb_channels, 2*emb_channels, 1)
        )

        # ----------- Reparameterization --------------
        self.quantizer = DiagonalGaussianDistribution()    


        # ----------- In-Decoder ------------
        self.inc_dec = ConvBlock(spatial_dims, emb_channels, hid_chs[-1], 3, act_name=act_name, norm_name=norm_name) 

        # ------------ Decoder ----------
        self.decoders = nn.ModuleList([
            UpBlock(
                spatial_dims = spatial_dims, 
                in_channels = hid_chs[i+1], 
                out_channels = hid_chs[i],
                kernel_size=kernel_sizes[i+1], 
                stride=strides[i+1], 
                upsample_kernel_size=upsample_kernel_sizes[i+1],
                norm_name=norm_name,  
                act_name=act_name, 
                dropout=dropout,
                use_res_block=use_res_block,
                learnable_interpolation=learnable_interpolation,
                use_attention=use_attention[i],
                emb_channels=None,
                skip_channels=0
            )
            for i in range(self.depth-1)
        ])

        # --------------- Out-Convolution ----------------
        self.outc = BasicBlock(spatial_dims, hid_chs[0], out_channels, 1, zero_conv=True)
        # if isinstance(deep_supervision, bool):
        #     deep_supervision = self.depth-1 if deep_supervision else 0 
        self.outc_ver = nn.ModuleList([
            BasicBlock(spatial_dims, hid_chs[i], out_channels, 1, zero_conv=True) 
            for i in range(1, deep_supervision+1)
        ])
        # self.logvar_ver = nn.ParameterList([
        #     nn.Parameter(torch.zeros(size=()) )
        #     for _ in range(1, deep_supervision+1)
        # ])

    
    def encode(self, x):
        h = self.inc(x)
        for i in range(len(self.encoders)):
            h = self.encoders[i](h)
        z = self.out_enc(h)
        z, _ = self.quantizer(z)
        return z 
            
    def decode(self, z):
        h = self.inc_dec(z)
        for i in range(len(self.decoders), 0, -1):
            h = self.decoders[i-1](h)
        x = self.outc(h)
        return x 

    def forward(self, x_in):
        # print('x_in shape:', x_in.shape)
        # --------- Encoder --------------
        h = self.inc(x_in)
        # print('after inc shape:', h.shape)
        for i in range(len(self.encoders)):
            h = self.encoders[i](h)
        # print('after encoders shape:', h.shape)
        z = self.out_enc(h)
        # print('after out_enc shape:', z.shape)

        # --------- Quantizer --------------
        z_q, emb_loss = self.quantizer(z)
        # print('after quantizer shape:', z_q.shape)

        # -------- Decoder -----------
        out_hor = []
        h = self.inc_dec(z_q)
        # print('after inc_dec shape:', h.shape)
        for i in range(len(self.decoders)-1, -1, -1):
            out_hor.append(self.outc_ver[i](h)) if i < len(self.outc_ver) else None 
            h = self.decoders[i](h)
        out = self.outc(h)
        # print('out shape:', out.shape)

        return out, out_hor[::-1], emb_loss 
    
    def perception_loss(self, pred, target, depth=0):
        if (self.perceiver is not None) and (depth<2):
            self.perceiver.eval()
            return self.perceiver(pred, target)*self.perceptual_loss_weight
        else:
            return 0 
    
    def ssim_loss(self, pred, target):
        return 1-ssim(((pred+1)/2).clamp(0,1), (target.type(pred.dtype)+1)/2, data_range=1, size_average=False, 
                        nonnegative_ssim=True).reshape(-1, *[1]*(pred.ndim-1))
    
    def rec_loss(self, pred, pred_vertical, target):
        interpolation_mode = 'nearest-exact'

        # Loss
        loss = 0
        l1_loss = F.l1_loss(pred, target)
        perception_loss = self.perception_loss(pred, target)
        ssim_loss = self.ssim_loss(pred, target)
        rec_loss = l1_loss + perception_loss + ssim_loss
        # print('l1 loss, perception loss, ssim loss, rec_loss:', torch.sum(l1_loss), torch.sum(perception_loss), torch.sum(ssim_loss), torch.sum(rec_loss))
        loss += torch.sum(rec_loss)/pred.shape[0]  
        
        for i, pred_i in enumerate(pred_vertical):
            # print('in vertical!!!')
            target_i = F.interpolate(target, size=pred_i.shape[2:], mode=interpolation_mode, align_corners=None)  
            l1_loss_i = F.l1_loss(pred_i, target_i, reduction = 'none')
            perception_loss_i = self.perception_loss(pred_i, target_i)
            ssim_loss_i = self.ssim_loss(pred_i, target_i)
            rec_loss_i = l1_loss_i + perception_loss_i + ssim_loss_i
            # print('in vertical, l1 loss, perception loss, ssim loss, rec_loss:', torch.sum(l1_loss_i), torch.sum(perception_loss_i), torch.sum(ssim_loss_i), torch.sum(rec_loss_i))
            loss += torch.sum(rec_loss_i)/pred.shape[0]  

        return loss 

    # def _step(self, batch: dict, batch_idx: int, state: str, step: int, optimizer_idx:int):
    #     # ------------------------- Get Source/Target ---------------------------
    #     x = batch['target']
    #     target = x

    #     # ------------------------- Run Model ---------------------------
    #     pred, pred_vertical, emb_loss = self(x)

    #     # ------------------------- Compute Loss ---------------------------
    #     loss = self.rec_loss(pred, pred_vertical, target)
    #     loss += emb_loss*self.embedding_loss_weight
         
    #     # --------------------- Compute Metrics  -------------------------------
    #     with torch.no_grad():
    #         logging_dict = {'loss':loss, 'emb_loss': emb_loss}
    #         logging_dict['L2'] = torch.nn.functional.mse_loss(pred, target)
    #         logging_dict['L1'] = torch.nn.functional.l1_loss(pred, target)
    #         logging_dict['ssim'] = ssim((pred+1)/2, (target.type(pred.dtype)+1)/2, data_range=1)
    #         # logging_dict['logvar'] = self.logvar

    #     # ----------------- Log Scalars ----------------------
    #     for metric_name, metric_val in logging_dict.items():
    #         self.log(f"{state}/{metric_name}", metric_val, batch_size=x.shape[0], on_step=True, on_epoch=True)     

    #     # ----------------- Save Image ------------------------------
    #     if self.global_step != 0 and self.global_step % self.sample_every_n_steps == 0:
    #         log_step = self.global_step // self.sample_every_n_steps
    #         path_out = Path(self.logger.log_dir)/'images'
    #         path_out.mkdir(parents=True, exist_ok=True)
    #         # for 3D images use depth as batch :[D, C, H, W], never show more than 16+16 =32 images 
    #         def depth2batch(image):
    #             return (image if image.ndim<5 else torch.swapaxes(image[0], 0, 1))
    #         images = torch.cat([depth2batch(img)[:16] for img in (x, pred)]) 
    #         save_image(images, path_out/f'sample_{log_step}.png', nrow=x.shape[0], normalize=True)
    
    #     return loss
    

# trainer class
class Trainer(object):
    def __init__(
        self,
        vae_model,
        generator_train,
        generator_val,
        train_batch_size,
        *,
        train_num_steps = 100000, # total training epochs
        results_folder = None,
        train_lr = 1e-4,
        train_lr_decay_every = 200, 
        save_models_every = 1,
        validation_every = 1,
        
        ema_update_every = 10,
        ema_decay = 0.995,
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
        self.model = vae_model   # it's not just the model architecture, but the actual model with loss calculation

        self.batch_size = train_batch_size
        self.train_num_steps = train_num_steps

        # dataset and dataloader
        self.ds = generator_train
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())
        self.dl = self.accelerator.prepare(dl)
        self.cycle_dl = cycle(dl)

        self.ds_val = generator_val
        dl_val = DataLoader(self.ds_val, batch_size = train_batch_size, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())
        self.dl_val = self.accelerator.prepare(dl_val)

        # optimizer
        self.opt = Adam(vae_model.parameters(), lr = train_lr, betas = adam_betas)
        self.scheduler = StepLR(self.opt, step_size = 1, gamma=0.95)
        self.max_grad_norm = max_grad_norm
        self.train_lr_decay_every = train_lr_decay_every
        self.save_model_every = save_models_every

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(vae_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = results_folder
        ff.make_folder([self.results_folder])

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        self.validation_every = validation_every

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
            'version': __version__,
        }
        
        torch.save(data, os.path.join(self.results_folder, 'model-' + str(stepNum) + '.pt'))

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
        val_loss = np.inf
        training_log = []
        
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
            
            while self.step < self.train_num_steps:
                print('training epoch: ', self.step + 1)
                print('learning rate: ', self.scheduler.get_last_lr()[0])

                average_loss = []
                count = 0

                for batch in self.dl:
                    self.opt.zero_grad()
                    batch_x0,_ = batch
                    target = torch.clone(batch_x0)

                    x = batch_x0.to(device)
                    target = target.to(device)

                    with self.accelerator.autocast():
                        pred, pred_vertical, emb_loss = self.model(x)
                    # print('max and min for x and pred:', torch.max(x), torch.min(x), torch.max(pred), torch.min(pred))

                    loss = self.model.rec_loss(pred, pred_vertical, target)
                    loss += emb_loss*self.model.embedding_loss_weight
                    
                    average_loss.append(loss.item())
                    count += 1

                    self.accelerator.backward(loss)
                    self.opt.step() 

                average_loss = sum(average_loss) / len(average_loss)
                pbar.set_description(f'average loss: {average_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.step += 1

                # save the model
                if self.step !=0 and divisible_by(self.step, self.save_model_every):
                   self.save(self.step)
                # update the parameter
                if self.step !=0 and divisible_by(self.step, self.train_lr_decay_every):
                    self.scheduler.step()

                self.ema.update()

                # do the validation if necessary
                if self.step !=0 and divisible_by(self.step, self.validation_every):
                    print('validation at step: ', self.step)
                    self.model.eval()
                    with torch.no_grad():
                        val_loss = []
                        for batch in self.dl_val:
                            batch_x0,_ = batch
                            target = torch.clone(batch_x0)

                            x = batch_x0.to(device)
                            target = target.to(device)

                            with self.accelerator.autocast():
                                pred, pred_vertical, emb_loss = self.model(x)

                            loss = self.model.rec_loss(pred, pred_vertical, target)
                            loss += emb_loss*self.model.embedding_loss_weight

                            val_loss.append(loss.item())
                        val_loss = sum(val_loss) / len(val_loss)
                        print('validation loss: ', val_loss)
                    self.model.train(True)

                # save the training log
                training_log.append([self.step,average_loss, self.scheduler.get_last_lr()[0], val_loss])
                df = pd.DataFrame(training_log,columns = ['iteration','training_loss','learning_rate', 'validation_loss'])
                log_folder = os.path.join(os.path.dirname(self.results_folder),'log');ff.make_folder([log_folder])
                df.to_excel(os.path.join(log_folder, 'training_log.xlsx'),index=False)

                # at the end of each epoch, call on_epoch_end
                self.ds.on_epoch_end(); self.ds_val.on_epoch_end()
                pbar.update(1)

        accelerator.print('training complete')



# Sampling class
class Predict(object):
    def __init__(
        self,
        vae_model,
        generator,
        batch_size,
        device = 'cuda',
    ):
        super().__init__()

        # model
        self.model = vae_model
        if device == 'cuda':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == 'cpu':
            self.device = torch.device("cpu")

        self.batch_size = batch_size

        # dataset and dataloader
        self.generator = generator
        dl = DataLoader(self.generator, batch_size = self.batch_size, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())

        self.dl = dl
        self.cycle_dl = cycle(dl)
 
        # EMA:
        self.ema = EMA(vae_model)
        self.ema.to(self.device)  

    def load_model(self, trained_model_filename):

        data = torch.load(trained_model_filename, map_location=self.device)

        self.model.load_state_dict(data['model'])

        self.step = data['step']

        self.ema.load_state_dict(data["ema"])

    def sample_3D_w_trained_model(self, trained_model_filename, affine,save_file, save_latent = False, save_decode = False, only_do_decode = False, save_gt = True):
     
        self.load_model(trained_model_filename) 
        
        device = self.device

        self.ema.ema_model.eval()

        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # start to run
        with torch.inference_mode():
            datas = next(self.cycle_dl)
            if only_do_decode == False:
                data_x0,picked_tf = datas
            else:
                data_x0,picked_tf,latent_space = datas
    
            # save data_x0
            data_x0_save = np.moveaxis(torch.clone(data_x0).numpy().squeeze(),0,-1)
            data_x0_save = Data_processing.normalize_image(data_x0_save, normalize_factor = 'equation', image_max = 20, image_min = -20, invert = True)
            if save_gt == True:
                nb.save(nb.Nifti1Image(data_x0_save[:,:,:,0], affine), os.path.join(os.path.dirname(save_file), 'gt_mvf_tf'+str(picked_tf.numpy()[0])+'_x.nii.gz'))
                nb.save(nb.Nifti1Image(data_x0_save[:,:,:,1], affine), os.path.join(os.path.dirname(save_file), 'gt_mvf_tf'+str(picked_tf.numpy()[0])+'_y.nii.gz'))
                nb.save(nb.Nifti1Image(data_x0_save[:,:,:,2], affine), os.path.join(os.path.dirname(save_file), 'gt_mvf_tf'+str(picked_tf.numpy()[0])+'_z.nii.gz'))

            x = data_x0.to(device)
            if only_do_decode == False:
                # two step
                # do the encoder
                pred_latent = self.ema.ema_model.encode(x)
                # print('latent shape:', pred_latent.shape, ' max and min:', torch.max(pred_latent), torch.min(pred_latent))
                # do the decoder
                pred_decode = self.ema.ema_model.decode(pred_latent)

                # one-step
                pred_img, _, _ = self.ema.ema_model(x)
            else:
                # only decode
                latent_space = latent_space.to(device) 
                pred_decode_direct = self.ema.ema_model.decode(latent_space)

        # save 
        if only_do_decode == False:           
            pred_img = pred_img.detach().cpu().numpy().squeeze()
            pred_img = Data_processing.normalize_image(pred_img, normalize_factor = 'equation', image_max = 20, image_min = -20, invert = True)
            mvf = np.copy(pred_img); mvf = np.moveaxis(mvf, 0, -1)

            pred_decode = pred_decode.detach().cpu().numpy().squeeze()
            pred_decode = Data_processing.normalize_image(pred_decode, normalize_factor = 'equation', image_max = 20, image_min = -20, invert = True)
            decode = np.copy(pred_decode); decode = np.moveaxis(decode, 0, -1)
        else:
            pred_decode_direct = pred_decode_direct.detach().cpu().numpy().squeeze()
            pred_decode_direct = Data_processing.normalize_image(pred_decode_direct, normalize_factor = 'equation', image_max = 20, image_min = -20, invert = True)
            decode_direct = np.copy(pred_decode_direct); decode_direct = np.moveaxis(decode_direct, 0, -1)
        
        if only_do_decode == False:
            # nb.save(nb.Nifti1Image(mvf[:,:,:,0], affine), os.path.join(os.path.dirname(save_file), 'pred_tf'+str(picked_tf.numpy()[0])+'_x.nii.gz'))
            # nb.save(nb.Nifti1Image(mvf[:,:,:,1], affine), os.path.join(os.path.dirname(save_file), 'pred_tf'+str(picked_tf.numpy()[0])+'_y.nii.gz'))
            # nb.save(nb.Nifti1Image(mvf[:,:,:,2], affine), os.path.join(os.path.dirname(save_file), 'pred_tf'+str(picked_tf.numpy()[0])+'_z.nii.gz'))

            if save_decode:
                nb.save(nb.Nifti1Image(decode[:,:,:,0], affine), os.path.join(os.path.dirname(save_file), 'pred_decode_tf'+str(picked_tf.numpy()[0])+'_x.nii.gz'))
                nb.save(nb.Nifti1Image(decode[:,:,:,1], affine), os.path.join(os.path.dirname(save_file), 'pred_decode_tf'+str(picked_tf.numpy()[0])+'_y.nii.gz'))
                nb.save(nb.Nifti1Image(decode[:,:,:,2], affine), os.path.join(os.path.dirname(save_file), 'pred_decode_tf'+str(picked_tf.numpy()[0])+'_z.nii.gz'))
            
            if save_latent:
                latent = np.copy(pred_latent.detach().cpu().numpy().squeeze())
                latent = np.moveaxis(latent, 0, -1)
                nb.save(nb.Nifti1Image(latent, affine), os.path.join(os.path.dirname(save_file), 'pred_latent_tf'+str(picked_tf.numpy()[0])+'.nii.gz'))
        else:
            nb.save(nb.Nifti1Image(decode_direct[:,:,:,0], affine), os.path.join(os.path.dirname(save_file), 'pred_decode_direct_tf'+str(picked_tf.numpy()[0])+'_x.nii.gz'))
            nb.save(nb.Nifti1Image(decode_direct[:,:,:,1], affine), os.path.join(os.path.dirname(save_file), 'pred_decode_direct_tf'+str(picked_tf.numpy()[0])+'_y.nii.gz'))
            nb.save(nb.Nifti1Image(decode_direct[:,:,:,2], affine), os.path.join(os.path.dirname(save_file), 'pred_decode_direct_tf'+str(picked_tf.numpy()[0])+'_z.nii.gz'))
            nb.save(nb.Nifti1Image(decode_direct, affine), os.path.join(os.path.dirname(save_file), 'pred_decode_direct_tf'+str(picked_tf.numpy()[0])+'.nii.gz'))

        if only_do_decode:
            pred_measure = decode_direct
        else:
            pred_measure = decode
        x_mae = np.mean(np.abs(data_x0_save[:,:,:,0] - pred_measure[:,:,:,0]))
        y_mae = np.mean(np.abs(data_x0_save[:,:,:,1] - pred_measure[:,:,:,1]))
        z_mae = np.mean(np.abs(data_x0_save[:,:,:,2] - pred_measure[:,:,:,2]))
        all_mae = np.mean(np.abs(data_x0_save - pred_measure))
        return all_mae, x_mae, y_mae, z_mae
    
