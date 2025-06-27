from math import sqrt
from random import random
import torch
from torch import nn, einsum
import torch.nn.functional as F

from tqdm import tqdm
from einops import rearrange, repeat, reduce
from scipy.ndimage import zoom
from skimage.measure import block_reduce

from Cardiac4DCT_Synth_Diffusion.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_diffusion_3D import *
from Cardiac4DCT_Synth_Diffusion.denoising_diffusion_pytorch.denoising_diffusion_pytorch.version import __version__
from Cardiac4DCT_Synth_Diffusion.denoising_diffusion_pytorch.denoising_diffusion_pytorch.conditional_EDM_warp import *

import Cardiac4DCT_Synth_Diffusion.functions_collection as ff
import Cardiac4DCT_Synth_Diffusion.Data_processing as Data_processing

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


# main class
class EDM(nn.Module):  ### both 2D and 3D
    def __init__(
        self,
        model, 

        *,
        image_size,
        num_sample_steps, # number of sampling steps
        clip_or_not = None,
        clip_range = None,

        sigma_min = 0.002,     # min noise level
        sigma_max = 80,        # max noise level
        sigma_data = 0.5,      # standard deviation of data distribution
        rho = 7,               # controls the sampling schedule
        P_mean = -1.2,         # mean of log-normal distribution from which noise is drawn for training
        P_std = 1.2,           # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn = 80,          # parameters for stochastic sampling - depends on dataset, Table 5 in apper
        S_tmin = 0.05,
        S_tmax = 50,
        S_noise = 1.003,
    ):
        super().__init__()

        self.model = model
        self.conditional_diffusion_image = model.conditional_diffusion_image
        self.conditional_diffusion_EF = model.conditional_diffusion_EF
        self.conditional_diffusion_seg = model.conditional_diffusion_seg

        # image dimensions
        self.channels = self.model.channels
        
        self.image_size = image_size
        if len(image_size) == 2:
            self.image_size_h, self.image_size_w = image_size
        elif len(image_size) == 3:
          self.image_size_h, self.image_size_w, self.image_size_slice_num = image_size
        # parameters

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.rho = rho

        self.P_mean = P_mean
        self.P_std = P_std

        self.num_sample_steps = num_sample_steps  # otherwise known as N in the paper

        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

        # clip to stablize
        self.clip_or_not = clip_or_not
        self.clip_range = clip_range
        assert self.clip_or_not is not None, 'clip_or_not must be specified'

        if self.clip_or_not:
            self.maybe_clip = partial(torch.clamp, min = self.clip_range[0], max = self.clip_range[1]) 
        else:
            self.maybe_clip = identity

    @property
    def device(self):
        return next(self.model.parameters()).device

    # derived preconditioning params - Table 1

    def c_skip(self, sigma):
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return log(sigma) * 0.25

    # preconditioned network output
    # equation (7) in the paper

    def preconditioned_network_forward(self, noised_images, sigma,  condition_image = None, condition_EF = None, condition_seg = None):
        batch, device = noised_images.shape[0], noised_images.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device = device)

        if noised_images.dim() == 4: # 2D image (batch, channel, height, width)
            padded_sigma = rearrange(sigma, 'b -> b 1 1 1')
        elif noised_images.dim() == 5: # 3D image (batch, channel, depth, height, width)
            padded_sigma = rearrange(sigma, 'b -> b 1 1 1 1')

        if self.conditional_diffusion_image == True:
            if exists(condition_image) == 0:
                raise ValueError('conditional diffusion image is specified, but no condition is provided')
        if self.conditional_diffusion_EF == True:
            if exists(condition_EF) == 0:
                raise ValueError('conditional diffusion EF is specified, but no condition is provided')
        if self.conditional_diffusion_seg == True:
            if exists(condition_seg) == 0:
                raise ValueError('conditional diffusion segmentation is specified, but no condition is provided')

        net_out = self.model(self.c_in(padded_sigma) * noised_images, self.c_noise(sigma),  condition_image = condition_image, condition_EF = condition_EF, condition_seg = condition_seg)
        if isinstance(net_out, tuple):
            net_out, initial_input = net_out[0], net_out[1]
        else:
            net_out, initial_input = net_out, None
        
        out = self.c_skip(padded_sigma) * noised_images +  self.c_out(padded_sigma) * net_out

        out = self.maybe_clip(out)  # clip to stablize

        if initial_input is not None:
            return out, initial_input
        else:
            return out

    # sampling

    # sample schedule
    # equation (5) in the paper

    def sample_schedule(self, num_sample_steps = None):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device = self.device, dtype = torch.float32)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value = 0.) # last step is sigma value of 0.
        return sigmas

    @torch.no_grad()
    def sample(self, condition_image = None, condition_EF = None, condition_seg = None, batch_size = 16, num_sample_steps = None):
        # need to assert self.clip_or_not is True, otherwise the sampling will be unstable
        assert self.clip_or_not, 'clip_or_not must be True for sampling'

        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        if len(self.image_size) == 2:
            shape = (batch_size, self.channels, self.image_size_h, self.image_size_w)
        elif len(self.image_size) == 3:
            shape = (batch_size, self.channels, self.image_size_h, self.image_size_w,  self.image_size_slice_num)


        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sample_steps)
        # print('sigmas: ', sigmas.shape, sigmas)

        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, sqrt(2) - 1),
            0.)

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        # images is noise at the beginning

        init_sigma = sigmas[0]

        images = init_sigma * torch.randn(shape, device = self.device)

        # for self conditioning

        x_start = None

        # gradually denoise
        count = 1
        for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas, desc = 'sampling time step'):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            eps = self.S_noise * torch.randn(shape, device = self.device) # stochastic sampling

            sigma_hat = sigma + gamma * sigma
            images_hat = images + sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            model_output = self.preconditioned_network_forward(images_hat, sigma_hat, condition_image = condition_image, condition_EF = condition_EF, condition_seg = condition_seg)
            model_output = model_output[0] if isinstance(model_output, tuple) else model_output
        
            denoised_over_sigma = (images_hat - model_output) / sigma_hat

            images_next = images_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # second order correction, if not the last timestep

            if sigma_next != 0:
                model_output_next = self.preconditioned_network_forward(images_next, sigma_next, condition_image = condition_image, condition_EF = condition_EF, condition_seg = condition_seg)
                model_output_next = model_output_next[0] if isinstance(model_output_next, tuple) else model_output_next
                
                denoised_prime_over_sigma = (images_next - model_output_next) / sigma_next
                images_next = images_hat + 0.5 * (sigma_next - sigma_hat) * (denoised_over_sigma + denoised_prime_over_sigma)

            images = images_next
            x_start_tem = model_output_next if sigma_next != 0 else model_output  # temporary clean image
            count += 1

        images = self.maybe_clip(images)  # clip to stablize

        return images
    

    ####### training

    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size):
        return torch.normal(mean=self.P_mean, std=self.P_std, size=(batch_size,)).exp().to(self.device)
        # return (self.P_mean + self.P_std * torch.randn((batch_size,), device = self.device)).exp()

    def forward(self, images, condition_image = None, condition_EF = None, condition_seg = None):
        # assert image's min value is larger or equal to -1 and max value is smaller or equal to 1
        assert images.min() >= -1. and images.max() <= 1., 'image must be normalized to [-1, 1] range'

        if len(self.image_size) == 2:
            batch_size, c, h, w, device, image_size, channels = *images.shape, images.device, self.image_size, self.channels
        elif len(self.image_size) == 3:
            batch_size, c, h, w, slice_num, device, image_size, channels = *images.shape, images.device, self.image_size, self.channels

        assert c == channels, 'mismatch of image channels'

        # images = normalize_to_neg_one_to_one(images)

        sigmas = self.noise_distribution(batch_size)
        # print('sigmas: ', sigmas.shape, sigmas.min(), sigmas.max(), sigmas)
        if images.dim() == 4:
            padded_sigmas = rearrange(sigmas, 'b -> b 1 1 1')
        elif images.dim() == 5:
            padded_sigmas = rearrange(sigmas, 'b -> b 1 1 1 1')

        noise = torch.randn_like(images)

        noised_images = images + padded_sigmas * noise  # alphas are 1. in the paper

        denoised = self.preconditioned_network_forward(noised_images, sigmas,  condition_image = condition_image, condition_EF = condition_EF, condition_seg = condition_seg)
        if isinstance(denoised, tuple):
            denoised , initial_input = denoised[0], denoised[1]
        else:
            initial_input = torch.zeros_like(denoised)

        losses = F.mse_loss(denoised, images, reduction = 'none')
        losses = reduce(losses, 'b ... -> b', 'mean')

        losses = losses * self.loss_weight(sigmas)

        return losses.mean(), denoised, initial_input

    
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        generator_train,
        generator_val,
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
        self.conditional_diffusion_image = self.model.conditional_diffusion_image
        self.conditional_diffusion_EF = self.model.conditional_diffusion_EF
        self.conditional_diffusion_seg = self.model.conditional_diffusion_seg
        print('conditional_image: ', self.conditional_diffusion_image, ' condition_EF: ', self.conditional_diffusion_EF, ' condition_seg: ', self.conditional_diffusion_seg)
        self.channels = diffusion_model.channels

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
        self.scheduler = StepLR(self.opt, step_size = 1, gamma=0.95)
        self.train_lr_decay_every = train_lr_decay_every
        self.save_model_every = save_models_every
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
            'version': __version__}
        
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

                average_loss = []; average_diffusion_loss = []
                count = 0
                # load data
                for batch in self.dl:
                    self.opt.zero_grad()
                    batch_x0_image, _,_, batch_condition_image, batch_condition_EF, batch_condition_seg, _ = batch
                    data_x0, data_condition_image, data_condition_EF, data_condition_seg = batch_x0_image.to(device), batch_condition_image.to(device), batch_condition_EF.to(device), batch_condition_seg.to(device)
                    
                    if not self.conditional_diffusion_image:
                        data_condition_image = None
                    if not self.conditional_diffusion_EF:
                        data_condition_EF = None
                    if not self.conditional_diffusion_seg:
                        data_condition_seg = None

                    with self.accelerator.autocast():
                        diffusion_loss,_,initial_input = self.model(data_x0,  condition_image = data_condition_image, condition_EF = data_condition_EF, condition_seg = data_condition_seg)
                        loss = diffusion_loss

                    average_loss.append(loss.item())
                    average_diffusion_loss.append(diffusion_loss.item())

                    self.accelerator.backward(loss)
                    self.opt.step()

                average_loss = sum(average_loss) / len(average_loss)
                average_diffusion_loss = sum(average_diffusion_loss) / len(average_diffusion_loss)
                print('average loss: ', average_loss, 'average diffusion loss: ', average_diffusion_loss)
                pbar.set_description(f'average loss: {average_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
               
                self.step += 1

                # save the model
                if self.step !=0 and divisible_by(self.step, self.save_model_every):
                   self.save(self.step)
                
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
                            batch_x0_image, _,_, batch_condition_image, batch_condition_EF, batch_condition_seg,_ = batch
                            data_x0, data_condition_image, data_condition_EF, data_condition_seg = batch_x0_image.to(device), batch_condition_image.to(device), batch_condition_EF.to(device), batch_condition_seg.to(device)

                            if not self.conditional_diffusion_image:
                                data_condition_image = None
                            if not self.conditional_diffusion_EF:
                                data_condition_EF = None
                            if not self.conditional_diffusion_seg:
                                data_condition_seg = None

                            with self.accelerator.autocast():
                                loss,_,_ = self.model(data_x0, condition_image = data_condition_image, condition_EF = data_condition_EF , condition_seg = data_condition_seg)
                            val_loss.append(loss.item())
                        val_loss = sum(val_loss) / len(val_loss)
                        print('validation loss: ', val_loss)
                    self.model.train(True)

                # save the training log
                training_log.append([self.step, self.scheduler.get_last_lr()[0], average_loss, average_diffusion_loss, val_loss])
                df = pd.DataFrame(training_log,columns = ['iteration', 'learning_rate', 'average_loss', 'average_diffusion_loss', 'validation_loss'])
                log_folder = os.path.join(os.path.dirname(self.results_folder),'log');ff.make_folder([log_folder])
                df.to_excel(os.path.join(log_folder, 'training_log.xlsx'),index=False)

                # at the end of each epoch, call on_epoch_end
                self.ds.on_epoch_end()
                self.ds_val.on_epoch_end()
                pbar.update(1)

        accelerator.print('training complete')


# Sampling class
class Sampler(object):
    def __init__(
        self,
        diffusion_model,
        generator,
        batch_size,
        image_size = None,
        device = 'cuda',

    ):
        super().__init__()

        # model
        self.model = diffusion_model  
        if device == 'cuda':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == 'cpu':
            self.device = torch.device("cpu")

        self.conditional_diffusion_image = self.model.conditional_diffusion_image
        self.conditional_diffusion_EF = self.model.conditional_diffusion_EF
        self.conditional_diffusion_seg = self.model.conditional_diffusion_seg

        self.channels = diffusion_model.channels
        if image_size is None:
            self.image_size = self.model.image_size
        else:
            self.image_size = image_size
        self.batch_size = batch_size

        # dataset and dataloader

        self.generator = generator
        dl = DataLoader(self.generator, batch_size = self.batch_size, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())     
        self.how_many_timeframes_together = self.generator.how_many_timeframes_together

        self.dl = dl
        self.cycle_dl = cycle(dl)
 
        # EMA:
        self.ema = EMA(diffusion_model)
        self.ema.to(self.device)

    def load_model(self, trained_model_filename):

        data = torch.load(trained_model_filename, map_location=self.device)

        self.model.load_state_dict(data['model'])

        self.step = data['step']

        self.ema.load_state_dict(data["ema"])


    def sample_3D_w_trained_model(self, trained_model_filename, cutoff_max = None, cutoff_min = None,
                save_file = None,  patient_class = None, patient_id = None, image_file = None):
  
        self.load_model(trained_model_filename) 
        
        device = self.device

        self.ema.ema_model.eval()
        # check whether model is on GPU:
        # print('model device: ', next(self.ema.ema_model.parameters()).device)

        affine = nb.load(image_file).affine


        # start to run
        with torch.inference_mode():
            datas = next(self.cycle_dl)
            batch_x0_image, batch_condition_tf, batch_condition_tf_normalized, batch_condition_image, batch_condition_EF, batch_condition_seg, batch_condition_seg_orires = datas
            data_x0, data_condition_image, data_condition_EF, data_condition_seg , data_condition_seg_orires = batch_x0_image.to(device), batch_condition_image.to(device), batch_condition_EF.to(device), batch_condition_seg.to(device), batch_condition_seg_orires.to(device)

            if not self.conditional_diffusion_image: 
                data_condition_image = None
            if not self.conditional_diffusion_EF:
                data_condition_EF = None
            if not self.conditional_diffusion_seg:
                data_condition_seg = None
            
            print('data_condition_EF: ', data_condition_EF)
            pred_mvf = self.ema.ema_model.sample( condition_image = data_condition_image, condition_EF = data_condition_EF, condition_seg = data_condition_seg, batch_size = self.batch_size)
        
        pred_mvf_saved = torch.clone(pred_mvf)
        
        ##### apply to segmentation
        denormalize_mvf = DeNormalizeMVF(image_min=cutoff_min, image_max=cutoff_max)
        EF_pred, volumes = warp_loss(torch.clone(pred_mvf), data_condition_seg_orires, denormalize_mvf)
        print('EF predicted: ', EF_pred)

        pred_mvf = pred_mvf.detach().cpu().numpy().squeeze()
        pred_mvf = Data_processing.normalize_image(pred_mvf, normalize_factor = 'equation', image_max = cutoff_max, image_min = cutoff_min, invert = True)
        nb.save(nb.Nifti1Image(np.transpose(pred_mvf,(1,2,3,0)), affine), os.path.join(os.path.dirname(save_file), 'pred_mvf.nii.gz'))
        return pred_mvf_saved, EF_pred, pred_mvf

    
 


    
 
