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


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        EF_predictor,
        EF_loss_weight, 
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
        self.ef_predictor = EF_predictor
        self.EF_loss_weight = EF_loss_weight
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
        self.save_models_every = save_models_every

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

        # freeze EF predictor
        for p in self.ef_predictor.parameters():
            p.requires_grad = False

        if start_step is not None:
            self.step = start_step

        self.scheduler.step_size = 1
        val_loss = np.inf; val_diffusion_loss = np.inf; val_EF_loss = np.inf
        training_log = []
        
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
            print('11111 save_models_every: ', self.save_models_every, 'validation_every: ', self.validation_every)
            
            while self.step < self.train_num_steps:
                print('training epoch: ', self.step + 1)
                print('learning rate: ', self.scheduler.get_last_lr()[0])

                average_loss = []; average_diffusion_loss = []; average_EF_loss = []
                count = 0
                # load data
                for batch in self.dl:
                    self.opt.zero_grad()
                    batch_x0_image, _,_, batch_condition_image, batch_condition_EF, batch_condition_seg,_ = batch
                    data_x0 , data_condition_image, data_condition_EF, data_condition_seg = batch_x0_image.to(device), batch_condition_image.to(device), batch_condition_EF.to(device), batch_condition_seg.to(device)
                    
                    if not self.conditional_diffusion_image:
                        data_condition_image = None
                    if not self.conditional_diffusion_EF:
                        data_condition_EF = None
                    if not self.conditional_diffusion_seg:
                        data_condition_seg = None
                
                    with self.accelerator.autocast():
                        # factual generation
                        diffusion_loss,_,_ = self.model(data_x0,  condition_image = data_condition_image, condition_EF = data_condition_EF, condition_seg = data_condition_seg)
                        # print('diffusion_loss value: ', diffusion_loss.item())

                        # counterfactual generation
                        condition_EF_random = generate_random_ef(batch_size = data_condition_EF.shape[0], low_end = 0.05, high_end = 0.85, device = device)
                        # print('condition_EF_random: ', condition_EF_random)
                        _,denoised_image,_ = self.model(data_x0, condition_image = data_condition_image, condition_EF = condition_EF_random, condition_seg = data_condition_seg)
                        # print('denoised_image shape: ', denoised_image.shape)
                        # get EF prediction
                        EF_pred,_,_,_ = self.ef_predictor(denoised_image)
                        # print('EF_pred value: ', EF_pred)
                        # calculate mse loss in EF
                        EF_loss = F.mse_loss(EF_pred, condition_EF_random)
                        # print('EF loss: ', EF_loss.item())
                        loss = diffusion_loss + self.EF_loss_weight * EF_loss

                    average_loss.append(loss.item()); average_diffusion_loss.append(diffusion_loss.item()); average_EF_loss.append(EF_loss.item())

                    self.accelerator.backward(loss)
                    self.opt.step()

                average_loss = sum(average_loss) / len(average_loss); average_diffusion_loss = sum(average_diffusion_loss) / len(average_diffusion_loss); average_EF_loss = sum(average_EF_loss) / len(average_EF_loss)
                print('average loss: ', average_loss, 'average diffusion loss: ', average_diffusion_loss ,' average EF loss: ', average_EF_loss)
                pbar.set_description(f'average loss: {average_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
               
                self.step += 1

                # save the model
                if self.step !=0 and divisible_by(self.step, self.save_models_every):
                   self.save(self.step)
                
                if self.step !=0 and divisible_by(self.step, self.train_lr_decay_every):
                    self.scheduler.step()
                    
                self.ema.update()

                # do the validation if necessary
                if self.step !=0 and divisible_by(self.step, self.validation_every):
                    print('validation at step: ', self.step)
                    self.model.eval()
                    with torch.no_grad():
                        val_loss = []; val_diffusion_loss = []; val_EF_loss = []
                        for batch in self.dl_val:
                            batch_x0_image, _, batch_condition_tf_normalized, batch_condition_image, batch_condition_EF, batch_condition_seg,_ = batch
                            data_x0, data_condition_tf_normalized, data_condition_image, data_condition_EF, data_condition_seg = batch_x0_image.to(device), batch_condition_tf_normalized.to(device), batch_condition_image.to(device), batch_condition_EF.to(device), batch_condition_seg.to(device)

                            if not self.conditional_diffusion_image:
                                data_condition_image = None
                            if not self.conditional_diffusion_EF:
                                data_condition_EF = None
                            if not self.conditional_diffusion_seg:
                                data_condition_seg = None

                            with self.accelerator.autocast():
                                # factual generation
                                diffusion_loss,_,_ = self.model(data_x0,  condition_image = data_condition_image, condition_EF = data_condition_EF , condition_seg = data_condition_seg)

                                # counterfactual generation
                                condition_EF_random = generate_random_ef(batch_size = data_condition_EF.shape[0], low_end = 0.05, high_end = 0.85, device = device)
                                _,denoised_image, _ = self.model(data_x0, condition_image = data_condition_image, condition_EF = condition_EF_random, condition_seg = data_condition_seg)
                                # get EF prediction
                                EF_pred,_,_,_ = self.ef_predictor(denoised_image)
                                # calculate mse loss in EF
                                EF_loss = F.mse_loss(EF_pred, condition_EF_random)
                                loss = diffusion_loss + self.EF_loss_weight * EF_loss

                            val_loss.append(loss.item()); val_diffusion_loss.append(diffusion_loss.item()); val_EF_loss.append(EF_loss.item())
                        val_loss = sum(val_loss) / len(val_loss); val_diffusion_loss = sum(val_diffusion_loss) / len(val_diffusion_loss); val_EF_loss = sum(val_EF_loss) / len(val_EF_loss)
                        print('validation loss: ', val_loss, ' validation diffusion loss: ', val_diffusion_loss ,' validation EF loss: ', val_EF_loss)
                    self.model.train(True)

                # save the training log
                training_log.append([self.step, self.scheduler.get_last_lr()[0], average_loss, average_diffusion_loss, average_EF_loss, val_loss, val_diffusion_loss, val_EF_loss])
                df = pd.DataFrame(training_log,columns = ['iteration', 'learning_rate', 'average_loss', 'average_diffusion_loss', 'average_EF_loss', 'val_loss', 'val_diffusion_loss', 'val_EF_loss'])
                log_folder = os.path.join(os.path.dirname(self.results_folder),'log');ff.make_folder([log_folder])
                df.to_excel(os.path.join(log_folder, 'training_log.xlsx'),index=False)

                # at the end of each epoch, call on_epoch_end
                self.ds.on_epoch_end()
                self.ds_val.on_epoch_end()
                pbar.update(1)

        accelerator.print('training complete')
