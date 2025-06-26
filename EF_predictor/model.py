import math
import copy
import os
import pandas as pd
import numpy as np
import nibabel as nb
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
from skimage.measure import block_reduce

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from Diffusion_motion_field.EF_predictor.attend import Attend
import Diffusion_motion_field.functions_collection as ff
import Diffusion_motion_field.Data_processing as Data_processing


# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val): 
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr



def Upsample3D(dim, dim_out = None, upsample_factor = (2,2,2)):
    return nn.Sequential(
        nn.Upsample(scale_factor = upsample_factor, mode = 'nearest'),
        nn.Conv3d(dim, default(dim_out, dim), 3, padding = 1)
    )


def Downsample3D(dim, dim_out = None):
    return nn.Sequential(
        nn.MaxPool3d(kernel_size=(2,2, 2), stride=(2,2, 2), padding=0),
        nn.Conv3d(dim, default(dim_out, dim), 1)
    )

class RMSNorm(nn.Module):
    '''RMSNorm applies channel-wise normalization to the input tensor, 
    scales the normalized values using the learnable parameter g, 
    and then further scales the result by the square root of the number of input channels. '''
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1)) # learnable

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)


class RMSNorm3D(nn.Module):
    '''RMSNorm applies channel-wise normalization to the input tensor, 
    scales the normalized values using the learnable parameter g, 
    and then further scales the result by the square root of the number of input channels. '''
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1 , 1)) # learnable

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

# building block modules
class ConvBlock1D(nn.Module):  # input dimension is dim, output dimension is dim_out
    def __init__(self, dim, dim_out, groups=8, dilation=None, act='ReLU'):
        super().__init__()
        if dilation is None:
            self.conv = nn.Conv1d(dim, dim_out, kernel_size=3, padding=1)
        else:
            self.conv = nn.Conv1d(dim, dim_out, kernel_size=3, padding=dilation, dilation=dilation)

        self.norm = nn.GroupNorm(groups, dim_out)

        if act == 'ReLU':
            self.act = nn.ReLU()
        elif act == 'LeakyReLU':
            self.act = nn.LeakyReLU()
        else:
            raise ValueError('activation function not supported')

    def forward(self, x):
        # x: [B, C, T]
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ConvBlock3D(nn.Module):  # input dimension is dim, output dimension is dim_out
    def __init__(self, dim, dim_out, groups = 8, dilation = None, act = 'ReLU'):
        super().__init__()
        if dilation == None:
            self.conv = nn.Conv3d(dim, dim_out, 3, padding = 1)
        else:
            self.conv = nn.Conv3d(dim, dim_out, 3, padding = dilation, dilation = dilation)
        self.norm = nn.GroupNorm(groups, dim_out) 
        if act == 'ReLU':
            self.act = nn.ReLU()
        elif act == 'LeakyReLU':
            self.act = nn.LeakyReLU()
        else:
            raise ValueError('activation function not supported')

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

# Attention:
class LinearAttention3D(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm3D(dim)
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv3d(hidden_dim, dim, 1),
            RMSNorm3D(dim)
        )

    def forward(self, x):
        b, c, h, w, d = x.shape  # Added dimension 'd' for depth

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)  # split input into q k v evenly
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z -> b h c (x y z)', h=self.heads), qkv)  # h = head, c = dim_head

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)  # matrix multiplication

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y z) -> b (h c) x y z', h = self.heads, x = h, y = w, z = d)
        return self.to_out(out)


class Attention3D(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm3D(dim)
        self.attend = Attend(flash = False)

        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w, d = x.shape  # Added dimension 'd' for depth

        x = self.norm(x) 

        qkv = self.to_qkv(x).chunk(3, dim=1)  # split input into q k v evenly
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z-> b h (x y z) c', h = self.heads), qkv)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y z) d -> b (h d) x y z', x = h, y = w, z = d)
        return self.to_out(out)
    
    
    
class ResnetBlock3D(nn.Module): 
    # conv + conv + attention + residual
    def __init__(self, dim, dim_out, groups = 8, use_full_attention = None, attn_head = 4, attn_dim_head = 32 , act = 'ReLU'):
        '''usee which attention: 'Full' or 'Linear'''
        super().__init__()
    
        self.block1 = ConvBlock3D(dim, dim_out, groups = groups, act = act)
        self.block2 = ConvBlock3D(dim_out, dim_out, groups = groups, act = act)
        
        if use_full_attention == True:
            self.attention = Attention3D(dim_out, heads = attn_head, dim_head = attn_dim_head)
        elif use_full_attention == False:
            self.attention = LinearAttention3D(dim_out, heads = attn_head, dim_head = attn_dim_head)
        else:
            self.attention = nn.Identity()

        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):

        h = self.block1(x) 

        h = self.block2(h)

        h = self.attention(h)

        return h + self.res_conv(x)
    

# model: Spatial CNN + LSTM
class CNN_EF_predictor_LSTM(nn.Module):
    def __init__(
        self,
        init_dim = 16,
        channels = 1,
        dim_mults = (2,4,8),
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = (None, None,None),
        act = 'ReLU',
        lstm_hidden_dim = 128,
    ):
        super().__init__()
    
        self.channels = channels
        input_channels = channels

        self.init_conv = nn.Conv3d(input_channels, init_dim, 3, padding = 1) # if want input and output to have same dimension, Kernel size to any odd number (e.g., 3, 5, 7, etc.). Padding to (kernel size - 1) / 2.

        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]  # if initi_dim = 16, then [16, 32, 64, 128, 256]

        in_out = list(zip(dims[:-1], dims[1:])) 
        print('in out is : ', in_out)
        # [(16,32), (32,64), (64,128), (128,256)]. Each tuple in in_out represents a pair of input and output dimensions for different stages in a neural network 


        # attention
        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out) # 4

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            self.downs.append(nn.ModuleList([
                ResnetBlock3D(dim_in, dim_in, use_full_attention = layer_full_attn, attn_head = layer_attn_heads, attn_dim_head = layer_attn_dim_head, act = act),
                Downsample3D(dim_in, dim_out)
            ]))

        mid_dim = dims[-1]
        self.mid_block = ResnetBlock3D(mid_dim, mid_dim, use_full_attention = False, attn_head = attn_heads[-1], attn_dim_head = attn_dim_head[-1], act = act)

        self.lstm = nn.LSTM(input_size=3 * mid_dim, hidden_size=lstm_hidden_dim, num_layers=1, batch_first=True)
        self.regressor = nn.Linear(lstm_hidden_dim, 1)

        

    def forward(self, x):
        # initial dimen = [batch, 30, 40,40,24]
        B = x.shape[0]
        # reshape
        x = rearrange(x, 'b t h w c -> (b t) 1 h w c')
        x = self.init_conv(x)

        x_init = x.clone()

        h = []
        for block, downsample in self.downs:
            x = block(x)
            h.append(x)

            x = downsample(x)
            # print('after this block, x shape is: ', x.shape)
        
        encode_x = self.mid_block(x) # [batch* 30, C, 5, 5, 3]
        C = encode_x.shape[1]

        # Reshape for LSTM
        x2 = encode_x.view(B, 10, 3, C, 5, 5, 3)          # [B, 10, 3, C, X, Y, Z]
        # print('x shape before permute is: ', x2.shape)
        x2 = x2.permute(0, 4, 5, 6, 1, 2, 3).contiguous()        # [B, X, Y, Z, T, 3, C]
        # print('x shape after permute is: ', x2.shape)
        x2 = x2.reshape(B * 5 * 5 * 3, 10, 3 * C)   # [N_voxel, T, 3C]
        # print('x shape after reshape is: ', x2.shape)

        out, (h_n, _) = self.lstm(x2)             # h_n[-1]: [N_voxel, H]
        lstm_output = h_n[-1]
        feature_vector = lstm_output.view(B, 5, 5, 3, -1).mean(dim=(1, 2, 3))  # [B, H]
        # print('x shape after LSTM is: ', feature_vector.shape)

        ef = self.regressor(feature_vector)                   # [B, 1]
        # print('ef shape is: ', ef.shape)

        # print('ef, encode_x, lstm_output, feature_vector shape is: ', ef.shape, encode_x.shape, lstm_output.shape, feature_vector.shape)

        return ef, encode_x, lstm_output, feature_vector
    
# model: spatial CNN + Temporal cov
class CNN_EF_predictor_temporalConv(nn.Module):
    def __init__(
        self,
        init_dim = 16,
        channels = 1,
        dim_mults = (2,4,8),
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = (None, None,None),
        act = 'ReLU',
    ):
        super().__init__()
    
        self.channels = channels
        input_channels = channels

        self.init_conv = nn.Conv3d(input_channels, init_dim, 3, padding = 1) # if want input and output to have same dimension, Kernel size to any odd number (e.g., 3, 5, 7, etc.). Padding to (kernel size - 1) / 2.

        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]  # if initi_dim = 16, then [16, 32, 64, 128, 256]

        in_out = list(zip(dims[:-1], dims[1:])) 
        print('in out is : ', in_out)
        # [(16,32), (32,64), (64,128), (128,256)]. Each tuple in in_out represents a pair of input and output dimensions for different stages in a neural network 


        # attention
        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out) # 4

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            self.downs.append(nn.ModuleList([
                ResnetBlock3D(dim_in, dim_in, use_full_attention = layer_full_attn, attn_head = layer_attn_heads, attn_dim_head = layer_attn_dim_head, act = act),
                ConvBlock1D(dim_in, dim_in, act = act),
                Downsample3D(dim_in, dim_out)
            ]))

        mid_dim = dims[-1]
        self.mid_block = ResnetBlock3D(mid_dim, mid_dim*2, use_full_attention = False, attn_head = attn_heads[-1], attn_dim_head = attn_dim_head[-1], act = act)
        self.mid_temporal = ConvBlock1D(mid_dim*2, mid_dim*2, act = act)

        # self.lstm = nn.LSTM(input_size=3 * mid_dim, hidden_size=lstm_hidden_dim, num_layers=1, batch_first=True)
        self.regressor = nn.Linear(mid_dim*2, 1)

        

    def forward(self, x):
        # initial dimen = [batch, 30, 40,40,24]
        B, T = x.shape[0] , 10
        # reshape
        x = rearrange(x, 'b t h w c -> (b t) 1 h w c')
        x = self.init_conv(x)

        h = []
        for spatial_block, temporal_block, downsample in self.downs:
            x = spatial_block(x)
            C, x_dim, y_dim, z_dim = x.shape[1], x.shape[2], x.shape[3], x.shape[4]

            # Temporal conv prep: reshape to [B*3*X*Y*Z, C, T]
            x = x.view(B, T, 3, C, x_dim, y_dim, z_dim)  # [B, 10, 3, C, X, Y, Z]
            x = x.permute(0, 2, 4, 5, 6, 3, 1).contiguous().reshape(B * 3 * x_dim * y_dim * z_dim, C, T)

            # Temporal conv (Conv1D along time)
            x = temporal_block(x)

            # Restore shape back to [B * T * 3, C, X, Y, Z]
            x = x.view(B, 3, x_dim, y_dim, z_dim, C, T).permute(0, 6, 1, 5, 2, 3, 4).contiguous()
            x = x.reshape(B * T * 3, C, x_dim, y_dim, z_dim)

            x = downsample(x)
    
        x = self.mid_block(x) 
        C, x_dim, y_dim, z_dim = x.shape[1], x.shape[2], x.shape[3], x.shape[4]

        # Temporal conv prep: reshape to [B*3*X*Y*Z, C, T]
        x = x.view(B, T, 3, C, x_dim, y_dim, z_dim)  
        x = x.permute(0, 2, 4, 5, 6, 3, 1).contiguous().reshape(B * 3 * x_dim * y_dim * z_dim, C, T)

        # Temporal conv (Conv1D along time)
        x = self.mid_temporal(x)
        # Restore shape back to [B * T * 3, C, X, Y, Z]
        encode_x_before_mean = x.reshape(B, 3, x_dim, y_dim, z_dim, C, T).permute(0, 6, 1, 5, 2, 3, 4).contiguous()
        encode_x_before_mean = encode_x_before_mean.resahpe(B,T,3,C, -1)
    
        encode_x = encode_x_before_mean.mean(dim = [2,4])
        encode_x = encode_x.mean(dim = 1)
        ef = self.regressor(encode_x)
        # print('encode_x shape is: ', encode_x.shape, ' ef shape is: ', ef.shape)
       
        return ef, encode_x, encode_x_before_mean, 1


class Trainer(object):
    def __init__(
        self,
        model,
        generator_train,
        generator_val,
        train_batch_size,

        *,
        accum_iter = 10, # gradient accumulation steps
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
        self.model = model  

        # sampling and training hyperparameters
        self.batch_size = train_batch_size

        self.accum_iter = accum_iter

        self.train_num_steps = train_num_steps

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader
        self.ds = generator_train
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())
        self.dl = self.accelerator.prepare(dl)

        self.ds_val = generator_val
        dl_val = DataLoader(self.ds_val, batch_size = train_batch_size, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())
        self.dl_val = self.accelerator.prepare(dl_val)

        # optimizer
        self.opt = Adam(model.parameters(), lr = train_lr, betas = adam_betas)
        self.scheduler = StepLR(self.opt, step_size = 1, gamma=0.95)
        self.train_lr_decay_every = train_lr_decay_every
        self.save_model_every = save_models_every
        self.validation_every = validation_every

        if self.accelerator.is_main_process:
            self.ema = EMA(model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = results_folder
    
        ff.make_folder([self.results_folder])

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)


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

        training_log = []

        # load pre-trained
        if pre_trained_model is not None:
            self.load_model(pre_trained_model)
            print('model loaded from ', pre_trained_model)

        if start_step is not None:
            self.step = start_step

        val_loss = np.inf
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
            
            while self.step < self.train_num_steps:
                print('training epoch: ', self.step + 1)
                print('learning rate: ', self.scheduler.get_last_lr()[0])

                average_loss = []
                count = 0
                # load data
                for batch in self.dl:
                    if count == 0 or count % self.accum_iter == 0 or count == len(self.dl) - 1 or count == len(self.dl):
                        self.opt.zero_grad()
         
                    # load data
                    batch_img, _,_, batch_EF = batch
                    data_input = batch_img.to(device)
                    gt_EF = batch_EF.to(device)

                    with self.accelerator.autocast():
                        pred_EF,_,_,_ = self.model(data_input)
                        loss = F.mse_loss(pred_EF, gt_EF)
                    
                      
                    # accumulate the gradient, typically used when batch size is small
                    if count % self.accum_iter == 0 or count == len(self.dl) - 1 or count == len(self.dl):
                        self.accelerator.backward(loss)
                        accelerator.wait_for_everyone()
                        accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.opt.step()

                    count += 1
                    average_loss.append(loss.item())
                            
                average_loss = sum(average_loss) / len(average_loss)
                print('average loss is: ', average_loss)
               
                accelerator.wait_for_everyone()
                self.step += 1

                # save the model
                if self.step !=0 and divisible_by(self.step, self.save_model_every):
                   self.save(self.step)
                
                if self.step !=0 and divisible_by(self.step, self.train_lr_decay_every):
                    self.scheduler.step()
                    
                self.ema.update()

                # do the validation if necessary
                if self.step !=0 and divisible_by(self.step, self.validation_every):
                    self.model.eval()
                    with torch.no_grad():
                        val_loss = []
                        for batch in self.dl_val:
                            batch_img, _,_, batch_EF = batch
                            data_input = batch_img.to(device)
                            gt_EF = batch_EF.to(device)

                            with self.accelerator.autocast():
                                pred_EF,_,_,_ = self.model(data_input)
                                loss = F.mse_loss(pred_EF, gt_EF)
                            
                            val_loss.append(loss.item())
                        val_loss = sum(val_loss) / len(val_loss)
                        print('validation loss is: ', val_loss)
                    self.model.train(True)

                # save the training log
                training_log.append([self.step,self.scheduler.get_last_lr()[0], average_loss, val_loss])
                df = pd.DataFrame(training_log,columns = ['iteration', 'learning_rate', 'training_loss', 'validation_loss'])
                log_folder = os.path.join(os.path.dirname(self.results_folder),'log');ff.make_folder([log_folder])
                df.to_excel(os.path.join(log_folder, 'training_log.xlsx'),index=False)
                        
                # at the end of each epoch, call on_epoch_end
                self.ds.on_epoch_end()

                pbar.update(1)

        accelerator.print('training complete')


# Sampling class
# class Sampler(object):
#     def __init__(
#         self,
#         model,
#         generator,
#         batch_size,
#         image_size = None,
#         device = 'cuda',

#     ):
#         super().__init__()

#         # model
#         self.model = model  
#         if device == 'cuda':
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         if device == 'cpu':
#             self.device = torch.device("cpu")

#         self.channels = model.channels
#         if image_size is None:
#             self.image_size = self.model.image_size
#         else:
#             self.image_size = image_size
#         self.batch_size = batch_size

#         # dataset and dataloader

#         self.generator = generator
#         dl = DataLoader(self.generator, batch_size = self.batch_size, shuffle = False, pin_memory = True, num_workers = 0)# cpu_count())

#         self.dl = dl
#         self.cycle_dl = cycle(dl)
 
#         # EMA:
#         self.ema = EMA(model)
#         self.ema.to(self.device)

#     def load_model(self, trained_model_filename):

#         data = torch.load(trained_model_filename, map_location=self.device)

#         self.model.load_state_dict(data['model'])

#         self.step = data['step']

#         self.ema.load_state_dict(data["ema"])


#     def sample(self, trained_model_filename, save_file,  patient_class, patient_id, picked_tf, reshape_pred = False, save_gt_and_img = True, main_folder = '/mnt/camca_NAS/4DCT'):
        

#         self.load_model(trained_model_filename) 
        
#         device = self.device

#         self.ema.ema_model.eval()

#         # load gt
#         affine = nb.load(os.path.join(main_folder,'nii-images',patient_class, patient_id, 'img-nii-resampled-1.5mm','0.nii.gz')).affine
#         original_shape = nb.load(os.path.join(main_folder,'nii-images',patient_class, patient_id, 'img-nii-resampled-1.5mm','0.nii.gz')).shape
#         if save_gt_and_img == True:
#             gt = os.path.join(main_folder,'predicted_seg',patient_class, patient_id, 'seg-pred-0.625-4classes-connected-retouch-resampled-1.5mm','pred_s_'+str(picked_tf)+'.nii.gz')
#             affine = nb.load(gt).affine
#             gt = nb.load(gt).get_fdata()
#             if len(gt.shape) == 4:
#                 gt = gt[:,:,:,0]
#             # gt = np.round(gt).astype(int)
#             gt[gt == 4] = 3
#             gt = Data_processing.crop_or_pad(gt, (self.image_size[0], self.image_size[1], self.image_size[2]),value = 0)
#             if save_gt_and_img:
#                 nb.save(nb.Nifti1Image(gt, affine), os.path.join(os.path.dirname(save_file), 'gt_s_tf'+str(picked_tf)+'.nii.gz'))

#             # load img
#             img = nb.load(os.path.join(main_folder,'nii-images',patient_class, patient_id, 'img-nii-resampled-1.5mm',str(picked_tf)+'.nii.gz')).get_fdata()
#             img = Data_processing.crop_or_pad(img, (self.image_size[0], self.image_size[1], self.image_size[2]),value = np.min(img))
#             if save_gt_and_img:
#                 nb.save(nb.Nifti1Image(img, affine), os.path.join(os.path.dirname(save_file), 'img_tf'+str(picked_tf)+'.nii.gz'))


#         # start to run
#         with torch.inference_mode():
#             datas = next(self.cycle_dl)
#             batch_img , _ = datas
#             data_input = batch_img.to(device)
#             pred = self.ema.ema_model(data_input)
#             # print('pred shape is: ', pred.shape)

#             pred_seg = pred.argmax(1).detach().cpu().numpy().squeeze()
#             pred_seg = pred_seg.astype(float)
#             # print('pred seg shape is: ', pred_seg.shape, ' unique values are: ', np.unique(pred_seg))

#             if reshape_pred == True:
#                 pred_seg = Data_processing.correct_shift_caused_in_pad_crop_loop(Data_processing.crop_or_pad(pred_seg, (original_shape[0], original_shape[1], original_shape[2]),value = 0))


#             nb.save(nb.Nifti1Image(pred_seg, affine), save_file)
    
            
   