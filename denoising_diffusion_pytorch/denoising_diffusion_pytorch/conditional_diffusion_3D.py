import sys 
sys.path.append('/workspace/Documents')
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
from scipy.ndimage import zoom

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

from Cardiac4DCT_Synth_Diffusion.denoising_diffusion_pytorch.denoising_diffusion_pytorch.attend import Attend

from Cardiac4DCT_Synth_Diffusion.denoising_diffusion_pytorch.denoising_diffusion_pytorch.version import __version__

import Cardiac4DCT_Synth_Diffusion.functions_collection as ff
import Cardiac4DCT_Synth_Diffusion.Data_processing as Data_processing

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

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

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def Upsample3D(dim, dim_out = None, upsample_factor = (2,2,2)):
    return nn.Sequential(
        nn.Upsample(scale_factor = upsample_factor, mode = 'nearest'),
        nn.Conv3d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample3D(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) (d p3) -> b (c p1 p2 p3) h w d', p1 = 2, p2 = 2, p3 = 2),
        nn.Conv3d(dim * 8, default(dim_out, dim), 1)
    )


class RMSNorm3D(nn.Module):
    '''RMSNorm applies channel-wise normalization to the input tensor, 
    scales the normalized values using the learnable parameter g, 
    and then further scales the result by the square root of the number of input channels. '''
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1 , 1)) # learnable

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module): # output dimension is dim
    '''https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/'''
    def __init__(self, dim, theta = 10000): # theta is the n on the guidance webpage, dim is d (Dimension of the output embedding space).
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)  # weights are not 1/(n**(2i/d)), instead it's learnable

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block3D(nn.Module):  # input dimension is dim, output dimension is dim_out
    def __init__(self, dim, dim_out, groups = 8): 
        super().__init__()
        self.conv = nn.Conv3d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)   
        self.act = nn.SiLU()  

    def forward(self, x, scale_shift = None, cond_emb = None):
        x = self.conv(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        if exists(cond_emb):
            x = x + cond_emb
            # print('yes we have cond_emb')

        x = self.act(x)
        return x

class ResnetBlock3D(nn.Module): # input dimension is dim , output dimension is dim_out. for time_emb, the input dimension is time_emb_dim, output dimension is dim_out * 2
    '''experience two basic convolution+group_normlization+SiLu blocks, and then add the input to the output of the second block.'''
    def __init__(self, dim, dim_out, *, time_emb_dim = None, cond_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp_time = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2) # fully-connected layer
        ) if exists(time_emb_dim) else None

        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_emb_dim, dim_out)  # Outputs conditional bias
        ) if exists(cond_emb_dim) else None

        self.block1 = Block3D(dim, dim_out, groups = groups)
        self.block2 = Block3D(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None, cond_emb = None):
        scale_shift = None
        if exists(self.mlp_time) and exists(time_emb):
            time_emb = self.mlp_time(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        # Process conditional embedding for bias
        cond_bias = None
        if exists(self.cond_proj) and exists(cond_emb):
            cond_emb = self.cond_proj(cond_emb)
            cond_bias = rearrange(cond_emb, 'b c -> b c 1 1 1')  # Reshape for broadcast

        h = self.block1(x, scale_shift=scale_shift, cond_emb=cond_bias)
        h = self.block2(h, cond_emb=cond_bias)

        return h + self.res_conv(x)
    
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
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm3D(dim)
        self.attend = Attend(flash = flash)

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

# model

class Unet3D_tfcondition(nn.Module):
    def __init__(
        self,
        init_dim,
        out_dim,
        channels,
        conditional_diffusion_image,
        conditional_diffusion_EF,
        conditional_diffusion_seg,

        dim_mults = (1, 2, 4, 8),
        downsample_list = (True, True, True, False),
        upsample_list = (True, True, True, False),
        self_condition = False,   # use the prediction from the previous iteration as the condition of next iteration
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False, 
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = (None, None, None, True),
        flash_attn = False
    ):
        super().__init__()

        # determine dimensions
        self.self_condition = self_condition
        self.conditional_diffusion_image = conditional_diffusion_image
        self.conditional_diffusion_EF = conditional_diffusion_EF
        self.conditional_diffusion_seg = conditional_diffusion_seg

        self.channels = channels
        input_channels = channels + (1 if self.conditional_diffusion_image else 0) 
        input_channels = input_channels + (1 if self.conditional_diffusion_seg else 0)

        self.init_conv = nn.Conv3d(input_channels, init_dim, 5, padding = 2) # if want input and output to have same dimension, Kernel size to any odd number (e.g., 3, 5, 7, etc.). Padding to (kernel size - 1) / 2.

        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]  # if initi_dim = 16, then [16,32,64,128]
        in_out = list(zip(dims[:-1], dims[1:])) 
        # [(16,16), (16, 32), (32, 64), (64, 128)]. Each tuple in in_out represents a pair of input and output dimensions for different stages in a neural network

        block_klass = partial(ResnetBlock3D, groups = resnet_block_groups)  # Here, block_klass is being defined as a new function that is essentially a ResnetBlock, but with the groups argument set to resnet_block_groups. This means that when you call block_klass, you only need to provide the remaining arguments that ResnetBlock expects (such as dim and dim_out), and groups will be automatically set to the value of resnet_block_groups.

        # time embeddings
        time_dim = init_dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(init_dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = init_dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),  # Gaussian error activation function
            nn.Linear(time_dim, time_dim)
        )

        # other condition embeddings
        if self.conditional_diffusion_EF:
            self.conditional_EF_emb = nn.Sequential(
                nn.Linear(1, init_dim * 4),
                nn.GELU(),
                nn.Linear(init_dim * 4, init_dim * 4))


        # attention
        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention3D, flash = flash_attn)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out) # 4

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = downsample_list[ind] == False

            if layer_full_attn == True:
                attn_klass = FullAttention
            elif layer_full_attn == False:
                attn_klass = LinearAttention3D

            # in each downsample stage, 
            # we have 4 layers: 2 resnet blocks (doesn't increase the feature number), 1 attention layer, and 1 downsampling layer (downsample x and y by 2, then increase the feature number by 2)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, cond_emb_dim = init_dim * 4),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, cond_emb_dim = init_dim * 4),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads) if layer_full_attn is not None else nn.Identity(),
                Downsample3D(dim_in, dim_out) if not is_last else nn.Conv3d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, cond_emb_dim = init_dim * 4)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, cond_emb_dim = init_dim * 4)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = upsample_list[ind] == False

            if layer_full_attn == True:
                attn_klass = FullAttention
            elif layer_full_attn == False:
                attn_klass = LinearAttention3D
          
            # in each upsample stage,
            # we have 4 layers: 2 resnet blocks (does change the feature number), 1 attention layer, and 1 upsampling layer (upsample x and y by 2, then decrease the feature number by 2)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, cond_emb_dim = init_dim * 4),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, cond_emb_dim = init_dim * 4),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads) if layer_full_attn is not None else nn.Identity(),
                Upsample3D(dim_out, dim_in) if not is_last else  nn.Conv3d(dim_out, dim_in, 5, padding = 2)  
            ]))

        # default_out_dim = channels * (1 if not learned_variance else 2)  # channels = input channels, learned_variance = False
        self.out_dim = out_dim

        self.final_res_block = block_klass(init_dim * 2, init_dim, time_emb_dim = time_dim, cond_emb_dim = init_dim * 4)
        self.final_conv = nn.Conv3d(init_dim, self.out_dim, 1)  # output channel is initial channel number

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1) # = 8

    def forward(self, x, time,  condition_image = None, condition_EF = None, condition_seg = None):

        # concatenate the condition to the input along the dimension = 1
        if self.conditional_diffusion_image:
            if exists(condition_image) == 0:
                raise ValueError('condition is required for conditional diffusion')
            x = torch.cat((x, condition_image), dim = 1)
            # print('in model we have condition_image as shape ', condition_image.shape, ' therefore x shape is ', x.shape)
        if self.conditional_diffusion_seg:
            if exists(condition_seg) == 0:
                raise ValueError('condition is required for conditional diffusion')
            x = torch.cat((x, condition_seg), dim = 1)
            # print('in model we have condition_seg as shape ', condition_seg.shape, ' therefore x shape is ', x.shape)
        
        # initialize the input
        initial_x = x.clone() 
        x = self.init_conv(x)
        r = x.clone()

        # embed the time
        t_emb = self.time_mlp(time)

        # Conditional embedding
        # embed the time frame
        if self.conditional_diffusion_EF:
            if condition_EF is None:
                raise ValueError("Condition Ejection Fraction is required for conditional diffusion")
            cond_emb = self.conditional_EF_emb(condition_EF)
            # print('in model we have only have condition_EF as shape ', condition_EF.shape, ' and the emb shape is ', cond_emb.shape)
        else:
            cond_emb = None
            # print('in model we have no condition tf and EF')

        h = []

        for block1, block2, attn, downsample in self.downs:
            # print('ready to downsample, x shape is ', x.shape)
            x = block1(x, time_emb = t_emb, cond_emb = cond_emb)
            h.append(x)

            x = block2(x, time_emb = t_emb, cond_emb = cond_emb)

            x = attn(x) + x
            h.append(x)

            x = downsample(x)
        #     print('after this downsample, x shape is ', x.shape)
        # print('after all downsample, x shape is ', x.shape)
              
        x = self.mid_block1(x, time_emb = t_emb, cond_emb = cond_emb)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, time_emb = t_emb, cond_emb = cond_emb)
        
        for block1, block2, attn, upsample in self.ups:
            # print('ready to upsample, x shape is ', x.shape)
            x = torch.cat((x, h.pop()), dim = 1)   # h.pop() is the output of the corresponding downsample stage
            x = block1(x, time_emb = t_emb, cond_emb = cond_emb)

            x = torch.cat((x, h.pop()), dim = 1)

            x = block2(x, time_emb = t_emb, cond_emb = cond_emb)

            x = attn(x) + x

            x = upsample(x)
            # print('after this upsample, x shape is ', x.shape)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, time_emb = t_emb, cond_emb = cond_emb)
        final_image = self.final_conv(x)
        # print('in model final image shape is ', final_image.shape, ' intial_x shape is ', initial_x.shape)

        if self.conditional_diffusion_seg:
            return  final_image, initial_x
        else:
            return final_image

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    # extract at from a list of a, then add empty axes to match the image dimension
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1))) 

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion3D(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size_3D,
        timesteps = 2000,
        sampling_timesteps = None,
        objective = 'pred_noise',  # previous definition is "pred_v"
        clip_or_not = None,
        clip_range = None,

        force_ddim = False,
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = False,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion3D and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.conditional_diffusion_image = model.conditional_diffusion_image
        self.conditional_diffusion_EF = model.conditional_diffusion_EF
        self.conditional_diffusion_seg = model.conditional_diffusion_seg

        self.image_size_3D = image_size_3D

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        # clip to stablize
        self.clip_or_not = clip_or_not
        self.clip_range = clip_range
        assert self.clip_or_not is not None, 'clip_or_not must be specified'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs) # pre-defined schedule_fn_kwargs in main function arguments

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)  # This one is alpha(t-1)-bar. Practically, this pads the tensor with one element at the beginning and no elements at the end, using a padding value of 1.

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        self.force_ddim = force_ddim

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)  

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))  # this is sqrt(1- alpha(t)-bar)  /  sqrt(alpha(t)-bar) in the note

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) # use option 2 in my note

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))  # The method a.clamp(min) applies element-wise clamping to a tensor min, ensuring that all values are greater than or equal to min
        register_buffer('posterior_mean_coef_x0', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))  # for x0
        register_buffer('posterior_mean_coef_xt', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))  # for xt

        # offset noise strength - in blogpost, they claimed 0.1 was ideal

        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)  # element-wise division
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    # @property is a built-in Python decorator that allows you to define a method as if it were a class attribute. This means you can access it like an attribute rather than calling it as a method.
    # for example, if in class "Circle" we have a function "area" as the property, then we can call this function as Circle.area instead of Circle.area()
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef_x0, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef_xt, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, condition_image = None, condition_EF = None, condition_seg = None):
        model_output = self.model(x, t, condition_image, condition_EF, condition_seg)
        model_output = model_output[0] if isinstance(model_output, tuple) else model_output
       
        if self.clip_or_not:
            maybe_clip = partial(torch.clamp, min = self.clip_range[0], max = self.clip_range[1]) 
        else:
            maybe_clip = identity
       
        if self.objective == 'pred_noise': 
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)

            x_start = maybe_clip(x_start)

            if self.clip_or_not:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)  # if prediction = ModelPrediction(pred_noise = a, x_start = b), then prediction.pred_noise = a, prediction.x_start = b


    def p_mean_variance(self, x, t,  condition_image = None, condition_EF = None, condition_seg = None, output_noise = False):
        preds = self.model_predictions(x, t, condition_image = condition_image, condition_EF = condition_EF, condition_seg = condition_seg)
        
        x_start = preds.pred_x_start

        if self.clip_or_not:
            x_start.clamp_(self.clip_range[0], self.clip_range[1])
            # print('in p_mean_variance, x_start max and min: ',torch.max(x_start), torch.min(x_start))

        pred_noise = self.predict_noise_from_start(x, t, x_start)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        if output_noise == False:
            return model_mean, posterior_variance, posterior_log_variance, x_start
        else:
            return model_mean, posterior_variance, posterior_log_variance, x_start, pred_noise

    @torch.inference_mode()
    # In PyTorch, torch.inference_mode() is a context manager that temporarily sets the mode of the autograd engine to inference mode. This means that operations inside the context are treated as if they are being used for inference, rather than for training.
    def p_sample(self, x, t, condition_image = None, condition_EF = None, condition_seg = None, output_noise = False):
        b, *_, device = *x.shape, self.device   # * in front of a list means unpacking the list, b = batch
        batched_times = torch.full((b,), t, device = device, dtype = torch.long) # torch.full() is a function in PyTorch that creates a tensor of a specified size and fills it with a specified value.
        
        if output_noise == False:
            model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, condition_image = condition_image, condition_EF = condition_EF, condition_seg = condition_seg)
        else:
            model_mean, _, model_log_variance, x_start, pred_noise = self.p_mean_variance(x = x, t = batched_times, condition_image = condition_image,  condition_EF = condition_EF, condition_seg = condition_seg, output_noise = output_noise)
        
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise

        if output_noise == False:
            return pred_img, x_start
        else:
            return pred_img, x_start, pred_noise
        

    @torch.inference_mode()
    def p_sample_loop(self, shape,  condition_image = None, condition_EF = None, condition_seg = None, output_noise = False):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device = device)  # this is random noise
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            # self_cond = x_start if self.self_condition else None
            model_output = self.p_sample(img, t,  condition_image = condition_image,condition_EF = condition_EF, condition_seg = condition_seg, output_noise = output_noise)
            if output_noise == True:
                img,x_start, pred_noise = model_output
            else:
                img, x_start = model_output
            imgs.append(img)
            
        final_answer_x0 = img #if not return_all_timesteps else torch.stack(imgs, dim = 1)

        final_answer_x0 = self.unnormalize(final_answer_x0)
        return final_answer_x0

    @torch.inference_mode() 
    def ddim_sample(self, shape,  condition_image = None, condition_EF = None, condition_seg = None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        print('using DDIM, eta: ',eta)
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        coefficients = []
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            
            pred = self.model_predictions(img, time_cond,  condition_image = condition_image, condition_EF = condition_EF, condition_seg = condition_seg)

            if time_next < 0:
                img = pred.pred_x_start
                imgs.append(img)
                
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = pred.pred_x_start * alpha_next.sqrt() + \
                  c * pred.pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img 

        ret = self.unnormalize(ret)

        return ret

    @torch.inference_mode()
    def sample(self,  condition_image = None, condition_EF = None,  condition_seg = None, batch_size = 16):
       
        if self.force_ddim == False:
            sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        else:
            sample_fn = self.ddim_sample
        return sample_fn((batch_size, self.channels, self.image_size_3D[0], self.image_size_3D[1], self.image_size_3D[2]),  condition_image = condition_image, condition_EF = condition_EF, condition_seg = condition_seg)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise = None):
        '''prepare random xt from x_start and t'''
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t,  condition_image = None,  condition_EF = None, condition_seg = None, noise = None, offset_noise_strength = None):
        '''loss_weight_class is a list of [loss for bone, loss for brain, loss for air]'''
        b, c, h, w ,d = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1 1')

        # noise sample

        x = self.q_sample(x_start = x_start, t = t, noise = noise)
        
        # predict and take gradient step

        model_out = self.model(x, t, condition_image = condition_image, condition_EF = condition_EF, condition_seg = condition_seg)
        if isinstance(model_out, tuple):
            model_out, initial_input = model_out[0], model_out[1]
        else:
            model_out, initial_input = model_out, torch.zeros_like(x_start)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        
        # MSE loss:
        loss = F.mse_loss(model_out, target, reduction = 'none')  #reduction='none' argument ensures that the loss is computed element-wise, without any reduction across batches.
        loss = reduce(loss, 'b ... -> b (...)', 'mean') # reduce() operates on the batch dimension (b) and potentially other dimensions (...). It reduces the loss tensor to have the same shape as the target tensor, with a mean reduction.
        loss = loss * extract(self.loss_weight, t, loss.shape)  # assign different loss weight to different timesteps
        return loss.mean(), target, initial_input

            
    def forward(self, img, condition_image = None, condition_EF = None, condition_seg= None, *args, **kwargs):
        b, c, h, w, d, device, img_size_3D, = *img.shape, img.device, self.image_size_3D
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long() #torch.randint(a, b, size, device=None, dtype=None): This is a function in PyTorch used to generate random integers in a specified range. .long(): This method converts the tensor to have a data type of long, which is equivalent to 64-bit integer in PyTorch.

        # img = self.normalize(img)
        loss, target, initial_input =  self.p_losses(img, t, condition_image = condition_image, condition_EF = condition_EF, condition_seg = condition_seg, *args, **kwargs)
        return loss, target, initial_input
   