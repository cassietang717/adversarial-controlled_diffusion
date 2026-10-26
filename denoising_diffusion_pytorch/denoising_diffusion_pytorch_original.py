import math
import copy
import numpy as np
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import os
import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR, LinearLR, CosineAnnealingLR, SequentialLR, ConstantLR
import wandb
from torchvision import transforms as T, utils
from torchvision.datasets import CIFAR10
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from scipy.optimize import linear_sum_assignment

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator
from denoising_diffusion_pytorch.attend import Attend

from denoising_diffusion_pytorch.version import __version__

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

class GradEqualize(torch.autograd.Function):
    """Identity forward pass; backward multiplies gradient by `correction` to
    normalize the effective gradient magnitude of t_adv across all t_base values."""
    @staticmethod
    def forward(ctx, x, correction):
        ctx.save_for_backward(correction)
        return x
    @staticmethod
    def backward(ctx, grad):
        (correction,) = ctx.saved_tensors
        return grad * correction, None

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

# small helper modules

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * self.scale

# sinusoidal positional embeds

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta = 10000):
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

class RandomOrLearnedSinusoidalPosEmb(Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0.):
        super().__init__()

        self.to_t_map = None
        self.to_scale_shift = None
        if exists(time_emb_dim):
            self.to_t_map = nn.Conv2d(time_emb_dim, dim_out, 1)
            self.to_scale_shift = nn.Conv2d(dim_out, dim_out * 2, 1)

        self.block1 = Block(dim, dim_out, dropout=dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        b, _, h, w = x.shape
        scale_shift = None

        if exists(self.to_scale_shift) and exists(time_emb):
            if time_emb.shape[-2:] != (h, w):
                time_emb = F.interpolate(time_emb, size=(h, w), mode='nearest')
            time_emb = self.to_t_map(time_emb)
            scale_shift = self.to_scale_shift(time_emb).chunk(2, dim=1)

        h_ = self.block1(x, scale_shift=scale_shift)
        h_ = self.block2(h_)
        return h_ + self.res_conv(x)


class LinearAttention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model
class Unet_gen(Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        self_condition=False,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        sinusoidal_pos_emb_theta=10000,
        dropout=0.,
        attn_dim_head=32,
        attn_heads=4,
        full_attn=None,
        flash_attn=False,
        cond_channels=3,
    ):
        super().__init__()

        self.channels = channels
        self.self_condition = self_condition
        self.cond_channels = cond_channels

        input_channels = channels * (2 if self_condition else 1) + cond_channels
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)
        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash=flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim=time_dim, dropout=dropout)

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)
            attn_klass = FullAttention if layer_full_attn else LinearAttention
            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attn_klass(dim_in, dim_head=layer_attn_dim_head, heads=layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)
            attn_klass = FullAttention if layer_full_attn else LinearAttention
            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                attn_klass(dim_out, dim_head=layer_attn_dim_head, heads=layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    def time_emb_map(self, t_hw):
        b, h, w = t_hw.shape
        t_flat = t_hw.reshape(-1).float()
        emb = self.time_mlp(t_flat)
        return emb.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

    def build_t_pyramid(self, t_hw, num_downs):
        t = t_hw.unsqueeze(1).float()  # (B, H, W) -> (B, 1, H, W)
        t_downs = [t]
        for i in range(num_downs):
            if i < num_downs - 1:  # last down is nn.Conv2d, keeps same spatial size
                t = F.avg_pool2d(t, kernel_size=2, stride=2)
            t_downs.append(t)
        return [x.squeeze(1) for x in t_downs]

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond=None, cond_map=None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]])

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        b, _, h_img, w_img = x.shape

        # normalise time to (B, H, W) regardless of how it arrives
        if time.dim() == 1:  # scalar per sample (sampling path)
            time = time.float().view(b, 1, 1).expand(b, h_img, w_img)
        elif time.dim() == 4:  # (B, 1, H, W) from spatial timestep map
            time = time.squeeze(1)

        if cond_map is None:
            cond_map = torch.zeros(
                b, self.cond_channels, h_img, w_img,
                device=x.device, dtype=x.dtype
            )
        else:
            assert cond_map.shape[1] == self.cond_channels

        x = torch.cat((x, cond_map), dim=1)
        x = self.init_conv(x)
        r = x.clone()
        h = []

        t_pyramid = self.build_t_pyramid(time, num_downs=len(self.downs))

        for level, (block1, block2, attn, downsample) in enumerate(self.downs):
            t_level = self.time_emb_map(t_pyramid[level])
            x = block1(x, t_level)
            h.append(x)
            x = block2(x, t_level)
            x = attn(x) + x
            h.append(x)
            x = downsample(x)

        t_mid = self.time_emb_map(t_pyramid[len(self.downs)])
        x = self.mid_block1(x, t_mid)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t_mid)

        t_up_pyramid = list(reversed(t_pyramid[:-1]))
        for level, (block1, block2, attn, upsample) in enumerate(self.ups):
            t_level = self.time_emb_map(t_up_pyramid[level])
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t_level)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t_level)
            x = attn(x) + x
            x = upsample(x)

        t_final = self.time_emb_map(time)
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t_final)
        return self.final_conv(x)

    
class Unet_dis(Module):
    """
    Lightweight U-Net-like discriminator / adversarial noising network.

    Differences from the old PatchGAN-style Unet_dis:
    - keeps skip connections, so output can depend more on image content
    - progressively decodes features instead of only upsampling final low-res logits
    - conditions on global diffusion timestep t and stochastic noise strength s via FiLM
    - much smaller than Unet_gen

    Conditioning design: t and s are each encoded with SinusoidalPosEmb (both are
    integers in [0, T-1] so the same frequency grid applies), concatenated, and passed
    through a 2-layer MLP to produce a shared cond vector.  This vector is broadcast
    to (B, cond_dim, 1, 1) and injected into every ResnetBlock as a FiLM scale+shift —
    reusing the existing ResnetBlock infrastructure with no new block types needed.
    """

    def __init__(
        self,
        dim = 32,
        init_dim = None,
        out_dim = 1,
        dim_mults = (1, 2, 4),
        channels = 3,
        dropout = 0.,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = (False, False, False),
        flash_attn = False,
        use_attention = True,
        out_activation = 'softmax',  # 'softmax' | 'sigmoid' | 'tanh'
        softmax_temp = 2.0,          # temperature for softmax output
        detach_sftm_denom = False,   # stop gradient through softmax denominator (removes cross-pixel competition)
        **kwargs   # ignore legacy args so train.py needs no changes
    ):
        super().__init__()

        self.channels = channels
        self.out_dim = out_dim
        self.use_attention = use_attention
        self.out_activation = out_activation
        self.softmax_temp = softmax_temp
        self.detach_sftm_denom = detach_sftm_denom

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        num_stages = len(dim_mults)
        full_attn = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        # conditioning on global timestep t and noise-strength s (both integers in [0, T-1])
        cond_dim = init_dim * 4
        self.t_emb = SinusoidalPosEmb(init_dim)
        self.s_emb = SinusoidalPosEmb(init_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(init_dim * 2, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        FullAttention = partial(Attention, flash = flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim = cond_dim, dropout = dropout)

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        # down path
        for ind, ((dim_in, dim_out_stage), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(
            zip(in_out, full_attn, attn_heads, attn_dim_head)
        ):
            is_last = ind >= (num_resolutions - 1)
            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads) if use_attention else nn.Identity(),
                Downsample(dim_in, dim_out_stage) if not is_last else nn.Conv2d(dim_in, dim_out_stage, 3, padding = 1)
            ]))

        # bottleneck
        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = LinearAttention(mid_dim, dim_head = attn_dim_head[-1], heads = attn_heads[-1]) if use_attention else nn.Identity()
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        # up path
        for ind, ((dim_in, dim_out_stage), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(
            zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))
        ):
            is_last = ind == (len(in_out) - 1)
            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(ModuleList([
                resnet_block(dim_out_stage + dim_in, dim_out_stage),
                attn_klass(dim_out_stage, dim_head = layer_attn_dim_head, heads = layer_attn_heads) if use_attention else nn.Identity(),
                Upsample(dim_out_stage, dim_in) if not is_last else nn.Conv2d(dim_out_stage, dim_in, 3, padding = 1)
            ]))

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, out_dim, 1)

        self._init_weights()

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a = 0.2, mode = 'fan_out', nonlinearity = 'leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # make output start near zero so tanh(output) is initially near 0
        nn.init.zeros_(self.final_conv.weight)
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)

    def forward(self, x, t=None, s=None):
        B, C, H, W = x.shape
        factor = self.downsample_factor

        pad_h = (factor - H % factor) % factor
        pad_w = (factor - W % factor) % factor
        x = F.pad(x, (0, pad_w, 0, pad_h), mode = 'constant', value = 0)

        x = self.init_conv(x)
        r = x.clone()

        # FiLM conditioning: encode t and s (both integers in [0, T-1]) with sinusoidal
        # embeddings, combine through MLP, then broadcast as (B, cond_dim, 1, 1) so that
        # each ResnetBlock can interpolate it to its own spatial resolution.
        cond_map = None
        if t is not None and s is not None:
            t_e = self.t_emb(t.float())              # (B, init_dim)
            s_e = self.s_emb(s.float())              # (B, init_dim)
            cond = self.cond_mlp(torch.cat([t_e, s_e], dim=-1))  # (B, cond_dim)
            cond_map = cond.unsqueeze(-1).unsqueeze(-1)           # (B, cond_dim, 1, 1)

        h = []

        for block, attn, downsample in self.downs:
            x = block(x, cond_map)
            x = attn(x) + x if self.use_attention else x
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, cond_map)
        x = self.mid_attn(x) + x if self.use_attention else x
        x = self.mid_block2(x, cond_map)

        for block, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block(x, cond_map)
            x = attn(x) + x if self.use_attention else x
            x = upsample(x)

        x = torch.cat((x, r), dim = 1)
        x = self.final_res_block(x, cond_map)
        out = self.final_conv(x)

        if self.out_activation == 'softmax':
            out_flat = out.flatten(1)
            if self.detach_sftm_denom:
                # Detach the denominator (partition function Z) so cross-pixel competition
                # is removed from the gradient. Each pixel updates independently based on
                # its own logit, preventing winner-takes-all collapse via Z domination.
                logits = out_flat / self.softmax_temp
                exp_logits = logits.exp()
                Z = exp_logits.sum(dim=-1, keepdim=True).detach()
                t_adv = (exp_logits / Z).reshape(out.shape) * (out_flat.shape[-1] / 10)
            else:
                t_adv = F.softmax(out_flat / self.softmax_temp, dim=-1).reshape(out.shape) * (out_flat.shape[-1] / 10)
            # Soft-cap via x/(1+x): maps [0,∞)→[0,1), preserving the linear regime near 0
            # (uniform ≈0.1 barely changes) while squashing collapsed values (102→0.99)
            # so t_adv_clamped never hits t_max, gradient always flows to the discriminator.
            return t_adv / (1.0 + t_adv)
        elif self.out_activation == 'sigmoid':
            # per-pixel independent gate in [0, 1]; average ≈ 0.5 so scale to match softmax's ~0.1 average
            return torch.sigmoid(out) / 10.0
        elif self.out_activation == 'softsign':
            # softsign: x/(1+|x|), range (-1,1), never saturates — gradient is always 1/(1+|x|)^2
            # Scaled to match the softmax typical magnitude, then soft-capped for gradient stability.
            scale = out.shape[-1] * out.shape[-2] / 10.0
            t_adv = (out / (1.0 + out.abs())) * scale
            return t_adv / (1.0 + t_adv.abs())
        elif self.out_activation == 'tanh':
            out_flat = out.flatten(1)
            norm = torch.norm(torch.tanh(out_flat), dim=-1, keepdim=True) + 1e-6
            t_adv = ((torch.tanh(out_flat) / norm) * (out_flat.shape[-1] / 10)).reshape(out.shape)
            return t_adv / (1.0 + t_adv.abs())
        else:
            return out


def extract(a, t, x_shape):
    if t.dim() == 1:
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    else:
        # pixel-wise t (e.g. B x 1 x H x W): index a at each location, result broadcasts with x_shape
        return a[t.long()]

def extract_continuous(a, t, x_shape):
    T = a.shape[0]
    t = t.float().clamp(0, T - 1)
    t0 = t.floor().long()
    t1 = (t0 + 1).clamp(max=T - 1)
    w1 = t - t0.float()
    w0 = 1. - w1
    return w0 * a[t0] + w1 * a[t1]

def tv_loss(x):
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw

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

class GaussianDiffusion_orig(Module):
    def __init__(
        self,
        model,
        discriminator,
        *,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_v',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight = False, # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5,
        immiscible = False,
        reg_loss_weight = 1e-1,  # weight for discriminator output regularisation (mean=0, std=1)
        fixed_stochastic = None,   # None = random per step; float = fixed multiplier in [0,1]
        disc_tv_weight = 0.0,       # TV smoothness penalty on the spatial t_adv map
        bidirectional = False,      # if True, use signed t_adv (requires tanh discriminator)
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion_orig and model.channels != model.out_dim)
        assert not hasattr(model, 'random_or_learned_sinusoidal_cond') or not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.discriminator = discriminator
        self._tadv_vis_step = 0

        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        assert isinstance(image_size, (tuple, list)) and len(image_size) == 2, 'image size must be a integer or a tuple/list of two integers'
        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

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
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # immiscible diffusion

        self.immiscible = immiscible

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
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity
        self.reg_loss_weight = reg_loss_weight
        self.fixed_stochastic = fixed_stochastic
        self.disc_tv_weight = disc_tv_weight
        self.bidirectional = bidirectional

    @property
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
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
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

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond = None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps = False):
        batch, device = shape[0], self.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        # for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
        for t in reversed(range(0, self.num_timesteps)):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def ddim_sample(self, shape, return_all_timesteps = False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        # for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, batch_size = 16, return_all_timesteps = False):
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, h, w), return_all_timesteps = return_all_timesteps)

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

    def noise_assignment(self, x_start, noise):
        x_start, noise = tuple(rearrange(t, 'b ... -> b (...)') for t in (x_start, noise))
        dist = torch.cdist(x_start, noise)
        _, assign = linear_sum_assignment(dist.cpu())
        return torch.from_numpy(assign).to(dist.device)

    @autocast('cuda', enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.immiscible:
            assign = self.noise_assignment(x_start, noise)
            noise = noise[assign]

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @autocast('cuda', enabled = False)
    def q_sample_continuous(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract_continuous(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract_continuous(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise = None, offset_noise_strength = None, stop_path_b = False):
        b, c, h, w = x_start.shape

        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # compute stochastic_step_int before discriminator so we can condition on it
        if self.fixed_stochastic is not None:
            stochastic_step_int = torch.full((b,), int(float(self.fixed_stochastic) * self.num_timesteps),
                                             device=t.device, dtype=torch.long)
        else:
            stochastic_step_int = torch.randint(0, self.num_timesteps - 1, (b,), device=t.device)

        # noise sample — discriminator conditioned on global t and stochastic noise strength s
        t_adv = self.discriminator(x_start, t=t, s=stochastic_step_int)  # (B, C, H, W)

        self._tadv_vis_step += 1

        self._t_adv_raw = t_adv  # stash for reg loss in training loop (avoids second forward pass)
        t_max = self.num_timesteps - 1
        t_base = t.float().view(b, 1, 1, 1)  # broadcast scalar t to spatial dims

        # normalise stochastic_step_int (already computed above) to [0, 1) and broadcast
        stochastic_step = (stochastic_step_int.float() / self.num_timesteps).view(b, 1, 1, 1)

        # Equalize gradient magnitude across (t_base, s).
        # Without this ∂t_adv_clamped/∂t_adv = (t_max-t_base)*s, which is up to
        # T_max× larger at (t_base=0, s=1) than at (t_base≈T_max, s≈0).
        # GradEqualize is identity in the forward pass; backward multiplies by
        # correction so the effective gradient is anchored to the reference scale
        # at (t = T_max/2, s = 0.5) — the visually best-structured operating point.
        with torch.no_grad():
            actual_scale = torch.where(
                t_adv > 0,
                (t_max - t_base) * stochastic_step,
                t_base * stochastic_step,
            ).clamp(min=1.0)          # guard ÷0 at t_base=0, t_base=t_max, or s=0
            reference_scale = float(t_max) / 2.0 * 0.5   # = 250 for T_max=1000
            correction = reference_scale / actual_scale
        t_adv = GradEqualize.apply(t_adv, correction)

        # bidirectional=True enables signed t_adv (tanh discriminator: negative = push t down)
        # bidirectional=False retains original behaviour (softmax/sigmoid: t_adv >= 0, only pushes up)
        t_adv_clamped = torch.where(
            t_adv > 0,
            t_base + (t_max - t_base) * t_adv * stochastic_step,
            t_base + t_base * t_adv * stochastic_step
        ).clamp(0, t_max)

        t_adv = t_adv_clamped  # continuous — gradients flow through extract_continuous interpolation
        self._t_adv_clamped = t_adv_clamped  # stash for TV loss in discriminator step

        if self._tadv_vis_step % 5000 == 1:
            tadv_vis        = self._t_adv_raw.mean(dim=1).detach().cpu()     # (B, H, W) raw discriminator output
            tadvc_vis       = t_adv_clamped.mean(dim=1).detach().cpu()       # (B, H, W) clamped timestep map
            x_start_vis     = x_start.detach().cpu()
            stoch_strengths = (stochastic_step_int.float() / self.num_timesteps).detach().cpu()  # (B,) in [0,1)

            n_samples = min(10, b)
            indices = torch.randperm(b)[:n_samples].sort().values
            fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
            if n_samples == 1:
                axes = axes[None]  # ensure 2D indexing
            for row, i in enumerate(indices.tolist()):
                global_t_i = t[i].item()
                s_i        = stoch_strengths[i].item()

                img = x_start_vis[i].permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                if img.shape[-1] == 1:
                    img = img[..., 0]
                elif img.shape[-1] == 2:
                    # R-B display: insert zero green channel
                    img = np.concatenate([img[:, :, :1], np.zeros((*img.shape[:2], 1), dtype=img.dtype), img[:, :, 1:]], axis=-1)
                axes[row, 0].imshow(img, cmap='gray' if img.ndim == 2 else None)
                axes[row, 0].set_title(f'sample {i}\nglobal_t={global_t_i}  s={s_i:.3f}')
                axes[row, 0].axis('off')

                raw_map = tadv_vis[i].numpy()
                im1 = axes[row, 1].imshow(raw_map, cmap='plasma')
                plt.colorbar(im1, ax=axes[row, 1], label='t_adv (raw)')
                axes[row, 1].set_title(f't_adv raw\nmin={raw_map.min():.3f}  max={raw_map.max():.3f}')
                axes[row, 1].axis('off')

                clamped_map = tadvc_vis[i].numpy()
                im2 = axes[row, 2].imshow(clamped_map, cmap='plasma')
                plt.colorbar(im2, ax=axes[row, 2], label='t_adv_clamped')
                axes[row, 2].set_title(f't_adv_clamped\nmin={clamped_map.min():.1f}  max={clamped_map.max():.1f}')
                axes[row, 2].axis('off')

            fig.tight_layout()
            wandb.log({"tadv_sanity": wandb.Image(fig)}, commit=False)
            plt.close(fig)

        x = self.q_sample_continuous(x_start=x_start, t=t_adv, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t_adv).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
        # stop_path_b=True: detach t_adv from the time embedding so adversarial gradient
        # only flows through the noised image (Path A), not through the UNet time conditioning.
        # Without this, Path B (time emb gradient) partially cancels Path A, collapsing to a spike.
        t_for_emb = t_adv.detach() if stop_path_b else t_adv
        model_out = self.model(x, t_for_emb, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            # detach t_adv so the target is treated as a constant w.r.t. the discriminator.
            # Without detach, d(target)/d(t_adv) != 0 via extract_continuous, which creates
            # a gradient that reduces the loss (wrong direction for the discriminator).
            t_adv_det = t_adv.detach()
            target = (
                extract_continuous(self.sqrt_alphas_cumprod, t_adv_det, x_start.shape) * noise -
                extract_continuous(self.sqrt_one_minus_alphas_cumprod, t_adv_det, x_start.shape) * x_start
            )
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = loss * extract_continuous(self.loss_weight, t_adv_det, loss.shape)
        loss = reduce(loss, 'b ... -> b', 'mean')
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size[0] and w == img_size[1], f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)

# dataset classes

class AddWhiteEdgeOutside:
    def __init__(self, pad=4, value=1.0):
        self.pad = pad
        self.value = value

    def __call__(self, img):
        c, h, w = img.shape
        new_h = h + 2 * self.pad
        new_w = w + 2 * self.pad
        canvas = torch.full((c, new_h, new_w), self.value, dtype=img.dtype)
        canvas[:, self.pad:self.pad + h, self.pad:self.pad + w] = img
        return canvas


class SyntheticPaddedDataset(Dataset):
    def __init__(self, base_dataset, pad_transform=None, to_minus1_1=False):
        self.base_dataset = base_dataset
        self.pad_transform = pad_transform
        self.to_minus1_1 = to_minus1_1

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        if self.pad_transform is not None:
            img = self.pad_transform(img)
        if self.to_minus1_1:
            img = img * 2.0 - 1.0
        return img, label


class MNISTCircleDataset(Dataset):
    """1-channel 32×32 dataset.

    MNIST digit zero-padded to 32×32, then multiplied by a binary circle mask
    (center = image center (15.5, 15.5), radius = 8). Everything outside the
    circle is set to 0; only the MNIST signal inside the circle is kept.
    """

    def __init__(self, root='data', train=True, download=True):
        self.mnist = datasets.MNIST(
            root=root, train=train, download=download,
            transform=transforms.ToTensor()
        )
        H, W = 32, 32
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0  # 15.5, 15.5
        ys = torch.arange(H).float()
        xs = torch.arange(W).float()
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        self.circle_mask = ((yy - cy) ** 2 + (xx - cx) ** 2 <= 8.0 ** 2).float().unsqueeze(0)  # (1, 32, 32)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        img, label = self.mnist[idx]  # (1, 28, 28) in [0, 1]
        img = F.pad(img, (0, 4, 4, 0), value=0.0)   # (1, 32, 32)
        img = img * self.circle_mask                  # zero outside circle
        return img, label  # (1, 32, 32)


class Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# trainer class

class Trainer:
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None,
        calculate_fid = True,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 50000,
        save_best_and_latest_only = False,
        adv_weights = 0.5,
        tv_weight = 5e-2,
        disc_tv_weight = 0.0,
        stop_path_b_disc = False,
        adv_start_step = 0,             # gen steps of denoiser-only warmup before adv training begins
        dataset = 'mnist_white_border',  # 'mnist_white_border' | 'cifar10' | 'mask_circle_mnist'
    ):
        super().__init__()

        self.tv_weight = tv_weight
        self.disc_tv_weight = disc_tv_weight
        self.stop_path_b_disc = stop_path_b_disc
        self.adv_start_step = adv_start_step
        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # default convert_image_to depending on channels

        if not exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.adv_weights = adv_weights

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader

        assert dataset in ('mnist_white_border', 'cifar10', 'mask_circle_mnist'), \
            f"dataset must be one of 'mnist_white_border', 'cifar10', 'mask_circle_mnist', got '{dataset}'"
        self.dataset_name = dataset

        if dataset == 'mnist_white_border':
            base_ds = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
            pad_aug = AddWhiteEdgeOutside(pad=2, value=1.0)
            self.ds = SyntheticPaddedDataset(base_dataset=base_ds, pad_transform=pad_aug, to_minus1_1=False)
        elif dataset == 'cifar10':
            base_ds = datasets.CIFAR10(root='data', train=True, download=True, transform=transforms.ToTensor())
            self.ds = SyntheticPaddedDataset(base_dataset=base_ds, pad_transform=None, to_minus1_1=False)
        elif dataset == 'mask_circle_mnist':
            self.ds = MNISTCircleDataset(root='data', train=True, download=True)

        assert len(self.ds) >= 100, 'dataset should have at least 100 samples'

        dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=min(4, cpu_count()))
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer — separate for generator (Unet_gen) and discriminator (Unet_dis)

        self.opt = Adam(diffusion_model.model.parameters(), lr = train_lr, betas = adam_betas)
        if self.adv_weights == 0:
            self.opt_disc = Adam(diffusion_model.discriminator.parameters(), lr = 0.0, betas = adam_betas)
        else:
            self.opt_disc = Adam(diffusion_model.discriminator.parameters(), lr = train_lr * (self.adv_weights), betas = adam_betas)

        disc_warmup_steps = 10000
        total_disc_steps = self.train_num_steps // 2  # disc is updated every other step
        disc_pretrain_steps = adv_start_step          # disc steps at LR=0 while denoiser warms up
        disc_cosine_steps = max(1, total_disc_steps - disc_pretrain_steps - disc_warmup_steps)
        if disc_pretrain_steps > 0:
            disc_pretrain = ConstantLR(self.opt_disc, factor=0.0, total_iters=disc_pretrain_steps)
            disc_warmup   = LinearLR(self.opt_disc, start_factor=1e-6, end_factor=1.0, total_iters=disc_warmup_steps)
            disc_cosine   = CosineAnnealingLR(self.opt_disc, T_max=disc_cosine_steps)
            self.sched_disc = SequentialLR(
                self.opt_disc,
                schedulers=[disc_pretrain, disc_warmup, disc_cosine],
                milestones=[disc_pretrain_steps, disc_pretrain_steps + disc_warmup_steps],
            )
        else:
            disc_warmup = LinearLR(self.opt_disc, start_factor=1e-6, end_factor=1.0, total_iters=disc_warmup_steps)
            disc_cosine = CosineAnnealingLR(self.opt_disc, T_max=disc_cosine_steps)
            self.sched_disc = SequentialLR(self.opt_disc, schedulers=[disc_warmup, disc_cosine], milestones=[disc_warmup_steps])

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizers with accelerator

        self.model, self.opt, self.opt_disc, self.sched_disc = self.accelerator.prepare(self.model, self.opt, self.opt_disc, self.sched_disc)

        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation

            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
                    "Consider using DDIM sampling to save time."
                )

            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx,
                stats_name=f"dataset_stats_{dataset}",
            )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # infinite

        self.save_best_and_latest_only = save_best_and_latest_only

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'opt_disc': self.opt_disc.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.opt_disc.load_state_dict(data['opt_disc'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.model.train()
                if self.step %2 == 0:
                    total_loss = 0.
                if self.step %2 == 1:
                    total_disc_loss = 0.
                    total_disc_reg_loss = 0.
                unwrapped = accelerator.unwrap_model(self.model)

                disc_did_backward = False  # tracks whether disc backward ran this step

                for _ in range(self.gradient_accumulate_every):
                    data, _ = next(self.dl)
                    data = data.to(device)

                    if self.step %2 == 0:
                        # --- generator step (freeze discriminator) ---
                        for p in unwrapped.discriminator.parameters():
                            p.requires_grad_(False)

                        with self.accelerator.autocast():
                            loss = self.model(data)
                            loss = loss / self.gradient_accumulate_every
                            total_loss += loss.item()

                        self.accelerator.backward(loss)

                        for p in unwrapped.discriminator.parameters():
                            p.requires_grad_(True)

                    if self.step %2 == 1:
                        # --- discriminator step (freeze generator) ---
                        # Skip disc update during denoiser-only pre-training phase.
                        if (self.step // 2) < self.adv_start_step:
                            continue

                        disc_did_backward = True
                        for p in unwrapped.model.parameters():
                            p.requires_grad_(False)

                        with self.accelerator.autocast():
                            adv_loss = -self.model(data, stop_path_b=self.stop_path_b_disc)
                            unwrapped_diff = accelerator.unwrap_model(self.model)
                            t_adv_raw = unwrapped_diff._t_adv_raw
                            b_r, c_r = t_adv_raw.shape[:2]
                            disc_act = getattr(unwrapped_diff.discriminator, 'out_activation', 'softmax')

                            # per-pixel binary entropy: -p*log(p) - (1-p)*log(1-p), peaks at p=0.5
                            if disc_act in ('softmax', 'sigmoid'):
                                # compute in fp32: 1e-8 underflows to 0 in fp16 → log(0)=-Inf → 0*-Inf=NaN
                                p = t_adv_raw.float().view(b_r, c_r, -1)
                                p = p / (p.sum(dim=-1, keepdim=True) + 1e-6)
                                entropy = -(p * (p + 1e-6).log()).sum(dim=-1).mean()
                            else:
                                # tanh: maximise entropy of |output| magnitude distribution
                                # compute in fp32: 1e-8 underflows to 0 in fp16 → 0/0 NaN at init
                                p = t_adv_raw.float().abs().view(b_r, c_r, -1) + 1e-6
                                p = p / p.sum(dim=-1, keepdim=True)
                                entropy = -(p * p.log()).sum(dim=-1).mean()

                            reg_loss = -self.tv_weight * entropy
                            # optional: TV smoothness on the spatial t_adv map
                            if self.disc_tv_weight > 0:
                                t_adv_clamped = unwrapped_diff._t_adv_clamped
                                tv_reg = tv_loss(t_adv_clamped / float(unwrapped_diff.num_timesteps - 1))
                                reg_loss = reg_loss + self.disc_tv_weight * tv_reg
                            disc_loss = (adv_loss + reg_loss) / self.gradient_accumulate_every
                            total_disc_loss += adv_loss.item() / self.gradient_accumulate_every
                            total_disc_reg_loss += reg_loss.item() / self.gradient_accumulate_every

                        self.accelerator.backward(disc_loss)

                        for p in unwrapped.model.parameters():
                            p.requires_grad_(True)

                if self.step %2 == 1:
                    wandb.log({"loss": total_loss}, step=self.step//2)
                    wandb.log({"adv_loss": total_disc_loss}, step=self.step//2)
                    wandb.log({"adv_reg_loss": total_disc_reg_loss}, step=self.step//2)
                    wandb.log({"adv_lr": self.sched_disc.get_last_lr()[0]}, step=self.step//2)
                    pbar.set_description(f'loss: {total_loss:.4f} | disc loss: {total_disc_loss:.4f} | reg: {total_disc_reg_loss:.4f}')
               

                accelerator.wait_for_everyone()

                if self.step %2 == 0:
                    accelerator.clip_grad_norm_(unwrapped.model.parameters(), self.max_grad_norm)
                    self.opt.step()
                    self.opt.zero_grad()
                if self.step %2 == 1:
                    if disc_did_backward:
                        accelerator.clip_grad_norm_(unwrapped.discriminator.parameters(), self.max_grad_norm)
                        self.opt_disc.step()
                        self.opt_disc.zero_grad()
                    self.sched_disc.step()  # always advance — keeps LR schedule in sync with wall-clock steps

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        print('waiting for sampling loop time step') 
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim = 0)

                        save_images = all_images
                        if save_images.shape[1] == 2:
                            # torchvision only handles 1/3-channel; insert zero green for R-B display
                            zeros = torch.zeros_like(save_images[:, :1])
                            save_images = torch.cat([save_images[:, :1], zeros, save_images[:, 1:]], dim=1)
                        utils.save_image(save_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')
                            wandb.log({"fid_score": fid_score}, step=self.step//2)

                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
