import math
import os
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import wandb
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision import transforms as T, utils
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from PIL import Image
from tqdm.auto import tqdm
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_original import MNISTCircleDataset
from ema_pytorch import EMA
from accelerate import Accelerator

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from denoising_diffusion_pytorch.attend import Attend
from denoising_diffusion_pytorch.version import __version__

# =====================================================
# constants
# =====================================================

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# =====================================================
# helpers
# =====================================================

def tensor_has_bad(x):
    return torch.isnan(x).any() or torch.isinf(x).any()


class BadStepError(RuntimeError):
    pass


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cast_tuple(t, length=1):
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


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def tv_loss(x):
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


def spatial_softmax(logits, temperature=0.1):
    b, c, h, w = logits.shape
    assert c == 1
    logits = logits / max(float(temperature), 1e-6)
    probs = torch.softmax(logits.view(b, -1), dim=-1).view(b, 1, h, w)
    return probs


def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)


def detach_alloc_dict(alloc):
    out = {}
    for k, v in alloc.items():
        if isinstance(v, dict):
            out[k] = {kk: (vv.detach() if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
        elif torch.is_tensor(v):
            out[k] = v.detach()
        else:
            out[k] = v
    return out

# =====================================================
# small helper modules
# =====================================================

def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * self.scale

# =====================================================
# sinusoidal positional embeds
# =====================================================

class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta=10000):
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
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

# =====================================================
# building block modules
# =====================================================

class Block(Module):
    def __init__(self, dim, dim_out, dropout=0.):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
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
    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-1), ((mk, k), (mv, v)))

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(Module):
    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4, flash=False):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash=flash)
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h=self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)

# =====================================================
# main U-Net with spatial timestep conditioning
# =====================================================

class UnetAllocator(Module):
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
        t = t_hw.unsqueeze(1).float()
        t_downs = [t]
        for _ in range(num_downs):
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

        if cond_map is None:  
            b, _, h_img, w_img = x.shape
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

# =====================================================
# schedules
# =====================================================

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# =====================================================
# differentiable extraction for continuous timestep maps
# =====================================================

def extract_continuous(a, t_hw, x_shape):
    T = a.shape[0]
    t_hw = t_hw.clamp(0, T - 1)

    t0 = t_hw.floor().long()
    t1 = (t0 + 1).clamp(max=T - 1)
    w1 = (t_hw - t0.float())
    w0 = 1. - w1

    b, h, w = t_hw.shape
    bank = a.view(1, -1).expand(b, -1)

    a0 = bank.gather(1, t0.view(b, -1)).view(b, 1, h, w)
    a1 = bank.gather(1, t1.view(b, -1)).view(b, 1, h, w)
    w0 = w0.view(b, 1, h, w)
    w1 = w1.view(b, 1, h, w)

    return w0 * a0 + w1 * a1


class ForwardNetMultiRes(Module):
    def __init__(
        self,
        in_channels=1,
        hidden_dim=64,
        use_time_cond=True,
        use_noise_map=True,
        use_multires=False,
        signed_shift=True,
        temperature=1.0,
        gate_scale=100,
        mask_thresh=0.05,
    ):
        super().__init__()
        self.use_time_cond = use_time_cond
        self.use_multires = use_multires
        self.signed_shift = signed_shift
        self.temperature = temperature
        self.gate_scale = gate_scale
        self.mask_thresh = mask_thresh
        self.use_noise_map = use_noise_map
        
        extra = 0
        if use_time_cond:
            extra += 1
        if use_noise_map:
            extra += 1

        in_ch = in_channels + extra
        out_heads = 2 if signed_shift else 1

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
        )
        self.down1 = nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1)

        self.enc2 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
        )
        self.down2 = nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1)

        self.enc3 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
        )

        self.head32 = nn.Conv2d(hidden_dim, out_heads, 3, padding=1)
        self.head16 = nn.Conv2d(hidden_dim, out_heads, 3, padding=1)
        self.head8 = nn.Conv2d(hidden_dim, out_heads, 3, padding=1)

        self.scale_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

    def _make_img_mask(self, x):
        # x expected normalized to [-1, 1]
        score = x.abs().mean(dim=1, keepdim=True)
        mask = (score > self.mask_thresh).float()

        mask_sum = mask.sum(dim=(-2, -1), keepdim=True)
        mask = torch.where(mask_sum > 0, mask, torch.ones_like(mask))
        return mask

    def logits_to_residual_gate(self, logits, mask=None, eps=1e-8):
        b, c, h, w = logits.shape

        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        logits = logits / max(float(self.temperature), 1e-4)

        logits_flat = logits.view(b, c, -1)
        logits_flat = logits_flat - logits_flat.amax(dim=-1, keepdim=True)
        logits_flat = logits_flat.clamp(-30.0, 30.0)

        prob = torch.softmax(logits_flat, dim=-1).view(b, c, h, w)
        
        # prob = torch.softmax(logits, dim=-1).view(b, c, h, w)

        if mask is not None:
            prob = prob * mask
            prob = prob / (prob.sum(dim=(-2, -1), keepdim=True) + eps)

            n = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
            p0 = mask / n
            scale = n
        else:
            p0 = torch.full_like(prob, 1.0 / float(h * w))
            scale = float(h * w / 10)

        # better than tanh(rel)
        # gate = self.gate_scale * (prob - p0) * scale
        gate = prob * scale
        # gate = gate.clamp(-self.gate_scale, self.gate_scale)
        # gate = gate.clamp(0, self.gate_scale)

        if mask is not None:
            gate = gate * mask

        # gate = torch.nan_to_num(gate, nan=0.0, posinf=self.gate_scale, neginf=-self.gate_scale)
        return gate, prob

    def forward(self, x, t_scalar=None,  noise_map=None):
        b, _, h, w = x.shape
        inputs = [x]
        
        if self.use_noise_map:
            assert noise_map is not None
            inputs.append(noise_map)
            
        if self.use_time_cond:
            assert t_scalar is not None
            t_map = t_scalar[:, None, None, None].expand(b, 1, h, w)
            inputs.append(t_map)

        inp = torch.cat(inputs, dim=1)

        # if self.use_time_cond:
        #     assert t_scalar is not None
        #     t_map = t_scalar[:, None, None, None].expand(b, 1, h, w)
        #     inp = torch.cat([x, t_map], dim=1)
        # else:
        #     inp = x

        f32 = self.enc1(inp)
        f16 = self.enc2(self.down1(f32))
        f8 = self.enc3(self.down2(f16))

        logit32 = self.head32(f32)

        if self.use_multires:
            logit16 = self.head16(f16)
            logit8 = self.head8(f8)

            up16 = F.interpolate(logit16, size=(h, w), mode='bilinear', align_corners=False)
            up8 = F.interpolate(logit8, size=(h, w), mode='bilinear', align_corners=False)

            w32, w16, w8 = torch.softmax(self.scale_weights, dim=0)
            logits_final = w32 * logit32 + w16 * up16 + w8 * up8
        else:
            logit16 = None
            logit8 = None
            up16 = None
            up8 = None
            logits_final = logit32

        # img_mask = self._make_img_mask(x)
        img_mask = None

        gates_final, probs_final = self.logits_to_residual_gate(
            logits_final,
            mask=img_mask,
        )

        aux = {
            'logits_final': logits_final,
            'probs_final': probs_final,
            'img_mask': img_mask,
            'logit32': logit32,
            'logit16': logit16,
            'logit8': logit8,
            'up16': up16,
            'up8': up8,
            'weights': torch.softmax(self.scale_weights, dim=0).detach(),
        }

        return gates_final, aux

# =====================================================
# discriminator-free diffusion wrapper
# =====================================================

class SpatialAllocationDiffusion(Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        sampling_timesteps=None,
        objective='pred_v',
        beta_schedule='sigmoid',
        schedule_fn_kwargs=dict(),
        ddim_sampling_eta=0.,
        auto_normalize=True,
        min_snr_loss_weight=False,
        min_snr_gamma=5,
        forward_hidden_dim=64,
        overlap_weight=0.05,
        balance_weight=0.05,
        collapse_weight = 0.1,
        rec_weight=1.0,
        budget_weight=0.1,
        tv_weight=1e-3,
        entropy_weight=1e-4,
        multi_res_weight=0.05,
        importance_temp=0.1,
        use_time_cond=True,
        use_multires=True,
        use_signed_shift=True,
        cond_dropout=0.1,
        # stop_path_b=True,
        # allocator_path_a_only=True,
        allocator_path_a_only = False,
        stop_path_b = False,
        budget_fraction=0.1,
        standard_fraction=0.5,
        use_alloc_cond=False,
    ):
        super().__init__()
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition
        self.objective = objective
        
        self.overlap_weight = overlap_weight
        self.balance_weight = balance_weight
        self.rec_weight = rec_weight
        self.budget_weight = budget_weight
        self.tv_weight = tv_weight
        self.entropy_weight = entropy_weight
        self.multi_res_weight = multi_res_weight
        self.importance_temp = importance_temp
        self.use_time_cond = use_time_cond
        self.use_multires = use_multires
        self.use_signed_shift = use_signed_shift
        self.cond_dropout = cond_dropout
        self.stop_path_b = stop_path_b
        self.allocator_path_a_only = allocator_path_a_only
        if not 0 < budget_fraction < 1:
            raise ValueError('budget_fraction must be in (0, 1)')
        if not 0 <= standard_fraction <= 1:
            raise ValueError('standard_fraction must be in [0, 1]')
        self.budget_fraction = budget_fraction
        self.standard_fraction = standard_fraction
        self.use_alloc_cond = use_alloc_cond
        self.collapse_weight = collapse_weight

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size

        self.forward_net = ForwardNetMultiRes(
            in_channels=self.channels,
            hidden_dim=forward_hidden_dim,
            use_time_cond=use_time_cond,
            use_multires=use_multires,
            signed_shift=use_signed_shift,
        )

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
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps_, = betas.shape
        self.num_timesteps = int(timesteps_)
        self.sampling_timesteps = default(sampling_timesteps, timesteps_)
        assert self.sampling_timesteps <= timesteps_
        self.is_ddim_sampling = self.sampling_timesteps < timesteps_
        self.ddim_sampling_eta = ddim_sampling_eta

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        snr = alphas_cumprod / (1 - alphas_cumprod)
        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device

    def _make_cond_map(self, p_up, p_down, signed_delta):
        """
        Build optional allocation-conditioning channels for the denoising U-Net.

        Inputs:
            p_up: (B, 1, H, W)
                Gate controlling where to increase noise variance.
                Despite its name, this is a fixed-budget gate, not a probability map.

            p_down: (B, 1, H, W) or None
                Gate controlling where to decrease noise variance.

            signed_delta: (B, 1, H, W) or None
                Actual timestep change: t_hw - t_base.
                Positive values mean later/noisier timesteps.
                Negative values mean earlier/cleaner timesteps.

        Returns:
            cond_map: (B, 3, H, W)
                Channel 0: noise-increase gate.
                Channel 1: noise-decrease gate.
                Channel 2: normalized timestep change.
        """

        # If noise-decrease allocation is disabled, use a zero channel
        # so the conditioning tensor still has three channels.
    
        if p_down is None:
            p_down = torch.zeros_like(p_up)
        if signed_delta is None:
            signed_delta = torch.zeros_like(p_up)
        delta_norm = signed_delta / float(max(self.num_timesteps - 1, 1))
        return torch.cat([p_up, p_down, delta_norm], dim=1)

    def _get_multi_res_consistency(self, aux, p_up_final, p_down_final=None):
        if (not self.use_multires) or aux['logit16'] is None or aux['logit8'] is None:
            return torch.tensor(0.0, device=p_up_final.device)

        p16_up = spatial_softmax(aux['logit16'][:, 0:1], self.importance_temp)
        p8_up = spatial_softmax(aux['logit8'][:, 0:1], self.importance_temp)
        p16_up = F.interpolate(p16_up, size=p_up_final.shape[-2:], mode='bilinear', align_corners=False)
        p8_up = F.interpolate(p8_up, size=p_up_final.shape[-2:], mode='bilinear', align_corners=False)
        loss = F.mse_loss(p16_up, p_up_final.detach()) + F.mse_loss(p8_up, p_up_final.detach())

        if self.use_signed_shift and p_down_final is not None:
            p16_down = spatial_softmax(aux['logit16'][:, 1:2], self.importance_temp)
            p8_down = spatial_softmax(aux['logit8'][:, 1:2], self.importance_temp)
            p16_down = F.interpolate(p16_down, size=p_down_final.shape[-2:], mode='bilinear', align_corners=False)
            p8_down = F.interpolate(p8_down, size=p_down_final.shape[-2:], mode='bilinear', align_corners=False)
            loss = loss + F.mse_loss(p16_down, p_down_final.detach()) + F.mse_loss(p8_down, p_down_final.detach())

        return loss
    
    def _fixed_budget_gate(self, prob):
        # Each image/head: mean = rho, range within [0, 1]. No post-clipping.
        p = prob.float()
        centered = p - p.mean(dim=(-2, -1), keepdim=True)
        radius = centered.abs().amax(dim=(-2, -1), keepdim=True)
        rho = self.budget_fraction
        return rho + min(rho, 1.0 - rho) * centered / (radius + 1e-6)

    def _variance_to_time(self, variance):
        # Inverse of the piecewise-linear variance schedule used in q_sample.
        bank = (1.0 - self.alphas_cumprod).float()
        v = variance.float().clamp(bank[0], bank[-1])
        hi = torch.searchsorted(bank, v.contiguous()).clamp(1, len(bank) - 1)
        lo = hi - 1
        frac = (v - bank[lo]) / (bank[hi] - bank[lo]).clamp_min(1e-12)
        return lo.float() + frac

    def sample_learned_tmap(self, x, noise_map=None):
        b, _, h, w = x.shape
        T = self.num_timesteps
        t_base = torch.randint(0, T, (b,), device=x.device)
        t_norm = t_base.float() / max(T - 1, 1)
        _, aux = self.forward_net(x, t_norm if self.use_time_cond else None,
                                  noise_map=noise_map)
        gates = self._fixed_budget_gate(aux['probs_final'])
        up = gates[:, 0:1]
        down = gates[:, 1:2] if self.use_signed_shift else None
        bank = 1.0 - self.alphas_cumprod
        base = bank[t_base].view(b, 1, 1, 1)
        # Per-image fixed headroom budgets, independent of allocator parameters.
        # Means: E_hw[dv_up] = rho * up_budget, similarly for down.
        up_budget = torch.rand(b, 1, 1, 1, device=x.device) * (bank[-1] - base)
        down_budget = torch.rand(b, 1, 1, 1, device=x.device) * (base - bank[0]) if down is not None else torch.zeros_like(base)
        dv_up = up * up_budget
        dv_down = down * down_budget if down is not None else torch.zeros_like(dv_up)
        t_hw = self._variance_to_time(base + dv_up - dv_down).squeeze(1)
        base_time = t_base[:, None, None, None].float()
        delta_up = self._variance_to_time(base + dv_up) - base_time
        delta_down = base_time - self._variance_to_time(base - dv_down)
        delta = t_hw[:, None] - base_time
        return dict(t_hw=t_hw, t_base=t_base, t_norm=t_norm,
                    signed_delta=delta, delta_up=delta_up, delta_down=delta_down,
                    p_up=up, p_down=down, gate_up=up, gate_down=down,
                    logits_final=aux['logits_final'], aux=aux,
                    shift_scale=float(T - 1), img_mask=aux['img_mask'],
                    variance_budget_up=up_budget, variance_budget_down=down_budget)

    def _standard_allocation(self, img):
        b, _, h, w = img.shape
        t = torch.randint(0, self.num_timesteps, (b,), device=img.device)
        z = torch.zeros(b, 1, h, w, device=img.device)
        return dict(t_hw=t[:, None, None].expand(b, h, w).float(),
                    t_base=t, signed_delta=z, p_up=z, p_down=None)

    def _signal_noise(self, t, shape):
        alpha = extract_continuous(self.alphas_cumprod, t, shape).float()
        return alpha.sqrt(), (1.0 - alpha).clamp_min(0).sqrt()

    def predict_start_from_noise(self, x_t, t, noise):
        a, sigma = self._signal_noise(t, x_t.shape)
        return (x_t - sigma * noise) / a.clamp_min(1e-8)

    def predict_noise_from_start(self, x_t, t, x0):
        a, sigma = self._signal_noise(t, x_t.shape)
        return (x_t - a * x0) / sigma.clamp_min(1e-8)

    def predict_v(self, x_start, t, noise):
        a, sigma = self._signal_noise(t, x_start.shape)
        return a * noise - sigma * x_start

    def predict_start_from_v(self, x_t, t, v):
        a, sigma = self._signal_noise(t, x_t.shape)
        return a * x_t - sigma * v

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_continuous(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract_continuous(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_continuous(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_continuous(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @autocast('cuda', enabled=False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        a, sigma = self._signal_noise(t, x_start.shape)
        return a * x_start + sigma * noise

    def model_predictions(self, x, t, x_self_cond=None, cond_map=None, clip_x_start=False, rederive_pred_noise=False):
        model_output = self.model(x, t, x_self_cond, cond_map=cond_map)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

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
        else:
            raise ValueError(f'unknown objective {self.objective}')

        return ModelPrediction(pred_noise, x_start)

    def shared_step_state(self, img, *, detach_allocator=False, standard=False, mix_standard=False):
        img = self.normalize(img)
        
        noise = torch.randn_like(img)
        noise_map = noise.abs().mean(dim=1, keepdim=True)

        alloc = self._standard_allocation(img) if standard else self.sample_learned_tmap(img, noise_map=noise_map)

        # alloc = self.sample_learned_tmap(img)
        if detach_allocator:
            alloc = detach_alloc_dict(alloc)

        if mix_standard and not standard:
            standard_alloc = self._standard_allocation(img)
            # Exact count per batch; randomly choose which examples are standard.
            n = round(img.shape[0] * self.standard_fraction)
            mask = torch.zeros(img.shape[0], device=img.device, dtype=torch.bool)
            mask[torch.randperm(img.shape[0], device=img.device)[:n]] = True
            for key in ('t_hw', 't_base', 'signed_delta', 'p_up', 'p_down'):
                if alloc[key] is not None:
                    ref = standard_alloc[key]
                    if ref is None:
                        ref = torch.zeros_like(alloc[key])
                    shape = (img.shape[0],) + (1,) * (alloc[key].ndim - 1)
                    alloc[key] = torch.where(mask.view(shape), ref, alloc[key])
        t_hw = alloc['t_hw']
        path_a_only = self.allocator_path_a_only and not detach_allocator
        # Retain the original independent noise draw for actual corruption.
        noise = torch.randn_like(img)
        x_t = self.q_sample(img, t_hw, noise=noise)  # Path A: gradient flows through noising

        x_self_cond = None
        cond_map = self._make_cond_map(alloc['p_up'], alloc['p_down'], alloc['signed_delta'])

        # stop_path_b: detach t_hw from the UNet time embedding so allocator gradient
        # only flows through the noised image (Path A), not the time conditioning (Path B).
        if not self.use_alloc_cond:
            cond_map = torch.zeros_like(cond_map)
        elif path_a_only:
            cond_map = cond_map.detach()
        t_hw_for_emb = t_hw.detach() if (self.stop_path_b or path_a_only) else t_hw
        t_target = t_hw.detach() if path_a_only else t_hw

        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x_t, t_hw_for_emb, cond_map=cond_map).pred_x_start
                x_self_cond.detach_()

        model_out = self.model(x_t, t_hw_for_emb, x_self_cond, cond_map)

        if self.objective == 'pred_noise':
            target = noise
            pred_noise = model_out
            x_recon = self.predict_start_from_noise(x_t, t_target, pred_noise)
        elif self.objective == 'pred_x0':
            target = img
            x_recon = model_out
            pred_noise = self.predict_noise_from_start(x_t, t_hw, x_recon)
        elif self.objective == 'pred_v':
            target = self.predict_v(img, t_target, noise)
            v = model_out
            x_recon = self.predict_start_from_v(x_t, t_target, v)
            pred_noise = self.predict_noise_from_start(x_t, t_hw, x_recon)
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if tensor_has_bad(x_recon) or tensor_has_bad(t_hw) or tensor_has_bad(x_t) or tensor_has_bad(model_out):
            return None

        x_recon = x_recon.clamp(-1., 1.)

        return {
            'img': img,
            't_hw': t_hw,
            'loss_time': t_target,
            'noise': noise,
            'x_t': x_t,
            'target': target,
            'model_out': model_out,
            'pred_noise': pred_noise,
            'x_recon': x_recon,
            'alloc': alloc,
            'cond_map': cond_map,
        }

    def diffusion_loss_from_state(self, state):
        model_out = state['model_out']
        target = state['target']
        t = state.get('loss_time', state['t_hw'])

        loss = F.mse_loss(model_out, target, reduction='none')
        loss = loss * extract_continuous(self.loss_weight, t, loss.shape)
        loss = loss.mean(dim=(1, 2, 3))
        return loss.mean()

    
    def _allocation_regularizers(self, img, alloc):
        signed_delta = alloc['signed_delta']

        # 如果有 gate，就用 gate；否则 fallback 到 p
        gate_up = alloc.get('gate_up', alloc['p_up'])
        gate_down = alloc.get('gate_down', alloc['p_down'])

        reg_tv = tv_loss(
            signed_delta / float(max(self.num_timesteps - 1, 1))
        )

        if self.use_signed_shift and gate_down is not None:
            overlap_loss = (gate_up * gate_down).mean()

            up_strength = gate_up.mean()
            down_strength = gate_down.mean()
            balance_loss = (up_strength - down_strength).pow(2)

            collapse_reward = torch.minimum(up_strength, down_strength)
        else:
            overlap_loss = torch.tensor(0.0, device=img.device)
            balance_loss = torch.tensor(0.0, device=img.device)
            collapse_reward = torch.tensor(0.0, device=img.device)

        # Budget equality is enforced by construction in variance space.
        budget_loss = torch.tensor(0.0, device=img.device)

        # binary entropy for gate map
        def binary_entropy(g, eps=1e-8):
            g = g.clamp(eps, 1. - eps)
            return -(g * g.log() + (1. - g) * (1. - g).log()).mean()

        entropy_up = binary_entropy(gate_up)

        if self.use_signed_shift and gate_down is not None:
            entropy_down = binary_entropy(gate_down)
            entropy = 0.5 * (entropy_up + entropy_down)
        else:
            entropy_down = torch.tensor(0.0, device=img.device)
            entropy = entropy_up

        multi_res_loss = self._get_multi_res_consistency(
            alloc['aux'],
            alloc['p_up'],
            alloc['p_down']
        )

        return {
            'budget': budget_loss,
            'tv': reg_tv,
            'multi_res': multi_res_loss,
            'entropy': entropy,
            'entropy_down': entropy_down,
            'overlap': overlap_loss,
            'balance': balance_loss,
            'collapse_loss': collapse_reward,
            'signed_delta': signed_delta,
        }

    def diffusion_player_loss_dict(self, img, state=None, *, standard=False):
        if state is None:
            state = self.shared_step_state(img, detach_allocator=True, standard=standard, mix_standard=not standard)
        if state is None:
            raise BadStepError('shared_step_state returned None')

        img = state['img']
        x_recon = state['x_recon']
        diff_loss = self.diffusion_loss_from_state(state)
        rec_loss = F.mse_loss(x_recon, img)
        total = diff_loss + self.rec_weight * rec_loss

        signed_delta = state['alloc']['signed_delta']
        return {
            'total': total,
            'diff': diff_loss.detach(),
            'rec': rec_loss.detach(),
            'delta_mean': signed_delta.mean().detach(),
            'delta_std': signed_delta.std().detach(),
            't_mean': state['t_hw'].mean().detach(),
            't_std': state['t_hw'].std().detach(),
            'x_recon': x_recon,
            'state': state,
        }

    def allocator_adv_loss_dict(self, img, state=None, adv_weight=1.0, include_rec_in_adv=False):
        if state is None:
            state = self.shared_step_state(img, detach_allocator=False)
        if state is None:
            raise BadStepError('shared_step_state returned None')

        img = state['img']
        x_recon = state['x_recon']
        alloc = state['alloc']

        diff_loss = self.diffusion_loss_from_state(state)
        rec_loss = F.mse_loss(x_recon, img)
        task_loss = diff_loss + (self.rec_weight * rec_loss if include_rec_in_adv else 0.0)
        # delta_penalty = (alloc['signed_delta'] / alloc['shift_scale']).pow(2).mean()

        regs = self._allocation_regularizers(img, alloc)
        total = (
            -adv_weight * task_loss
            # + self.tv_weight * regs['tv']
            # + self.overlap_weight * regs['overlap']
            # + self.balance_weight * regs['balance']
            # - self.entropy_weight * regs['entropy']  # maximise gate entropy → stay spread
            # + self.budget_weight * regs['budget']
            # + self.multi_res_weight * regs['multi_res']
        )

        signed_delta = regs['signed_delta']
        return {
            'total': total,
            'task': task_loss.detach(),
            'diff': diff_loss.detach(),
            'overlap': regs['overlap'].detach(),
            'balance': regs['balance'].detach(),
            'rec': rec_loss.detach(),
            'budget': regs['budget'].detach(),
            'tv': regs['tv'].detach(),
            'multi_res': regs['multi_res'].detach(),
            'entropy': regs['entropy'].detach(),
            'entropy_down': regs['entropy_down'].detach(),
            'delta_mean': signed_delta.mean().detach(),
            'delta_std': signed_delta.std().detach(),
            't_mean': state['t_hw'].mean().detach(),
            't_std': state['t_hw'].std().detach(),
            'x_recon': x_recon,
            'state': state,
        }

    def generator_main_loss_dict(self, img, state=None):
        return self.diffusion_player_loss_dict(img, state=state)

    @torch.inference_mode()
    def ddim_sample(self, shape, return_all_timesteps=False):
        batch, device, total_timesteps, sampling_timesteps, eta = (
            shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
        )
        _, _, H, W = shape

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=device)
        imgs = [img]
        x_start = None

        for time, time_next in time_pairs:
            t_hw = torch.full((batch, H, W), float(time), device=device)
            self_cond = x_start if self.self_condition else None
            cond_map = torch.zeros(batch, self.model.cond_channels, H, W, device=device, dtype=img.dtype)

            pred_noise, x_start = self.model_predictions(
                img,
                t_hw,
                self_cond,
                cond_map=cond_map,
                clip_x_start=True,
                rederive_pred_noise=True
            )

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        return self.unnormalize(ret)

    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond=None):
        b, *_, device = *x.shape, self.device
        t_hw = torch.full((b, x.shape[-2], x.shape[-1]), float(t), device=device)
        cond_map = torch.zeros(b, self.model.cond_channels, x.shape[-2], x.shape[-1], device=device, dtype=x.dtype)

        preds = self.model_predictions(
            x,
            t_hw,
            x_self_cond,
            cond_map=cond_map,
            clip_x_start=True,
            rederive_pred_noise=True
        )
        x_start = preds.pred_x_start
        model_mean, _, model_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t_hw)
        noise = torch.randn_like(x) if t > 0 else 0.
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps=False):
        batch, device = shape[0], self.device
        img = torch.randn(shape, device=device)
        imgs = [img]
        x_start = None

        for t in reversed(range(0, self.num_timesteps)):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        return self.unnormalize(ret)

    @torch.inference_mode()
    def sample(self, batch_size=16, return_all_timesteps=False):
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, h, w), return_all_timesteps=return_all_timesteps)

# =====================================================
# dataset helpers
# =====================================================

class FolderDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts=('jpg', 'jpeg', 'png', 'tiff'),
        augment_horizontal_flip=False,
        convert_image_to=None
    ):
        super().__init__()
        self.paths = [p for ext in exts for p in Path(folder).glob(f'**/*.{ext}')]
        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()
        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = Image.open(self.paths[index])
        return self.transform(img)


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

# =====================================================
# trainer
# =====================================================

class TrainerAllocator:
    def __init__(
        self,
        diffusion_model,
        folder=None,
        *,
        dataset='mnist_white_border',
        train_batch_size=16,
        gradient_accumulate_every=1,
        augment_horizontal_flip=True,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=25,
        results_folder='./results_allocator',
        amp=False,
        mixed_precision_type='fp16',
        split_batches=True,
        convert_image_to=None,
        calculate_fid=True,
        inception_block_idx=2048,
        max_grad_norm=1.,
        num_fid_samples=50000,
        save_best_and_latest_only=False,
        adv_weight=0.1,
        adv_update_every=1,
        adv_start_step=0,
        alloc_lr_scale=0.1,
        include_rec_in_adv=False,
        log_every=500,
        log_tmap_every=2000,
        log_tmap_num_samples=8,
        alloc_sched='step_start',
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else 'no'
        )

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        if not exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.max_grad_norm = max_grad_norm
        self.log_every = log_every
        self.log_tmap_every = log_tmap_every
        self.log_tmap_num_samples = log_tmap_num_samples

        if dataset == 'cifar10':
            self.ds = datasets.CIFAR10(
                root='data',
                download=True,
                train=True,
                transform=transforms.Compose([transforms.ToTensor()])
            )
        elif dataset == 'mnist_circle':
            self.ds = MNISTCircleDataset(root='data', train=True, download=True)
        else:
            base_ds = datasets.MNIST(
                root='data',
                train=True,
                download=True,
                transform=transforms.ToTensor()
            )
            pad_aug = AddWhiteEdgeOutside(pad=2, value=1.0)
            self.ds = SyntheticPaddedDataset(
                base_dataset=base_ds,
                pad_transform=pad_aug,
                to_minus1_1=False
            )

        assert len(self.ds) >= 100
        dl = DataLoader(
            self.ds,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=min(4, cpu_count())
        )
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        self.adv_weight = adv_weight
        self.adv_update_every = adv_update_every
        self.adv_start_step = adv_start_step
        self.include_rec_in_adv = include_rec_in_adv

        self.opt_model = torch.optim.Adam(self.model.model.parameters(), lr=train_lr, betas=adam_betas)
        self.opt_alloc = torch.optim.Adam(self.model.forward_net.parameters(), lr=train_lr * alloc_lr_scale, betas=adam_betas)

        self.alloc_sched_mode = alloc_sched
        self.sched_alloc = None
        if alloc_sched == 'warmup_cosine':
            alloc_warmup_steps = 10000
            total_alloc_steps = max(2, (train_num_steps - adv_start_step + adv_update_every - 1) // adv_update_every)
            alloc_warmup_steps = min(alloc_warmup_steps, total_alloc_steps - 1)
            alloc_warmup = LinearLR(self.opt_alloc, start_factor=1e-6, end_factor=1.0, total_iters=alloc_warmup_steps)
            alloc_cosine = CosineAnnealingLR(self.opt_alloc, T_max=total_alloc_steps - alloc_warmup_steps)
            self.sched_alloc = SequentialLR(self.opt_alloc, schedulers=[alloc_warmup, alloc_cosine], milestones=[alloc_warmup_steps])
            self.model, self.opt_model, self.opt_alloc, self.sched_alloc = self.accelerator.prepare(
                self.model, self.opt_model, self.opt_alloc, self.sched_alloc
            )
        else:
            self.model, self.opt_model, self.opt_alloc = self.accelerator.prepare(self.model, self.opt_model, self.opt_alloc)
        
        self.vis_batch = None
        if self.accelerator.is_main_process:
            try:
                vis_batch = self._extract_images(next(self.dl))
                vis_batch = vis_batch[:self.log_tmap_num_samples].to(self.device)
                self.vis_batch = vis_batch
            except Exception as e:
                self.accelerator.print(f'failed to create vis batch: {e}')
                self.vis_batch = None

        if self.accelerator.is_main_process:
            self.ema = EMA(
                self.accelerator.unwrap_model(self.model),
                beta=ema_decay,
                update_every=ema_update_every
            )
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.step = 0

        self.calculate_fid = bool(calculate_fid) and self.accelerator.is_main_process
        if self.calculate_fid:
            from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation
            if not is_ddim_sampling:
                self.accelerator.print('WARNING: DDPM FID is expensive. DDIM is faster for evaluation.')
            fid_stats_dir = os.path.join(os.path.dirname(results_folder.rstrip('/')), 'fid_stats')
            os.makedirs(fid_stats_dir, exist_ok=True)
            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=fid_stats_dir,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx,
                stats_name=dataset,
            )

        self.save_best_and_latest_only = save_best_and_latest_only
        self.best_fid = 1e10

    @property
    def device(self):
        return self.accelerator.device
    
    def _to_display_image(self, x):
        """
        x: (C,H,W), expected in [0,1] or [-1,1]
        return: numpy image for matplotlib
        """
        x = x.detach().float().cpu()

        if x.ndim == 3 and x.shape[0] == 1:
            img = x[0]
            # robust display
            if img.min() < 0:
                img = (img + 1.) * 0.5
            img = img.clamp(0, 1).numpy()
            return img

        if x.ndim == 3 and x.shape[0] == 2:
            if x.min() < 0:
                x = (x + 1.) * 0.5
            x = x.clamp(0, 1)
            zeros = torch.zeros(1, x.shape[1], x.shape[2])
            img = torch.cat([x, zeros], dim=0).permute(1, 2, 0).numpy()
            return img

        if x.ndim == 3 and x.shape[0] == 3:
            if x.min() < 0:
                x = (x + 1.) * 0.5
            img = x.clamp(0, 1).permute(1, 2, 0).numpy()
            return img

        raise ValueError(f'unsupported image shape: {x.shape}')
    
    @torch.inference_mode()
    def _log_tmap_visuals(self):
        if (not self.accelerator.is_main_process) or (self.vis_batch is None):
            return

        model = self.accelerator.unwrap_model(self.model)
        was_training = model.training
        model.eval()

        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = None
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state_all()

        try:
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)

            x = self.vis_batch.to(self.device)
            x_norm = model.normalize(x)

            # same noise_map used by allocator visualization
            noise = torch.randn_like(x_norm)
            noise_map = noise.abs().mean(dim=1, keepdim=True)

            alloc = (model._standard_allocation(x_norm) if self.step < self.adv_start_step
                     else model.sample_learned_tmap(x_norm, noise_map=noise_map))

            t_hw = alloc['t_hw'].detach().float().cpu()
            t_base = alloc['t_base'].detach().float().cpu()
            signed_delta = alloc['signed_delta'].detach().float().cpu()
            p_up = alloc['p_up'].detach().float().cpu()

            p_down = alloc['p_down']
            if p_down is not None:
                p_down = p_down.detach().float().cpu()

            noise_map_cpu = noise_map.detach().float().cpu()

            num_show = min(x.shape[0], self.log_tmap_num_samples)
            images_to_log = []

            for i in range(num_show):
                cols = 6 if p_down is not None else 5
                fig, axes = plt.subplots(1, cols, figsize=(3.2 * cols, 3.2))

                img_np = self._to_display_image(x[i])
                axes[0].imshow(img_np, cmap='gray' if img_np.ndim == 2 else None)
                axes[0].set_title(f'img\nbase t={t_base[i].item():.1f}')
                axes[0].axis('off')

                im1 = axes[1].imshow(
                    t_hw[i].numpy(),
                    cmap='plasma',
                    vmin=0,
                    vmax=model.num_timesteps - 1
                )
                axes[1].set_title('t_hw')
                axes[1].axis('off')
                plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

                delta_i = signed_delta[i, 0].numpy()
                vmax = max(abs(delta_i.min()), abs(delta_i.max()), 1e-6)
                im2 = axes[2].imshow(
                    delta_i,
                    cmap='coolwarm',
                    vmin=-vmax,
                    vmax=vmax
                )
                axes[2].set_title('signed_delta')
                axes[2].axis('off')
                plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

                im_noise = axes[3].imshow(
                    noise_map_cpu[i, 0].numpy(),
                    cmap='gray'
                )
                axes[3].set_title('noise_map')
                axes[3].axis('off')
                plt.colorbar(im_noise, ax=axes[3], fraction=0.046, pad=0.04)

                im3 = axes[4].imshow(p_up[i, 0].numpy(), cmap='plasma')
                axes[4].set_title('p_up')
                axes[4].axis('off')
                plt.colorbar(im3, ax=axes[4], fraction=0.046, pad=0.04)

                if p_down is not None:
                    im4 = axes[5].imshow(p_down[i, 0].numpy(), cmap='plasma')
                    axes[5].set_title('p_down')
                    axes[5].axis('off')
                    plt.colorbar(im4, ax=axes[5], fraction=0.046, pad=0.04)

                fig.tight_layout()
                images_to_log.append(
                    wandb.Image(fig, caption=f'step={self.step} sample={i}')
                )
                plt.close(fig)

            log_dict = {
                'sanity/t_hw_maps': images_to_log,
                'sanity/t_hw_mean': t_hw.mean().item(),
                'sanity/t_hw_std': t_hw.std().item(),
                'sanity/delta_mean': signed_delta.mean().item(),
                'sanity/delta_std': signed_delta.std().item(),
                'sanity/p_up_mean': p_up.mean().item(),
                'sanity/p_up_max': p_up.max().item(),
                'sanity/noise_map_mean': noise_map_cpu.mean().item(),
                'sanity/noise_map_std': noise_map_cpu.std().item(),
            }

            if p_down is not None:
                log_dict['sanity/p_down_mean'] = p_down.mean().item()
                log_dict['sanity/p_down_max'] = p_down.max().item()

            wandb.log(log_dict, step=self.step)

        except Exception as e:
            self.accelerator.print(
                f'failed to log tmap visuals at step {self.step}: {e}'
            )

        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

            if was_training:
                model.train()

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.step,
            'diffusion_wrapper': self.accelerator.get_state_dict(self.model),
            'opt_model': self.opt_model.state_dict(),
            'opt_alloc': self.opt_alloc.state_dict(),
            'sched_alloc': self.sched_alloc.state_dict() if exists(self.sched_alloc) else None,
            'ema': self.ema.state_dict() if self.accelerator.is_main_process else None,
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__,
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)
        self.accelerator.unwrap_model(self.model).load_state_dict(data['diffusion_wrapper'])
        self.step = data['step']
        self.opt_model.load_state_dict(data['opt_model'])
        self.opt_alloc.load_state_dict(data['opt_alloc'])
        if exists(self.sched_alloc) and exists(data.get('sched_alloc')):
            self.sched_alloc.load_state_dict(data['sched_alloc'])
        if self.accelerator.is_main_process and exists(data['ema']):
            self.ema.load_state_dict(data['ema'])
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def _extract_images(self, batch):
        if isinstance(batch, (list, tuple)):
            return batch[0]
        return batch

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                self.model.train()
                total_main_loss = 0.0
                total_alloc_loss = 0.0
                latest_main_logs = {}
                latest_alloc_logs = {}

                # -----------------------------
                # 1) diffusion player step
                # -----------------------------
                self.opt_model.zero_grad(set_to_none=True)
                try:
                    set_requires_grad(self.model.model, True)
                    set_requires_grad(self.model.forward_net, False)

                    for _ in range(self.gradient_accumulate_every):
                        data = self._extract_images(next(self.dl)).to(device)
                        with self.accelerator.autocast():
                            out_main = self.model.diffusion_player_loss_dict(
                                data, standard=self.step < self.adv_start_step
                            )
                            loss_main = out_main['total'] / self.gradient_accumulate_every
                        self.accelerator.backward(loss_main)
                        total_main_loss += loss_main.item()

                    accelerator.clip_grad_norm_(list(self.model.model.parameters()), self.max_grad_norm)
                    self.opt_model.step()

                    for k, v in out_main.items():
                        if isinstance(v, torch.Tensor) and v.ndim == 0:
                            latest_main_logs[k] = v.detach()
                except BadStepError:
                    accelerator.print(f'[skip model step {self.step}] bad state')
                    self.opt_model.zero_grad(set_to_none=True)

                # -----------------------------
                # 2) allocator adversary step
                # -----------------------------
                do_adv = (self.step >= self.adv_start_step) and divisible_by(self.step, self.adv_update_every)
                if do_adv:
                    self.opt_alloc.zero_grad(set_to_none=True)
                    try:
                        set_requires_grad(self.model.model, False)
                        set_requires_grad(self.model.forward_net, True)

                        for _ in range(self.gradient_accumulate_every):
                            data = self._extract_images(next(self.dl)).to(device)
                            with self.accelerator.autocast():
                                out_alloc = self.model.allocator_adv_loss_dict(
                                    data,
                                    adv_weight=self.adv_weight,
                                    include_rec_in_adv=self.include_rec_in_adv,
                                )
                                loss_alloc = out_alloc['total'] / self.gradient_accumulate_every
                            self.accelerator.backward(loss_alloc)
                            total_alloc_loss += loss_alloc.item()

                        accelerator.clip_grad_norm_(list(self.model.forward_net.parameters()), self.max_grad_norm)
                        self.opt_alloc.step()
                        if exists(self.sched_alloc):
                            self.sched_alloc.step()

                        for k, v in out_alloc.items():
                            if isinstance(v, torch.Tensor) and v.ndim == 0:
                                latest_alloc_logs[k] = v.detach()
                    except BadStepError:
                        accelerator.print(f'[skip alloc step {self.step}] bad state')
                        self.opt_alloc.zero_grad(set_to_none=True)

                accelerator.wait_for_everyone()
                self.step += 1

                if accelerator.is_main_process:
                    self.ema.update()

                desc = f'main: {total_main_loss:.4f}'
                if do_adv:
                    desc += f' | alloc: {total_alloc_loss:.4f}'
                for name in ['diff', 'rec']:
                    if name in latest_main_logs:
                        desc += f' | m_{name}: {latest_main_logs[name].item():.4f}'
                for name in ['task', 'budget', 'tv', 'multi_res', 'overlap', 'balance', 'entropy']:
                    if name in latest_alloc_logs:
                        desc += f' | a_{name}: {latest_alloc_logs[name].item():.4f}'
                pbar.set_description(desc)

                if accelerator.is_main_process:
                    log_payload = {'train/loss': total_main_loss}
                    if do_adv:
                        log_payload['train/alloc_loss'] = total_alloc_loss
                    if exists(self.sched_alloc):
                        log_payload['train/alloc_lr'] = self.sched_alloc.get_last_lr()[0]

                    if do_adv and latest_alloc_logs and divisible_by(self.step, self.log_every):
                        # collapse indicators — most important diagnostics
                        for key in ('t_std', 'delta_std', 'task'):
                            v = latest_alloc_logs.get(key)
                            if isinstance(v, torch.Tensor) and v.ndim == 0:
                                log_payload[f'alloc/{key}'] = v.item()
                        # regularizer breakdown
                        for key in ('tv', 'entropy', 'balance', 'overlap'):
                            v = latest_alloc_logs.get(key)
                            if isinstance(v, torch.Tensor) and v.ndim == 0:
                                log_payload[f'reg/{key}'] = v.item()

                    wandb.log(log_payload, step=self.step)
                    
                    if self.step != 0 and divisible_by(self.step, self.log_tmap_every):
                        self._log_tmap_visuals()

                if accelerator.is_main_process and self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                    self.ema.ema_model.eval()
                    with torch.inference_mode():
                        milestone = self.step // self.save_and_sample_every
                        batches = num_to_groups(self.num_samples, self.batch_size)
                        all_images_list = [self.ema.ema_model.sample(batch_size=n) for n in batches]
                        all_images = torch.cat(all_images_list, dim=0)
                        img_path = str(self.results_folder / f'sample-{milestone}.png')
                        utils.save_image(all_images, img_path, nrow=int(math.sqrt(self.num_samples)))
                        wandb.log({'samples/grid': wandb.Image(img_path)}, step=self.step)

                    fid_score = None
                    if self.calculate_fid:
                        fid_score = float(self.fid_scorer.fid_score())
                        accelerator.print(f'fid_score: {fid_score}')
                        wandb.log({'metrics/fid_score': fid_score}, step=self.step)

                    if self.save_best_and_latest_only:
                        if fid_score is not None and fid_score < self.best_fid:
                            self.best_fid = fid_score
                            self.save('best')
                        self.save('latest')
                    else:
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
        
        
# =====================================================
# Fixed-U-Net diagnostic experiment (single device, FP32)
# =====================================================
class FixedUNetAdvExperiment:
    """Use this instead of TrainerAllocator.train() for the diagnostic experiment.

    Setup (keep your existing UnetAllocator / SpatialAllocationDiffusion config):
        diffusion = SpatialAllocationDiffusion(unet, ...).to(device)
        experiment = FixedUNetAdvExperiment(diffusion, lr=1e-5)
        experiment.load_unet('path/to/model-8.pt')
        experiment.fit(train_loader, val_loader, steps=2000,
                       eval_every=200, eval_batches=10, noise_repeats=4)

    train_loader and val_loader must be separate, reiterable DataLoaders.
    Inputs are the original [0,1] images when auto_normalize=True.
    This copies ONLY the raw U-Net from your TrainerAllocator checkpoint;
    initialize a fresh diffusion object to start with a fresh allocator.
    The existing budget construction is retained. Attack objective is the full
    weighted diffusion loss (default), or unclipped reconstruction MSE ('rec').
    No allocation conditioning, self-conditioning, EMA updates, or regularizers
    are used in this controlled diagnostic. The U-Net stays in eval mode.
    """

    def __init__(self, diffusion, lr=1e-5, objective='diff', max_grad_norm=1.0):
        if objective not in ('diff', 'rec'):
            raise ValueError("objective must be 'diff' or 'rec'")
        if next(diffusion.model.parameters()).dtype != torch.float32:
            raise ValueError('Use an FP32 diffusion model for this diagnostic.')
        self.diffusion = diffusion
        self.objective = objective
        self.max_grad_norm = max_grad_norm
        self.loaded_checkpoint = None
        self.steps_completed = 0
        self._freeze_unet()
        set_requires_grad(diffusion.forward_net, True)
        self.optimizer = torch.optim.Adam(
            diffusion.forward_net.parameters(), lr=lr, betas=(0.9, 0.99)
        )

    def _freeze_unet(self):
        # eval() fixes dropout behavior; requires_grad_(False) freezes weights.
        # Neither operation prevents gradients with respect to the U-Net input.
        self.diffusion.model.eval()
        set_requires_grad(self.diffusion.model, False)
        for p in self.diffusion.model.parameters():
            p.grad = None

    def load_unet(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=True
        )
        if 'diffusion_wrapper' not in checkpoint:
            raise ValueError('Expected your TrainerAllocator checkpoint with diffusion_wrapper.')
        state = checkpoint['diffusion_wrapper']
        unet_state = {k[len('model.'):]: v for k, v in state.items()
                      if k.startswith('model.')}
        if not unet_state:
            raise ValueError('No model.* U-Net parameters found in checkpoint.')
        self.diffusion.model.load_state_dict(unet_state, strict=True)
        self.loaded_checkpoint = str(checkpoint_path)
        self._freeze_unet()
        print('Loaded raw U-Net; source training step:', checkpoint.get('step'))
        print('Allocator and its optimizer were NOT loaded from this checkpoint.')

    @staticmethod
    def _images(batch):
        return batch[0] if isinstance(batch, (list, tuple)) else batch

    def _draw(self, batch):
        d = self.diffusion
        x = d.normalize(self._images(batch).to(d.device, dtype=torch.float32))
        auxiliary_noise = torch.randn_like(x)
        noise_map = auxiliary_noise.abs().mean(dim=1, keepdim=True)
        alloc = d.sample_learned_tmap(x, noise_map=noise_map)
        # Independent of the allocator's auxiliary noise, matching your code.
        actual_noise = torch.randn_like(x)
        return x, alloc, actual_noise

    def _variance_map(self, alloc, up, down):
        d = self.diffusion
        base = (1.0 - d.alphas_cumprod)[alloc['t_base']][:, None, None, None]
        variance = base + up * alloc['variance_budget_up']
        if down is not None:
            variance = variance - down * alloc['variance_budget_down']
        return variance

    def _evaluate_map(self, x, t_hw, noise):
        d = self.diffusion
        x_t = d.q_sample(x, t_hw, noise=noise)
        cond = x.new_zeros(x.shape[0], d.model.cond_channels, *x.shape[-2:])
        # Full derivatives: do NOT detach time, targets or weighting here.
        # Self-conditioning is deliberately fixed to zero for all comparisons.
        output = d.model(x_t, t_hw, x_self_cond=None, cond_map=cond)
        if d.objective == 'pred_v':
            target = d.predict_v(x, t_hw, noise)
            recon = d.predict_start_from_v(x_t, t_hw, output)
        elif d.objective == 'pred_noise':
            target = noise
            recon = d.predict_start_from_noise(x_t, t_hw, output)
        else:
            target = x
            recon = output
        diff = (output - target).square()
        diff = (diff * extract_continuous(d.loss_weight, t_hw, diff.shape)).mean((1, 2, 3))
        rec = (recon - x).square().mean((1, 2, 3))
        rec_clipped = (recon.clamp(-1, 1) - x).square().mean((1, 2, 3))
        # All returned metrics are per image, in normalized image units.
        return {'diff': diff, 'rec': rec, 'rec_clipped': rec_clipped}

    def train_step(self, batch):
        self._freeze_unet()
        self.diffusion.forward_net.train()
        self.optimizer.zero_grad(set_to_none=True)
        x, alloc, noise = self._draw(batch)
        metrics = self._evaluate_map(x, alloc['t_hw'], noise)
        task = metrics[self.objective].mean()
        if not torch.isfinite(task):
            raise FloatingPointError('Nonfinite attack objective; no optimizer update performed.')
        loss = -task  # minimize negative error = maximize error
        loss.backward()
        params = list(self.diffusion.forward_net.parameters())
        norm = torch.nn.utils.clip_grad_norm_(
            params, self.max_grad_norm, error_if_nonfinite=True
        )
        self.optimizer.step()
        self.steps_completed += 1
        return {'attack_loss': loss.item(), 'task_before_update': task.item(),
                'grad_norm': norm.item()}

    @torch.no_grad()
    def evaluate(self, val_loader, max_batches=10, noise_repeats=4, seed=1234):
        """Paired learned/uniform/shuffled comparison; restores training RNG.

        Shuffled uses one shared spatial permutation for both heads per image.
        This preserves their pairing and the final variance histogram exactly.
        It is a randomized placement baseline, not a separately trained network.
        Metrics are averaged over noise draws within each image; standard errors
        of paired gaps are then estimated across validation images.
        """
        import itertools
        if max_batches < 1 or noise_repeats < 1:
            raise ValueError('max_batches and noise_repeats must be positive')
        d = self.diffusion
        previous_mode = d.forward_net.training
        d.forward_net.eval()
        self._freeze_unet()
        modes = ('learned', 'uniform', 'shuffled')
        names = ('diff', 'rec', 'rec_clipped', 'variance_mean', 'variance_std')
        values = {mode: {name: [] for name in names} for mode in modes}
        cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(seed)
                if cuda_devices:
                    torch.cuda.manual_seed_all(seed)
                for batch in itertools.islice(val_loader, max_batches):
                    repeats = {m: {n: [] for n in names} for m in modes}
                    for _ in range(noise_repeats):
                        x, alloc, noise = self._draw(batch)
                        up, down = alloc['gate_up'], alloc['gate_down']
                        b, _, h, w = up.shape
                        perm = torch.stack([torch.randperm(h*w, device=x.device) for _ in range(b)])[:, None]
                        def shuffle(g):
                            return g.flatten(2).gather(2, perm).reshape_as(g)
                        gates = {
                            'learned': (up, down),
                            'uniform': (torch.full_like(up, d.budget_fraction),
                                        None if down is None else torch.full_like(down, d.budget_fraction)),
                            'shuffled': (shuffle(up), None if down is None else shuffle(down)),
                        }
                        reference_mean = None
                        for mode, (gu, gd) in gates.items():
                            variance = self._variance_map(alloc, gu, gd)
                            mean = variance.mean((1, 2, 3))
                            if reference_mean is None:
                                reference_mean = mean
                            elif not torch.allclose(mean, reference_mean, atol=2e-6, rtol=1e-5):
                                raise AssertionError('Paired mean variance budgets do not match')
                            time_map = d._variance_to_time(variance).squeeze(1)
                            metric = self._evaluate_map(x, time_map, noise)
                            metric['variance_mean'] = mean
                            metric['variance_std'] = variance.flatten(1).std(1, unbiased=False)
                            for name in names:
                                repeats[mode][name].append(metric[name].cpu())
                    for mode in modes:
                        for name in names:
                            values[mode][name].append(torch.stack(repeats[mode][name]).mean(0))
        finally:
            d.forward_net.train(previous_mode)
        if not values['learned']['diff']:
            raise ValueError('Validation loader was empty')
        all_values = {m: {n: torch.cat(values[m][n]) for n in names} for m in modes}
        result = {'n_images': all_values['learned']['diff'].numel(),
                  'noise_repeats': noise_repeats,
                  'objective': self.objective, 'step': self.steps_completed}
        for mode in modes:
            result[mode] = {n: a.mean().item() for n, a in all_values[mode].items()}
        result['paired_gaps'] = {}
        for baseline in ('uniform', 'shuffled'):
            result['paired_gaps'][baseline] = {}
            for name in ('diff', 'rec', 'rec_clipped'):
                gap = all_values['learned'][name] - all_values[baseline][name]
                se = (gap.std(unbiased=True) / gap.numel()**0.5).item() if gap.numel() > 1 else None
                result['paired_gaps'][baseline][name] = {'mean': gap.mean().item(), 'se_across_images': se}
        return result

    def fit(self, train_loader, val_loader, steps=2000, eval_every=200,
            eval_batches=10, noise_repeats=4, output_dir='fixed_unet_adv'):
        import json
        if steps < 1 or eval_every < 1:
            raise ValueError('steps and eval_every must be positive')
        if train_loader is val_loader:
            raise ValueError('Use a separate validation loader, not the training loader')
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        # Exact final check: the frozen U-Net state must not change.
        frozen = {k: v.detach().cpu().clone() for k, v in self.diffusion.model.state_dict().items()}
        def record_eval():
            result = self.evaluate(val_loader, max_batches=eval_batches, noise_repeats=noise_repeats)
            with (folder / 'evaluation.jsonl').open('a') as f:
                f.write(json.dumps(result) + '\n')
            print(json.dumps(result, indent=2))
        record_eval()  # baseline before training allocator
        iterator = iter(train_loader)
        for local_step in tqdm(range(1, steps + 1)):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                try:
                    batch = next(iterator)
                except StopIteration as exc:
                    raise ValueError('Training loader was empty') from exc
            log = self.train_step(batch)
            with (folder / 'train.jsonl').open('a') as f:
                f.write(json.dumps({'step': self.steps_completed, **log}) + '\n')
            if local_step % eval_every == 0 or local_step == steps:
                record_eval()
                torch.save({'forward_net': self.diffusion.forward_net.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'steps_completed': self.steps_completed,
                            'objective': self.objective,
                            'source_unet_checkpoint': self.loaded_checkpoint,
                            'budget_fraction': self.diffusion.budget_fraction,
                            'num_timesteps': self.diffusion.num_timesteps,
                            'alphas_cumprod': self.diffusion.alphas_cumprod.detach().cpu()},
                           folder / 'allocator-latest.pt')
        for k, v in self.diffusion.model.state_dict().items():
            if not torch.equal(frozen[k], v.detach().cpu()):
                raise AssertionError(f'Frozen U-Net changed: {k}')
        print('Finished: U-Net parameters and buffers are unchanged.')
