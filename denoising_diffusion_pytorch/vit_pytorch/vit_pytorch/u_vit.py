import torch
import torch.nn as nn
import math
from .timm import trunc_normal_, Mlp
import einops
import torch.utils.checkpoint
from collections import defaultdict

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
print(f'attention mode is {ATTENTION_MODE}')


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def unpatchify(x, coords, output_size, patch_size, channels=3):
    B, N, D = x.shape
    H, W = output_size
    p = patch_size

    canvas = torch.zeros(B, channels, H, W, device=x.device, dtype=x.dtype)
    for i, (yy, xx) in enumerate(coords):
        patch = x[:, i].reshape(B, channels, p, p)
        canvas[:, :, yy:yy + p, xx:xx + p] = patch
    return canvas


class SpiralPatchEmbed(nn.Module):
    def __init__(self, patch_size, embed_dim, subimage_metadata, in_chans=3):
        super().__init__()
        self.patch_size = patch_size
        self.subimage_metadata = subimage_metadata
        self.in_chans = in_chans
        self.proj = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Linear(patch_size * patch_size * in_chans, embed_dim)
        )

    def forward(self, x, return_coords=False):
        B, C, H, W = x.shape
        p = self.patch_size
        patches_all = []
        patch_coords = []
        patch_to_subimage = []

        for subimage_index, (x0, y0, w, h) in enumerate(self.subimage_metadata):
            x1 = min(x0 + w, W)
            y1 = min(y0 + h, H)

            for yy in range(y0, y1 - p + 1, p):
                for xx in range(x0, x1 - p + 1, p):
                    patch = x[:, :, yy:yy + p, xx:xx + p]
                    patches_all.append(patch)
                    patch_coords.append((yy, xx))
                    patch_to_subimage.append(subimage_index)

        patches = torch.stack(patches_all, dim=1)
        patches = patches.reshape(B, -1, C * p * p)
        out = self.proj(patches)

        N = len(patch_to_subimage)
        attn_mask = torch.full((N+1, N+1), float('-inf'), device=x.device)
        attn_mask[:,N] =0
        subimage_to_indices = defaultdict(list)
        for idx, sid in enumerate(patch_to_subimage):
            subimage_to_indices[sid].append(idx)
        
        sorted_sids = sorted(subimage_to_indices.keys())
        # fill attention mask such that: group i attends to itself and all groups after
        all_indices = []
        for sid in sorted_sids:
            indices = subimage_to_indices[sid]
            indices_tensor = torch.tensor(indices, device=x.device)
            all_indices.append(indices_tensor)

        # Construct cumulative structure
        for i, src_indices in enumerate(all_indices):
            visible_indices = torch.cat(all_indices[i:], dim=0)
            attn_mask[src_indices[:, None], visible_indices[None, :]] = 0

        if return_coords:
            return out, patch_coords, (H, W), attn_mask
        return out
    


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == 'flash':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask[None, None, :, :])
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if attn_mask is not None:
                attn = attn + attn_mask[None, None, :, :]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None,attn_mask=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip, attn_mask)

    def _forward(self, x, skip=None, attn_mask=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# [The rest of the Attention, Block, PatchEmbed, and unchanged classes remain the same.]

class UViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1,
                 use_checkpoint=False, conv=True, skip=True, spiral_metadata=None):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.spiral_metadata = spiral_metadata

        if spiral_metadata is not None:
            self.patch_embed = SpiralPatchEmbed(patch_size=patch_size, embed_dim=embed_dim,
                                                subimage_metadata=spiral_metadata, in_chans=in_chans)
            num_patches = sum((w // patch_size) * (h // patch_size) for _, _, w, h in spiral_metadata)

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        self.extras = 2 if num_classes > 0 else 1
        self.label_emb = nn.Embedding(num_classes, embed_dim) if num_classes > 0 else None

        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, embed_dim))

        self.in_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_scale=qk_scale, norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                               qk_scale=qk_scale, norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_scale=qk_scale, norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        self.patch_dim = patch_size ** 2 * in_chans
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)
        self.final_layer = nn.Conv2d(self.in_chans, self.in_chans, 3, padding=1) if conv else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, x, timesteps, y=None):
        if self.spiral_metadata is not None:
            x, coords, canvas_size, atten_mask = self.patch_embed(x, return_coords=True)
            
        B, L, D = x.shape

        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim)).unsqueeze(1)
        x = torch.cat((time_token, x), dim=1)

        if y is not None:
            label_emb = self.label_emb(y).unsqueeze(1)
            x = torch.cat((label_emb, x), dim=1)

        x = x + self.pos_embed[:, :x.size(1)]

        skips = []
        for blk in self.in_blocks:
            x = blk(x, attn_mask = atten_mask)
            skips.append(x)

        x = self.mid_block(x, attn_mask = atten_mask)

        for blk in self.out_blocks:
            x = blk(x, skips.pop(), attn_mask = atten_mask)

        x = self.norm(x)
        x = self.decoder_pred(x)
        x = x[:, self.extras:, :]

        if self.spiral_metadata is not None:
            x = unpatchify(x, coords=coords, output_size=canvas_size, patch_size=self.patch_size, channels=self.in_chans)
        else:
            h = w = int(math.sqrt(x.size(1)))
            x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=self.patch_size, p2=self.patch_size)

        return self.final_layer(x)
