import torch
from torch import nn
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert divisible_by(dim, 4), "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = temperature ** -omega

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# classes

def FeedForward(dim, hidden_dim):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, dim),
    )    

class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.depth = depth
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        for layer in range(1, depth + 1):
            latter_half = layer >= (depth / 2 + 1)

            self.layers.append(nn.ModuleList([
                nn.Linear(dim * 2, dim) if latter_half else None,
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))

    def forward(self, x):

        skips = []

        for ind, (combine_skip, attn, ff) in enumerate(self.layers):
            layer = ind + 1
            first_half = layer <= (self.depth / 2)

            if first_half:
                skips.append(x)

            if exists(combine_skip):
                skip = skips.pop()
                skip_and_x = torch.cat((skip, x), dim = -1)
                x = combine_skip(skip_and_x)

            x = attn(x) + x
            x = ff(x) + x

        assert len(skips) == 0

        return self.norm(x)

class SpiralPatchEmbedding(nn.Module):
    def __init__(self, patch_size, dim, subimage_metadata, channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.subimage_metadata = subimage_metadata  # List of (x, y, w, h) tuples
        patch_dim = channels * patch_size * patch_size

        self.norm1 = nn.LayerNorm(patch_dim)
        self.linear = nn.Linear(patch_dim, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):  # x: (b, c, H, W)
        B, C, H, W = x.shape
        p = self.patch_size
        patches_all = []

        for x0, y0, w, h in self.subimage_metadata:
            x1 = min(x0 + w, W)
            y1 = min(y0 + h, H)

            for yy in range(y0, y1 - p + 1, p):
                for xx in range(x0, x1 - p + 1, p):
                    patch = x[:, :, yy:yy + p, xx:xx + p]  # (b, c, p, p)
                    patches_all.append(patch)

        patches = torch.stack(patches_all, dim=1)  # (b, n_patches, c, p, p)
        patches = rearrange(patches, 'b n c h w -> b n (h w c)')

        patches = self.norm1(patches)
        patches = self.linear(patches)
        patches = self.norm2(patches)
        return patches  # (b, n_patches, dim)


class SimpleUViT(Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, subimage_metadata=[], num_register_tokens = 4, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert divisible_by(image_height, patch_height) and divisible_by(image_width, patch_width), 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = SpiralPatchEmbedding(patch_size=patch_height, dim=dim, subimage_metadata=subimage_metadata, channels=channels)
        pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim
        )

        self.register_buffer('pos_embedding', pos_embedding, persistent = False)

        self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        batch, device = img.shape[0], img.device

        x = self.to_patch_embedding(img)
        x = x + self.pos_embedding.type(x.dtype)

        r = repeat(self.register_tokens, 'n d -> b n d', b = batch)

        x, ps = pack([x, r], 'b * d')

        x = self.transformer(x)

        x, _ = unpack(x, ps, 'b * d')

        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)

# quick test on odd number of layers

if __name__ == '__main__':

    v = SimpleUViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 7,
        heads = 16,
        mlp_dim = 2048
    ).cuda()

    img = torch.randn(2, 3, 256, 256).cuda()

    preds = v(img)
    assert preds.shape == (2, 1000)
