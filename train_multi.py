# from keras.datasets import mnist
import torch
from denoising_diffusion_pytorch import Unet_multi, GaussianDiffusion_multi, Trainer_multi
import numpy as np
# import tensorflow as tf
import wandb
import argparse
from denoising_diffusion_pytorch.vit_pytorch.vit_pytorch import UViT
parser = argparse.ArgumentParser()
parser.add_argument('--training_mode', type=str, default="fullres_only", help='multi-resolution training mode')
args = parser.parse_args()

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True
seed_everything(3407)

model = Unet_multi(
    dim = 128,
    dim_mults = (1, 2, 2, 2),
    dropout = 0.1,
    channels=3,
    out_dim = 3
)


diffusion = GaussianDiffusion_multi(
    model,
    image_size = 32,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250,
    objective = 'pred_v',   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

wandb.init(
    project="multi-stage-training",
    name=f'multi-{args.training_mode}-replicate',
)
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# x_train = np.asarray(x_train)
# x_train = x_train.astype(np.float16)
# new_train = torch.from_numpy(np.swapaxes(x_train,1,3))
# training_images = torch.randn(8, 3, 128, 128) # images are normalized from 0 to 1

trainer = Trainer_multi(
    diffusion,
    '/pscratch/sd/d/dwlyu/ddpm/',
    train_batch_size = 128,
    train_lr = 2e-4,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.9999,                # exponential moving average decay
    amp = True,                        # turn on mixed precision
    num_fid_samples = 1000,
    results_folder = f"/pscratch/sd/d/dwlyu/ddpm/results_multi_{args.training_mode}-replicate/",
    save_and_sample_every = 1000,
    training_scenario = args.training_mode
)
trainer.train()