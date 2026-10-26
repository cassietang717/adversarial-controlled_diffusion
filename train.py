import torch
from denoising_diffusion_pytorch import Unet_gen, GaussianDiffusion_orig, Trainer, Unet_dis
import numpy as np
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--adv_w', type=float, default=0.5, help='adversarial loss weight')
parser.add_argument('--tv_weight', type=float, default=5e-2, help='entropy regulariser weight on discriminator output')
parser.add_argument('--disc_activation', type=str, default='softmax',
                    choices=['softmax', 'sigmoid', 'tanh', 'softsign'],
                    help='output activation for discriminator: softmax (global, competitive), '
                         'sigmoid (per-pixel, independent), tanh (signed, tanh/norm), '
                         'softsign (signed, x/(1+|x|), no saturation)')
parser.add_argument('--disc_temp', type=float, default=2.0,
                    help='softmax temperature for discriminator (only used when disc_activation=softmax)')
parser.add_argument('--bidirectional', action='store_true',
                    help='allow discriminator to push timestep both up and down (requires tanh activation)')
parser.add_argument('--fixed_stochastic', type=float, default=None,
                    help='fix the stochastic step multiplier to this value instead of sampling randomly; '
                         'None = random, 0.5 = fixed half, 1.0 = full range')
parser.add_argument('--disc_tv_weight', type=float, default=0.0,
                    help='TV smoothness penalty weight on the spatial t_adv map')
parser.add_argument('--stop_path_b', action='store_true',
                    help='In discriminator step, detach t_adv before generator time embedding '
                         'so gradient only flows through the noised image (Path A), '
                         'not through the UNet time conditioning (Path B).')
parser.add_argument('--detach_sftm_denom', action='store_true',
                    help='Detach softmax denominator Z so cross-pixel competition is removed '
                         'from the gradient. Prevents winner-takes-all collapse via Z domination.')
parser.add_argument('--adv_start_step', type=int, default=0,
                    help='number of denoiser-only gen steps before adversarial training begins; '
                         '0 = start adv from the beginning (default)')
parser.add_argument('--run_name', type=str, default=None,
                    help='override wandb run name')
parser.add_argument('--dataset', type=str, default='mnist_white_border',
                    choices=['mnist_white_border', 'cifar10', 'mask_circle_mnist'],
                    help='training dataset: mnist_white_border (1ch), cifar10 (3ch), mask_circle_mnist (1ch)')
args = parser.parse_args()

_DATASET_CHANNELS = {'mnist_white_border': 1, 'cifar10': 3, 'mask_circle_mnist': 1}
channels = _DATASET_CHANNELS[args.dataset]

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(3407)

model = Unet_gen(
    dim=128,
    dim_mults=(1, 2, 2, 2),
    dropout=0.1,
    channels=channels,
    out_dim=channels
)

adv_model = Unet_dis(
    dim=32,
    dim_mults=(1, 2, 4),
    channels=channels,
    out_dim=1,
    use_attention=False,
    out_activation=args.disc_activation,
    softmax_temp=args.disc_temp,
    detach_sftm_denom=args.detach_sftm_denom,
)

diffusion = GaussianDiffusion_orig(
    model,
    adv_model,
    image_size=32,
    timesteps=1000,
    sampling_timesteps=250,
    objective='pred_v',
    fixed_stochastic=args.fixed_stochastic,
    disc_tv_weight=args.disc_tv_weight,
    bidirectional=args.bidirectional,
)

run_name = args.run_name or (
    f'adv{args.adv_w}_act{args.disc_activation}_temp{args.disc_temp}'
    f'_bidir{int(args.bidirectional)}_fs{args.fixed_stochastic}'
    f'_tv{args.disc_tv_weight}_spb{int(args.stop_path_b)}'
)

wandb.init(
    project="block-wise-test",
    name=run_name,
)

results_dir = f"/pscratch/sd/d/dwlyu/ddpm/results_{run_name}/"

trainer = Trainer(
    diffusion,
    '/pscratch/sd/d/dwlyu/ddpm/',
    train_batch_size=128,
    train_lr=2e-4,
    train_num_steps=700000,
    gradient_accumulate_every=2,
    ema_decay=0.9999,
    amp=True,
    num_fid_samples=1000,
    results_folder=results_dir,
    save_and_sample_every=2000,
    adv_weights=args.adv_w,
    tv_weight=args.tv_weight,
    disc_tv_weight=args.disc_tv_weight,
    stop_path_b_disc=args.stop_path_b,
    adv_start_step=args.adv_start_step,
    dataset=args.dataset,
)
trainer.train()
