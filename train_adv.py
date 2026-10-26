import torch
import numpy as np
import wandb
import argparse

from denoising_diffusion_pytorch.denoising_diffusion_pytorch_adv1 import (
    UnetAllocator,
    SpatialAllocationDiffusion,
    TrainerAllocator,
)

parser = argparse.ArgumentParser()
# training
parser.add_argument('--adv_weight', type=float, default=0.01)
parser.add_argument('--adv_update_every', type=int, default=2)
parser.add_argument('--adv_start_step', type=int, default=8000,
                    help='only used with --alloc_sched step_start')
parser.add_argument('--alloc_sched', type=str, default='step_start',
                    choices=['step_start', 'warmup_cosine'])
# allocator mode
parser.add_argument('--signed_shift', action='store_true',
                    help='enable gate_down (negative head). default: positive-only (gate_up only)')
parser.add_argument('--stop_path_b', action='store_true',
                    help='detach t_hw from UNet time embedding so allocator gradient only flows '
                         'through the noised image (Path A), not the time conditioning (Path B)')
# regularizer weights (zero to ablate; balance/overlap only apply when --signed_shift)
parser.add_argument('--entropy_weight', type=float, default=1e-4)
parser.add_argument('--tv_weight', type=float, default=1e-3)
parser.add_argument('--balance_weight', type=float, default=0.0)
parser.add_argument('--overlap_weight', type=float, default=0.0)
parser.add_argument('--alloc_lr_scale', type=float, default=0.1,
                    help='alloc LR = train_lr * alloc_lr_scale. original.py uses 0.5 (adv_weights)')
parser.add_argument('--run_name', type=str, default=None)
parser.add_argument('--dataset', type=str, default='mnist_white_border',
                    choices=['mnist_white_border', 'mnist_circle', 'cifar10'])
args = parser.parse_args()


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(3407)

dataset_channels = {'mnist_white_border': 1, 'mnist_circle': 2, 'cifar10': 3}
channels = dataset_channels[args.dataset]

model = UnetAllocator(
    dim=128,
    dim_mults=(1, 2, 2, 2),
    dropout=0.1,
    channels=channels,
)

diffusion = SpatialAllocationDiffusion(
    model,
    image_size=32,
    timesteps=1000,
    sampling_timesteps=250,
    objective='pred_v',
    use_multires=False,
    use_signed_shift=args.signed_shift,
    stop_path_b=args.stop_path_b,
    entropy_weight=args.entropy_weight,
    tv_weight=args.tv_weight,
    balance_weight=args.balance_weight,
    overlap_weight=args.overlap_weight,
)

run_name = args.run_name or (
    f'adv{args.adv_weight}_sched{args.alloc_sched}'
    f'_ue{args.adv_update_every}'
    + (f'_ss{args.adv_start_step}' if args.alloc_sched == 'step_start' else '')
)

# wandb.init(
#     project="block-wise-adv",
#     name=run_name,
# )
wandb.init(
    project="block-wise-noise",
    entity="dongweilyu-The University of Chicago",
    # name=f'cifar10_clusternet',
    # name=f'cifar10_wteacher_cluste_k16',
    name=run_name,
    # resume = 'allow',
    # id='ged1vj6s' 
)

# results_dir = f"/pscratch/sd/d/dwlyu/ddpm/results_adv_{run_name}/"
results_dir = f"/net/spaces/scratch/ctang-scratch2/adv_new/results_adv2_{run_name}/"

trainer = TrainerAllocator(
    diffusion,
    train_batch_size=128,
    train_lr=2e-4,
    train_num_steps=700000,
    gradient_accumulate_every=2,
    ema_decay=0.9999,
    amp=True,
    num_fid_samples=1000,
    results_folder=results_dir,
    save_and_sample_every=2000,
    adv_weight=args.adv_weight,
    adv_update_every=args.adv_update_every,
    adv_start_step=args.adv_start_step if args.alloc_sched == 'step_start' else 0,
    alloc_lr_scale=args.alloc_lr_scale,
    alloc_sched=args.alloc_sched,
    dataset=args.dataset,
    save_best_and_latest_only = False,
)
trainer.train()
