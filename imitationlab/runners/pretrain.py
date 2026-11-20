# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml

from imitationlab.runners.args_file import expand_args_files
from imitationlab.runners.runtime_bootstrap import (
    assert_kit_not_loaded,
    config_contains_type_name,
    install_kit_import_guard,
)


sys.argv[1:] = expand_args_files(sys.argv[1:])
strict_kitless = "--assert-kitless" in sys.argv
if strict_kitless:
    install_kit_import_guard()

from imitationlab.runners.launch_compat import add_launcher_args

parser = argparse.ArgumentParser(
    description="Train an offline high-level skill encoder."
)
parser.add_argument(
    "--task",
    type=str,
    default="imitation-g1-v1",
    help="Isaac Lab task used only for expert trajectory sampling.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of envs.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Output directory. Defaults to outputs/encoders/<latent_mode>/<timestamp>.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Optional checkpoint to load before training or eval-only diagnostics.",
)
parser.add_argument(
    "--eval_only",
    action="store_true",
    default=False,
    help="Load --checkpoint and run held-out diagnostics without updates.",
)
parser.add_argument("--horizon_steps", type=int, default=25, help="Macro horizon W.")
parser.add_argument(
    "--encoder_window_mode",
    type=str,
    default="full",
    help=(
        "Future window visible to E_skill. 'full' keeps legacy s_{t+1:t+W}; "
        "'intermediate' hides target s_{t+W} and uses s_{t+1:t+W-1}; "
        "'suffix<N>' keeps only the last N slots of the intermediate window "
        "(s_{t+W-N:t+W-1}), the window-usage ablation. Validated by the "
        "trainer config."
    ),
)
parser.add_argument(
    "--source_history_steps",
    type=int,
    default=0,
    help=(
        "Past frames added to phi's conditioning on top of s_t. 0 (default) "
        "reproduces every existing arm. The macro state carries no velocities, "
        "so a single frame cannot express which way the motion is already "
        "going; a past chunk supplies that. phi/g/mu are pretrain-only heads, "
        "so this does not change the deployed command."
    ),
)
parser.add_argument(
    "--source_anchor",
    type=str,
    default="current",
    choices=("current", "past_start"),
    help=(
        "Heading frame of the sampled window when --source_history_steps > 0. "
        "'current' anchors on s_t, so the past reads as negative displacement "
        "and the encoder input is unchanged from a history-free run. "
        "'past_start' anchors on the OLDEST past frame, making one long macro "
        "window; the encoder's input distribution changes and no tracker can "
        "bind the result until the frozen sampler can anchor on a delayed "
        "robot heading."
    ),
)
parser.add_argument(
    "--transition_objective",
    type=str,
    default="endpoint",
    choices=(
        "endpoint",
        "state_occupancy",
        "semimarkov_chain",
        "endpoint_delta",
        "jepa_ntp",
        "reconstruction",
    ),
    help=(
        "DiffSR skill factorization. 'endpoint' predicts s[t+H] from (s[t], z); "
        "'state_occupancy' predicts a sampled checkpoint s[t+h_k] from (s[t], z); "
        "'semimarkov_chain' predicts each checkpoint from the preceding checkpoint "
        "under one held z; 'endpoint_delta' predicts s[t+H]-s[t]; 'jepa_ntp' is "
        "chunk-wise next-token prediction against an EMA target encoder with a "
        "bilinear (spectral) energy head trained by symmetric InfoNCE; "
        "'reconstruction' decodes the exact encoder input window from z."
    ),
)
parser.add_argument(
    "--reconstruction_target",
    type=str,
    default="input_window",
    choices=("input_window", "endpoint", "full_window"),
    help=(
        "What the 'reconstruction' objective decodes from z: 'input_window' "
        "reproduces the exact encoder input (purely local); 'endpoint' regresses "
        "the macro state one horizon later; 'full_window' decodes the state plus "
        "every future slot including the endpoint the encoder never sees."
    ),
)
parser.add_argument(
    "--jepa_loss",
    type=str,
    default="sigreg",
    choices=("sigreg", "sigreg_ebm", "infonce"),
    help=(
        "jepa_ntp anti-collapse mechanism: 'sigreg' (LeJEPA prediction MSE + "
        "sketched isotropic-Gaussian regularization, no negatives) or "
        "'infonce' (bilinear spectral energy over in-batch negatives)."
    ),
)
parser.add_argument("--jepa_sigreg_coeff", type=float, default=1.0)
parser.add_argument("--jepa_ntp_coeff", type=float, default=1.0)
parser.add_argument(
    "--jepa_ntp_chunk_anchor",
    type=str,
    default="executed",
    choices=("executed", "next"),
    help=(
        "Heading frame of the diff_chunk target: 'executed' keeps the next "
        "chunk in the executed chunk's slot-0 frame (drift in the target); "
        "'next' re-anchors onto s_{t+H}'s own frame (drift erased)."
    ),
)
parser.add_argument(
    "--jepa_ntp_chunk_span",
    type=str,
    default="next",
    choices=("next", "boundary_next"),
    help=(
        "Frames the diff_chunk head denoises: 'next' is the next chunk's H "
        "frames s[t+H+1..t+2H] (the round-4 recipe); 'boundary_next' widens "
        "the target to s[t+H..t+2H] (H+1 frames) so ONE head covers the "
        "endpoint and the next chunk — pair with --jepa_endpoint_coeff 0 for "
        "the merged-head cell. Requires --jepa_ntp_chunk_anchor executed."
    ),
)
parser.add_argument(
    "--jepa_token_pred_coeff",
    type=float,
    default=0.0,
    help=(
        "Weight of the mlp token-prediction MSE ||P(z1) - z2||^2 ADDED "
        "alongside a diffusion NTP head. 0 (default) reproduces every "
        "existing arm; positive requires a diffusion --jepa_ntp_head."
    ),
)
parser.add_argument(
    "--jepa_endpoint_coeff",
    type=float,
    default=1.0,
    help=(
        "Weight of the endpoint DiffSR term p(s[t+H]|s[t],z) inside "
        "sigreg_ebm. 0 drops it from the loss; the head is still built and "
        "serialized, so the checkpoint contract is unchanged."
    ),
)
parser.add_argument(
    "--jepa_ntp_head",
    type=str,
    default="mlp",
    choices=("mlp", "diff_token", "diff_chunk", "diff_pair"),
    help=(
        "Next-chunk prediction estimator for sigreg_ebm: 'mlp' is the "
        "deterministic MSE predictor; 'diff_token' models p(z_next|s_t,z_t) "
        "with a second DiffSR diffusion head; 'diff_chunk' models the next "
        "chunk's H raw frames p(x_{t+H+1:t+2H}|s_t,z_t) in the executed "
        "chunk's heading frame."
    ),
)
parser.add_argument("--jepa_sigreg_sketches", type=int, default=64)
parser.add_argument(
    "--jepa_context_chunks",
    type=int,
    default=0,
    choices=(0, 1),
    help=(
        "0 = chunk pair (predict the next chunk token from the current one); "
        "1 = chunk triplet (the predictor also reads the PRECEDING chunk's "
        "token, wiki/skill-encoder-jepa-plan.md phase 2)."
    ),
)
parser.add_argument(
    "--jepa_target_encoder_mode",
    type=str,
    default="ema",
    choices=("ema", "online", "stopgrad"),
    help=(
        "Where the next chunk's target token comes from: 'ema' (momentum copy "
        "of the encoder) or 'online' (the ONE encoder on both sides, no "
        "stop-gradient -- the LeJEPA shape; SIGReg alone prevents collapse)."
    ),
)
parser.add_argument(
    "--transition_offsets",
    type=int,
    nargs="*",
    default=None,
    help=(
        "Strictly increasing h_k checkpoints ending at --horizon_steps. Empty uses "
        "only H for endpoint objectives and every step for occupancy/chain."
    ),
)
parser.add_argument("--z_dim", type=int, default=256, help="Skill latent dimension.")
parser.add_argument(
    "--latent_mode",
    type=str,
    default="deterministic",
    choices=(
        "deterministic",
        "gaussian",
        "categorical",
        "gumbel_multicat",
        "gumbel",
        "fsq",
        "sonic_fsq",
        "vq",
    ),
    help=(
        "Latent bottleneck design: 'deterministic' (z_norm), 'gaussian' (beta-VAE), "
        "'categorical' (multi-categorical straight-through codebook), 'gumbel' "
        "(Gumbel-softmax, annealed tau), 'fsq' (finite scalar quant), 'vq' (EMA VQ-VAE)."
    ),
)
parser.add_argument(
    "--reg_coeff",
    type=float,
    default=1.0e-3,
    help="Latent regularizer weight: L2 on z for 'deterministic', KL bottleneck "
    "for 'gaussian'/'categorical'.",
)
parser.add_argument(
    "--categorical_groups",
    type=int,
    default=8,
    help="Number of categorical groups G for --latent_mode categorical (must divide "
    "z_dim; per-group code dim = z_dim // G).",
)
parser.add_argument(
    "--categorical_categories",
    type=int,
    default=32,
    help="Categories per group (vocab size) for --latent_mode categorical. Independent "
    "of z_dim; larger = more capacity per group.",
)
parser.add_argument(
    "--gumbel_codebook_size", type=int, default=512, help="Codebook size for gumbel/."
)
parser.add_argument("--gumbel_tau_start", type=float, default=2.0)
parser.add_argument("--gumbel_tau_end", type=float, default=0.5)
parser.add_argument("--gumbel_tau_anneal_iters", type=int, default=2000)
parser.add_argument(
    "--gumbel_hard", action=argparse.BooleanOptionalAction, default=True
)
parser.add_argument(
    "--fsq_levels",
    type=int,
    nargs="+",
    default=[8, 8, 8, 5, 5],
    help="Per-dim levels for --latent_mode fsq (codebook size = product).",
)
parser.add_argument(
    "--sonic_fsq_levels",
    type=int,
    nargs="+",
    default=[32] * 64,
    help=(
        "Per-dim levels for --latent_mode sonic_fsq. Default is the SONIC-matched "
        "token space (64 dims x 32 levels ~= 320 bits per command, i.e. gear_sonic "
        "tokens of shape (2, 32) at num_fsq_levels=32). This mode publishes the "
        "quantizer output directly as the command, so --z_dim must equal the "
        "number of levels given here."
    ),
)
parser.add_argument(
    "--encoder_hidden_dims",
    type=int,
    nargs="+",
    default=None,
    help="Skill-encoder trunk MLP hidden widths (default: config default).",
)
parser.add_argument(
    "--encoder_activation",
    type=str,
    choices=("elu", "mish", "relu", "silu"),
    default="mish",
    help="Skill-encoder trunk activation.",
)
parser.add_argument(
    "--encoder_layer_norm",
    dest="encoder_layer_norm",
    action="store_true",
    default=True,
    help="Apply LayerNorm after every skill-encoder hidden linear layer.",
)
parser.add_argument(
    "--no_encoder_layer_norm",
    dest="encoder_layer_norm",
    action="store_false",
    help="Disable skill-encoder hidden LayerNorm, matching SONIC's MLP recipe.",
)
parser.add_argument("--vq_codebook_size", type=int, default=512)
parser.add_argument("--vq_ema_decay", type=float, default=0.99)
parser.add_argument(
    "--vq_dead_code_reset_iters",
    type=int,
    default=0,
    help="Revive unused VQ codes every N updates (0 disables).",
)
parser.add_argument(
    "--diffsr_feature_dim",
    type=int,
    default=128,
    help="DiffSR spectral feature dimension.",
)
parser.add_argument(
    "--diffsr_embed_dim",
    type=int,
    default=512,
    help="DiffSR bilinear embedding dimension.",
)
parser.add_argument(
    "--diffsr_f_hidden_dims",
    type=int,
    nargs="+",
    default=None,
    help="DiffSR legacy bilinear state-encoder hidden widths.",
)
parser.add_argument(
    "--diffsr_g_hidden_dims",
    type=int,
    nargs="+",
    default=None,
    help=(
        "DiffSR concat state, skill, and phi encoder hidden widths. This is the "
        "active state-encoder knob under --diffsr_phi_parameterization concat."
    ),
)
parser.add_argument(
    "--diffsr_mu_hidden_dims",
    type=int,
    nargs="+",
    default=None,
    help="DiffSR next-state diffusion encoder hidden widths.",
)
parser.add_argument(
    "--diffsr_phi_parameterization",
    type=str,
    default="concat",
    choices=(
        "concat",
        "bilinear",
        "affine",
        "affine_no_bias",
        "affine_poe",
        "identity",
        "identity_bias",
        "linear",
        "linear_bias",
    ),
    help=(
        "DiffSR phi(s,z) parameterization. 'concat' is the newer simple-concat "
        "path; 'bilinear' restores the legacy matrix F(s) with g(z)^T F(s); "
        "'affine' uses that matrix form with a single linear g(z) = A z + b, "
        "which makes phi and the diffusion score field affine in z. "
        "'affine_no_bias' removes b, giving <z, A^T F(s) mu>. "
        "'affine_poe' reassociates the affine model as <[1; z], E>. "
        "Field affinity alone does not guarantee exact clean PoE sampling. "
        "'identity' sets phi(s, z) = z (requires "
        "--diffsr_feature_dim == --z_dim); with --diffsr_mu_conditioning pair "
        "the denoiser becomes <z, E(s, s', t)>, the product-of-experts form. "
        "'identity_bias' sets phi = [1; z] (requires feature_dim = z_dim + 1), "
        "adding a command-independent base denoiser. 'linear' sets phi = A z "
        "with a bias-free linear A into --diffsr_feature_dim, which may be "
        "wider than z, so the expert count is untied from the latent width; "
        "'linear_bias' is [1; A z], the same projection with the base "
        "denoiser channel."
    ),
)
parser.add_argument(
    "--diffsr_mu_conditioning",
    type=str,
    default="next",
    choices=("next", "pair", "film"),
    help=(
        "What the DiffSR denoiser mu sees. 'next': mu(s', t), the default. "
        "'pair': mu(s, s', t), so each row of mu is a joint expert field over "
        "the transition pair. 'film': source-dependent scale and shift on "
        "noisy-target hidden features before the expert readout; never uses z."
    ),
)
parser.add_argument("--batch_size", type=int, default=8192, help="Training batch size.")
parser.add_argument(
    "--encoder_lr",
    type=float,
    default=3.0e-4,
    help="Learning rate for the skill encoder and JEPA next-chunk head.",
)
parser.add_argument("--num_updates", type=int, default=2000, help="Training updates.")
parser.add_argument("--log_interval", type=int, default=100, help="Log cadence.")
parser.add_argument(
    "--dynamics_probe",
    action="store_true",
    help="After training or eval-only, run matched-noise raw-chunk diagnostics.",
)
parser.add_argument("--dynamics_probe_batch_size", type=int, default=256)
parser.add_argument("--dynamics_probe_batches", type=int, default=4)
parser.add_argument("--dynamics_probe_seed", type=int, default=1729)
parser.add_argument(
    "--eval_batches",
    type=int,
    default=4,
    help="Held-out batches per diagnostic evaluation.",
)
parser.add_argument(
    "--eval_batch_size",
    type=int,
    default=None,
    help="Held-out diagnostic batch size. Defaults to --batch_size.",
)
parser.add_argument(
    "--train_split",
    type=str,
    default="train",
    choices=("all", "train", "eval"),
    help="Expert trajectory split used for optimizer updates.",
)
parser.add_argument(
    "--eval_split",
    type=str,
    default="eval",
    choices=("all", "train", "eval"),
    help="Expert trajectory split used for diagnostics.",
)
parser.add_argument(
    "--eval_trajectory_fraction",
    type=float,
    default=0.1,
    help="Fraction of nonempty expert trajectories reserved for eval split.",
)
parser.add_argument(
    "--trajectory_split_seed",
    type=int,
    default=0,
    help="Deterministic seed for train/eval trajectory split.",
)
parser.add_argument(
    "--grad_clip_norm",
    type=float,
    default=1.0,
    help="Global gradient max norm. Use <=0 to disable.",
)
parser.add_argument(
    "--reconstruction_norm_eps",
    type=float,
    default=1.0e-6,
    help="Variance floor for normalized reconstruction diagnostics.",
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=None,
    help=(
        "AdamW weight decay for the encoder/DiffSR optimizer. Omit to keep the "
        "trainer default (0.0)."
    ),
)
parser.add_argument(
    "--reconstruction_eval",
    action="store_true",
    default=False,
    help="Include stochastic DiffSR sampled reconstruction errors in eval metrics.",
)
parser.add_argument(
    "--window_probe_eval",
    action="store_true",
    default=False,
    help="Fit eval-only linear probes from frozen z/state to held-out future windows.",
)
parser.add_argument(
    "--window_probe_train_batches",
    type=int,
    default=4,
    help="Train batches for eval-only closed-form window probes.",
)
parser.add_argument(
    "--window_probe_eval_batches",
    type=int,
    default=None,
    help="Held-out batches for eval-only window probes. Defaults to --eval_batches.",
)
parser.add_argument(
    "--window_probe_ridge",
    type=float,
    default=1.0e-3,
    help="Ridge coefficient for eval-only window probes.",
)
parser.add_argument(
    "--cotrain_commander",
    action="store_true",
    default=False,
    help="Co-train a skill commander (BC to encoder z from state + language goal).",
)
parser.add_argument(
    "--commander_language_embeddings",
    type=str,
    default=None,
    help="Language-goal embedding table (.pt); required with --cotrain_commander.",
)
parser.add_argument(
    "--commander_lr", type=float, default=3.0e-4, help="Commander BC learning rate."
)
parser.add_argument(
    "--commander_hidden_dims",
    type=int,
    nargs="+",
    default=[1024, 512, 512],
    help="Commander MLP hidden widths.",
)
parser.add_argument(
    "--commander_cosine_loss_coeff",
    type=float,
    default=1.0,
    help="Weight on the (1 - cosine) commander BC term.",
)
parser.add_argument(
    "--cotrain_language_planner",
    action="store_true",
    default=False,
    help=(
        "Co-train the language-tilted flow planner p(z|X,l) ∝ p_base(z|X) "
        "exp(z^T b(X,l)) on the detached encoder code (base flow + log-linear "
        "language expert). See algorithms/diffsr/language_tilt_planner.py."
    ),
)
parser.add_argument(
    "--planner_language_embeddings",
    type=str,
    default=None,
    help="Language-goal embedding table (.pt); required with --cotrain_language_planner.",
)
parser.add_argument(
    "--planner_lr", type=float, default=3.0e-4, help="Planner AdamW lr."
)
parser.add_argument(
    "--planner_weight_decay",
    type=float,
    default=0.0,
    help="Planner AdamW weight decay.",
)
parser.add_argument(
    "--planner_flow_hidden_dims",
    type=int,
    nargs="+",
    default=[1024, 512, 512],
    help="Base-flow velocity MLP widths.",
)
parser.add_argument(
    "--planner_expert_hidden_dims",
    type=int,
    nargs="+",
    default=[512, 512],
    help="Language-expert state-matrix MLP widths.",
)
parser.add_argument(
    "--planner_lang_proj_dim",
    type=int,
    default=64,
    help="Width of the projected language embedding inside the log-linear expert.",
)
parser.add_argument(
    "--planner_flow_noise_std",
    type=float,
    default=1.0,
    help="Std of the base flow's Gaussian source (train and inference).",
)
parser.add_argument(
    "--planner_flow_num_inference_steps",
    type=int,
    default=16,
    help="Euler steps when sampling the planner.",
)
parser.add_argument(
    "--planner_state_noise_std",
    type=float,
    default=0.0,
    help="Per-dim Gaussian noise on the planner state input during training.",
)
parser.add_argument(
    "--planner_eval_betas",
    type=float,
    nargs="+",
    default=[0.0, 1.0, 2.0],
    help="Guidance strengths swept in the held-out planner diagnostics.",
)
parser.add_argument(
    "--planner_eval_num_samples",
    type=int,
    default=8,
    help="Samples per row in the held-out planner diagnostics.",
)
parser.add_argument(
    "--jepa_reconstruction_coeff",
    type=float,
    default=0.0,
    help=(
        "Weight of an autoencoder term added to jepa_ntp: a decoder z -> "
        "--reconstruction_target (default the exact encoder input) scored on a "
        "running per-dimension z-score, summed over dims, inside the encoder's "
        "optimizer step. 0 (default) reproduces every existing arm."
    ),
)
parser.add_argument(
    "--cotrain_regression_planner",
    action="store_true",
    default=False,
    help=(
        "Co-train a deterministic planner z_hat = P(X, l) from phi's source (the "
        "causal macro history) and the caption embedding, with the Gaussian NLL "
        "0.5||z - z_hat||^2 in the encoder's own optimizer step. Joint (gradient "
        "into the encoder) unless --regression_planner_detach_target. Language "
        "table: --planner_language_embeddings."
    ),
)
parser.add_argument("--regression_planner_coeff", type=float, default=1.0)
parser.add_argument(
    "--regression_planner_detach_target",
    action="store_true",
    default=False,
    help="Detach the planner's target code (no planner gradient into the encoder).",
)
parser.add_argument(
    "--no_regression_planner_language",
    dest="regression_planner_use_language",
    action="store_false",
    default=True,
    help="Condition the regression planner on the causal history only.",
)
parser.add_argument(
    "--regression_planner_hidden_dims",
    type=int,
    nargs="+",
    default=[1024, 1024, 512],
)
parser.add_argument("--regression_planner_lr", type=float, default=3.0e-4)
parser.add_argument(
    "--heads_drop_qvel",
    action="store_true",
    default=False,
    help=(
        "With 67-wide full_body macro frames: the ENCODER reads the full frame "
        "(joint velocities included), while phi's source and the endpoint / "
        "next-chunk targets use the root_qpos slice."
    ),
)
parser.add_argument(
    "--macro_qvel_scale",
    type=float,
    default=1.0,
    help=(
        "Multiplier on the joint-velocity block of 67-wide full_body macro frames "
        "(trainer sampling and, via the checkpoint config, the tracker's live "
        "encoder input). 1.0 (default) changes nothing."
    ),
)
parser.add_argument(
    "--macro_offset_aug",
    type=str,
    default="none",
    choices=("none", "lognormal"),
    help=(
        "Move every training jepa draw (encoder window, phi source, next-chunk "
        "target) by one random planar offset, so the encoder also learns windows "
        "anchored at a live robot that is off the reference (robot_pos_heading rollout)."
    ),
)
parser.add_argument("--macro_offset_xy_median", type=float, default=0.05)
parser.add_argument("--macro_offset_xy_log_sigma", type=float, default=1.448)
parser.add_argument("--macro_offset_yaw_median_deg", type=float, default=1.1)
parser.add_argument("--macro_offset_yaw_log_sigma", type=float, default=1.10)
parser.add_argument(
    "--commander_state_noise_std",
    type=float,
    default=0.0,
    help="Per-dim Gaussian noise on the commander state input (M3 robustness).",
)
parser.add_argument(
    "--logger_backend",
    type=str,
    default="none",
    help="Metrics backend for pretraining. Use 'wandb' to log to Weights & Biases, "
    "or 'none' to keep local JSONL/stdout logging only.",
)
parser.add_argument(
    "--wandb_project",
    type=str,
    default="G1-Imitation-HL-Skill-DiffSR",
    help="W&B project name when --logger_backend=wandb.",
)
parser.add_argument(
    "--wandb_entity",
    type=str,
    default=None,
    help="Optional W&B entity (team/user) when --logger_backend=wandb.",
)
parser.add_argument(
    "--wandb_group",
    type=str,
    default=None,
    help="Optional W&B group when --logger_backend=wandb.",
)
parser.add_argument(
    "--wandb_run_name",
    type=str,
    default=None,
    help="Optional W&B run name. Defaults to the run output directory name.",
)
parser.add_argument(
    "--wandb_mode",
    type=str,
    default="online",
    choices=("online", "offline", "disabled"),
    help="W&B mode passed to wandb.init when --logger_backend=wandb.",
)
parser.add_argument(
    "--assert-kitless",
    action="store_true",
    help="Require a Newton configuration that never imports or starts Kit.",
)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym
import imitationlab.envs.isaaclab.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.app import launch_simulation
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.io import dump_yaml
from imitationlab.envs.isaaclab.envs.torchrl_adapter import IsaacLabWrapper
from imitationlab.runners.launch_compat import hydra_task_config
from imitationlab.runners.launch_compat import needs_kit as config_needs_kit
from imitationlab.encoders.diffsr.pretrain import (
    HighLevelSkillDiffSRConfig,
    HighLevelSkillDiffSRTrainer,
)


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _wandb_enabled() -> bool:
    return str(args_cli.logger_backend).strip().lower() == "wandb"


def _init_wandb(log_dir: Path, config_payload: dict[str, Any]) -> Any:
    """Initialize a W&B run for pretraining, or return None if disabled."""
    if not _wandb_enabled():
        return None
    import wandb  # local import so the dependency is only needed for wandb runs

    run = wandb.init(
        project=args_cli.wandb_project,
        entity=args_cli.wandb_entity,
        group=args_cli.wandb_group,
        name=args_cli.wandb_run_name or log_dir.name,
        dir=str(log_dir),
        mode=args_cli.wandb_mode,
        config=config_payload,
    )
    print(f"[INFO] W&B logging enabled: {run.url if run else '(no run)'}")
    return run


def _wandb_log(run: Any, row: dict[str, Any]) -> None:
    """Log the numeric fields of a metrics row to W&B, stepped by 'update'."""
    if run is None:
        return
    step = int(row["update"]) if "update" in row else None
    payload = {
        key: value
        for key, value in row.items()
        if key != "update"
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }
    if payload:
        run.log(payload, step=step)


def _build_trainer_config(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg | None = None,
) -> HighLevelSkillDiffSRConfig:
    grad_clip_norm = None if args_cli.grad_clip_norm <= 0 else args_cli.grad_clip_norm
    # The environment owns the macro-window cadence; this only records it, so
    # the checkpoint carries the stride its encoder was actually trained on and
    # a low-level run cannot pair with the wrong one. There is deliberately no
    # CLI flag: two sources for one value is how they drift apart.
    macro_frame_stride = int(getattr(env_cfg, "expert_macro_frame_stride", 1) or 1)
    # Same contract for the macro-window frame convention: recorded from the
    # environment, no CLI flag.
    macro_anchor_mode = str(
        getattr(env_cfg, "expert_macro_anchor_mode", "robot") or "robot"
    )
    config = HighLevelSkillDiffSRConfig(
        horizon_steps=args_cli.horizon_steps,
        macro_frame_stride=macro_frame_stride,
        macro_anchor_mode=macro_anchor_mode,
        macro_origin=str(getattr(env_cfg, "expert_macro_origin", "xy") or "xy"),
        macro_heading=str(getattr(env_cfg, "expert_macro_heading", "twist") or "twist"),
        rollout_anchor=str(
            getattr(env_cfg, "skill_rollout_anchor", "robot") or "robot"
        ),
        encoder_window_mode=args_cli.encoder_window_mode,
        source_history_steps=args_cli.source_history_steps,
        source_anchor=args_cli.source_anchor,
        transition_objective=args_cli.transition_objective,
        reconstruction_target=args_cli.reconstruction_target,
        jepa_loss=args_cli.jepa_loss,
        jepa_sigreg_coeff=args_cli.jepa_sigreg_coeff,
        jepa_ntp_coeff=args_cli.jepa_ntp_coeff,
        jepa_ntp_head=args_cli.jepa_ntp_head,
        jepa_ntp_chunk_anchor=args_cli.jepa_ntp_chunk_anchor,
        jepa_ntp_chunk_span=args_cli.jepa_ntp_chunk_span,
        jepa_endpoint_coeff=args_cli.jepa_endpoint_coeff,
        jepa_token_pred_coeff=args_cli.jepa_token_pred_coeff,
        jepa_sigreg_sketches=args_cli.jepa_sigreg_sketches,
        jepa_context_chunks=args_cli.jepa_context_chunks,
        jepa_target_encoder_mode=args_cli.jepa_target_encoder_mode,
        transition_offsets=tuple(args_cli.transition_offsets or ()),
        z_dim=args_cli.z_dim,
        latent_mode=args_cli.latent_mode,
        reg_coeff=args_cli.reg_coeff,
        categorical_groups=args_cli.categorical_groups,
        categorical_categories=args_cli.categorical_categories,
        gumbel_codebook_size=args_cli.gumbel_codebook_size,
        gumbel_tau_start=args_cli.gumbel_tau_start,
        gumbel_tau_end=args_cli.gumbel_tau_end,
        gumbel_tau_anneal_iters=args_cli.gumbel_tau_anneal_iters,
        gumbel_hard=args_cli.gumbel_hard,
        fsq_levels=tuple(args_cli.fsq_levels),
        sonic_fsq_levels=tuple(args_cli.sonic_fsq_levels),
        **(
            {"encoder_hidden_dims": tuple(args_cli.encoder_hidden_dims)}
            if args_cli.encoder_hidden_dims
            else {}
        ),
        encoder_activation=args_cli.encoder_activation,
        encoder_layer_norm=bool(args_cli.encoder_layer_norm),
        vq_codebook_size=args_cli.vq_codebook_size,
        vq_ema_decay=args_cli.vq_ema_decay,
        vq_dead_code_reset_iters=args_cli.vq_dead_code_reset_iters,
        diffsr_feature_dim=args_cli.diffsr_feature_dim,
        diffsr_embed_dim=args_cli.diffsr_embed_dim,
        diffsr_phi_parameterization=args_cli.diffsr_phi_parameterization,
        diffsr_mu_conditioning=args_cli.diffsr_mu_conditioning,
        **(
            {"diffsr_f_hidden_dims": tuple(args_cli.diffsr_f_hidden_dims)}
            if args_cli.diffsr_f_hidden_dims
            else {}
        ),
        **(
            {"diffsr_g_hidden_dims": tuple(args_cli.diffsr_g_hidden_dims)}
            if args_cli.diffsr_g_hidden_dims
            else {}
        ),
        **(
            {"diffsr_mu_hidden_dims": tuple(args_cli.diffsr_mu_hidden_dims)}
            if args_cli.diffsr_mu_hidden_dims
            else {}
        ),
        batch_size=args_cli.batch_size,
        encoder_lr=args_cli.encoder_lr,
        num_updates=args_cli.num_updates,
        log_interval=args_cli.log_interval,
        eval_batches=args_cli.eval_batches,
        eval_batch_size=args_cli.eval_batch_size,
        train_split=args_cli.train_split,
        eval_split=args_cli.eval_split,
        eval_trajectory_fraction=args_cli.eval_trajectory_fraction,
        trajectory_split_seed=args_cli.trajectory_split_seed,
        grad_clip_norm=grad_clip_norm,
        reconstruction_norm_eps=args_cli.reconstruction_norm_eps,
        **(
            {"weight_decay": float(args_cli.weight_decay)}
            if args_cli.weight_decay is not None
            else {}
        ),
        device=args_cli.device if args_cli.device is not None else "auto",
        cotrain_commander=bool(args_cli.cotrain_commander),
        commander_language_embeddings_path=(
            str(Path(args_cli.commander_language_embeddings).expanduser())
            if args_cli.commander_language_embeddings
            else ""
        ),
        commander_hidden_dims=tuple(int(d) for d in args_cli.commander_hidden_dims),
        commander_lr=args_cli.commander_lr,
        commander_cosine_loss_coeff=args_cli.commander_cosine_loss_coeff,
        commander_state_noise_std=args_cli.commander_state_noise_std,
        cotrain_language_planner=bool(args_cli.cotrain_language_planner),
        planner_language_embeddings_path=(
            str(Path(args_cli.planner_language_embeddings).expanduser())
            if args_cli.planner_language_embeddings
            else ""
        ),
        planner_lr=args_cli.planner_lr,
        planner_weight_decay=args_cli.planner_weight_decay,
        planner_flow_hidden_dims=tuple(
            int(d) for d in args_cli.planner_flow_hidden_dims
        ),
        planner_expert_hidden_dims=tuple(
            int(d) for d in args_cli.planner_expert_hidden_dims
        ),
        planner_lang_proj_dim=args_cli.planner_lang_proj_dim,
        planner_flow_noise_std=args_cli.planner_flow_noise_std,
        planner_flow_num_inference_steps=args_cli.planner_flow_num_inference_steps,
        planner_state_noise_std=args_cli.planner_state_noise_std,
        planner_eval_betas=tuple(float(b) for b in args_cli.planner_eval_betas),
        planner_eval_num_samples=args_cli.planner_eval_num_samples,
        jepa_reconstruction_coeff=args_cli.jepa_reconstruction_coeff,
        cotrain_regression_planner=bool(args_cli.cotrain_regression_planner),
        regression_planner_coeff=args_cli.regression_planner_coeff,
        regression_planner_detach_target=bool(
            args_cli.regression_planner_detach_target
        ),
        regression_planner_use_language=bool(args_cli.regression_planner_use_language),
        regression_planner_hidden_dims=tuple(
            int(d) for d in args_cli.regression_planner_hidden_dims
        ),
        regression_planner_lr=args_cli.regression_planner_lr,
        macro_qvel_scale=args_cli.macro_qvel_scale,
        macro_offset_aug=args_cli.macro_offset_aug,
        macro_offset_xy_median=args_cli.macro_offset_xy_median,
        macro_offset_xy_log_sigma=args_cli.macro_offset_xy_log_sigma,
        macro_offset_yaw_median_deg=args_cli.macro_offset_yaw_median_deg,
        macro_offset_yaw_log_sigma=args_cli.macro_offset_yaw_log_sigma,
        heads_drop_qvel=bool(args_cli.heads_drop_qvel),
    )
    config.validate()
    return config


def _run_dir() -> Path:
    if args_cli.output_dir is not None:
        return Path(args_cli.output_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path("outputs", "encoders", args_cli.latent_mode, timestamp).resolve()


def _run_training(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
) -> None:
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        random.seed(int(args_cli.seed))
        torch.manual_seed(int(args_cli.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args_cli.seed))
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    torch.set_float32_matmul_precision("high")
    log_dir = _run_dir()
    checkpoints_dir = log_dir / "checkpoints"
    metrics_path = log_dir / "metrics.jsonl"
    checkpoint_path = checkpoints_dir / "latest.pt"
    log_dir.mkdir(parents=True, exist_ok=True)

    trainer_config = _build_trainer_config(env_cfg)
    config_payload = {
        "task": args_cli.task,
        "num_envs": int(env_cfg.scene.num_envs),
        "seed": args_cli.seed,
        "checkpoint": args_cli.checkpoint,
        "eval_only": bool(args_cli.eval_only),
        "dynamics_probe": bool(args_cli.dynamics_probe),
        "dynamics_probe_batch_size": args_cli.dynamics_probe_batch_size,
        "dynamics_probe_batches": args_cli.dynamics_probe_batches,
        "dynamics_probe_seed": args_cli.dynamics_probe_seed,
        "reconstruction_eval": bool(args_cli.reconstruction_eval),
        "window_probe_eval": bool(args_cli.window_probe_eval),
        "window_probe_train_batches": args_cli.window_probe_train_batches,
        "window_probe_eval_batches": args_cli.window_probe_eval_batches,
        "window_probe_ridge": args_cli.window_probe_ridge,
        "trainer": trainer_config.to_dict(),
        "command": " ".join(sys.orig_argv),
        "hardware_selection": json.loads(
            os.environ.get("IMITATIONLAB_PRETRAIN_HARDWARE", "null")
        ),
    }
    (log_dir / "config.yaml").write_text(
        yaml.safe_dump(config_payload, sort_keys=True),
        encoding="utf-8",
    )
    dump_yaml(str(log_dir / "env.yaml"), env_cfg)
    print(f"[INFO] Logging high-level DiffSR run to: {log_dir}")

    wandb_run = _init_wandb(log_dir, config_payload)

    env_cfg.log_dir = str(log_dir)
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        raise NotImplementedError("DirectMARLEnv is not supported by this script.")

    wrapped_env = IsaacLabWrapper(env)
    trainer = HighLevelSkillDiffSRTrainer(config=trainer_config, env=wrapped_env)
    try:
        if args_cli.checkpoint is not None:
            trainer.load_checkpoint(args_cli.checkpoint)
            print(f"[INFO] Loaded checkpoint: {Path(args_cli.checkpoint).resolve()}")

        if args_cli.eval_only:
            if args_cli.checkpoint is None:
                raise ValueError("--eval_only requires --checkpoint.")
            # load_checkpoint restored the run's training config, so the
            # eval-only knobs come from the CLI, not from the checkpoint:
            # the eval batch (the affine head's [B, feature, embed] matrix
            # would otherwise be sized by the cluster training batch) and the
            # planner's guidance sweep, which is a sampling-time choice.
            if args_cli.eval_batch_size is not None:
                trainer.config.eval_batch_size = int(args_cli.eval_batch_size)
            trainer.config.eval_batches = int(args_cli.eval_batches)
            if trainer.planner is not None:
                trainer.config.planner_eval_betas = tuple(
                    float(b) for b in args_cli.planner_eval_betas
                )
                trainer.config.planner_eval_num_samples = int(
                    args_cli.planner_eval_num_samples
                )
            metrics = trainer.evaluate(
                prefix="train",
                include_reconstruction=bool(args_cli.reconstruction_eval),
            )
            if args_cli.window_probe_eval:
                metrics.update(
                    trainer.evaluate_window_probe(
                        train_batches=args_cli.window_probe_train_batches,
                        eval_batches=args_cli.window_probe_eval_batches,
                        prefix="train",
                        ridge=args_cli.window_probe_ridge,
                    )
                )
            if args_cli.dynamics_probe:
                metrics.update(
                    trainer.evaluate_dynamics_probe(
                        batch_size=args_cli.dynamics_probe_batch_size,
                        num_batches=args_cli.dynamics_probe_batches,
                        seed=args_cli.dynamics_probe_seed,
                    )
                )
            row: dict[str, Any] = {
                "update": int(trainer.update),
                "eval_only": True,
                **metrics,
            }
            _write_jsonl(metrics_path, row)
            _wandb_log(wandb_run, row)
            print(json.dumps(row, indent=2, sort_keys=True))
            return

        def _log(row: dict[str, float | int]) -> None:
            _write_jsonl(metrics_path, dict(row))
            _wandb_log(wandb_run, dict(row))
            print(json.dumps(row, sort_keys=True))

        trainer.train(
            log_callback=_log,
            checkpoint_path=checkpoint_path,
            reconstruction_eval=bool(args_cli.reconstruction_eval),
            # Saved once below, after the final evaluation.
            save_final=False,
        )
        final_metrics = trainer.evaluate(
            prefix="train",
            include_reconstruction=bool(args_cli.reconstruction_eval),
        )
        if args_cli.window_probe_eval:
            final_metrics.update(
                trainer.evaluate_window_probe(
                    train_batches=args_cli.window_probe_train_batches,
                    eval_batches=args_cli.window_probe_eval_batches,
                    prefix="train",
                    ridge=args_cli.window_probe_ridge,
                )
            )
        if args_cli.dynamics_probe:
            final_metrics.update(
                trainer.evaluate_dynamics_probe(
                    batch_size=args_cli.dynamics_probe_batch_size,
                    num_batches=args_cli.dynamics_probe_batches,
                    seed=args_cli.dynamics_probe_seed,
                )
            )
        final_row: dict[str, Any] = {
            "update": int(trainer.update),
            "post_train_eval": True,
            **final_metrics,
        }
        _write_jsonl(metrics_path, final_row)
        _wandb_log(wandb_run, final_row)
        trainer.save_checkpoint(checkpoint_path)
        print(f"[INFO] Saved checkpoint: {checkpoint_path}")
        encoder_path = checkpoints_dir / "encoder.pt"
        with TemporaryDirectory(dir=checkpoints_dir) as staging_dir:
            staged = Path(staging_dir) / encoder_path.name
            torch.save(trainer.encoder_checkpoint_state_dict(), staged)
            staged.replace(encoder_path)
        print(f"[INFO] Saved encoder: {encoder_path}")
        if trainer.commander is not None:
            commander_path = checkpoints_dir / "commander.pt"
            trainer.save_commander_checkpoint(
                commander_path, skill_checkpoint_path=str(checkpoint_path)
            )
            print(f"[INFO] Saved commander checkpoint: {commander_path}")
        if trainer.planner is not None:
            planner_path = checkpoints_dir / "planner.pt"
            trainer.save_planner_checkpoint(
                planner_path, skill_checkpoint_path=str(checkpoint_path)
            )
            print(f"[INFO] Saved language planner checkpoint: {planner_path}")
        if trainer.regression_planner is not None:
            regression_path = checkpoints_dir / "regression_planner.pt"
            trainer.save_regression_planner_checkpoint(
                regression_path, skill_checkpoint_path=str(checkpoint_path)
            )
            print(f"[INFO] Saved regression planner checkpoint: {regression_path}")
        print(json.dumps(final_row, indent=2, sort_keys=True))
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        wrapped_env.close()


@hydra_task_config(args_cli.task, None)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    _agent_cfg: object,
) -> None:
    needs_kit = config_needs_kit(env_cfg, args_cli)
    if args_cli.assert_kitless:
        if needs_kit or not config_contains_type_name(env_cfg, "NewtonCfg"):
            raise RuntimeError(
                "--assert-kitless requires a resolved NewtonCfg with no Kit cameras or Kit visualizer."
            )
        assert_kit_not_loaded()
        print("[INFO] Strict kit-less Newton pretraining runtime validated.")
    if os.environ.get("ISAACLAB_SPLIT_RUNTIME") == "1" and needs_kit:
        raise RuntimeError(
            "The split runtime cannot start Kit from the offline skill-pretraining entrypoint. "
            "Use Newton with --assert-kitless on compute-only GPUs."
        )
    # Resolve the encoder view from the task before sampling reference windows.
    interface = getattr(env_cfg, "command_interface", None)
    if interface is not None:
        interface.resolve()

    with launch_simulation(env_cfg, args_cli):
        _run_training(env_cfg)


if __name__ == "__main__":
    main()
