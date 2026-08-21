# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Path-tracing denoising experiment: PidModelPT teacher training from scratch.

Start command (single node, 4 GPUs):
    PYTHONPATH=. torchrun --nproc_per_node=4 --master_port=12341 -m scripts.train \
          --config=pid/_src/configs/pid_training/config.py \
          -- experiment="pid_pt_teacher_512crop_4bs"

Debug (1 GPU, short run):
    PYTHONPATH=. torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train \
          --config=pid/_src/configs/pid_training/config.py \
          -- experiment="pid_pt_teacher_512crop_4bs_debug"
"""

from hydra.core.config_store import ConfigStore

from pid._ext.imaginaire.lazy_config import LazyDict

# =============================================================================
# Net config: PidNet tuned for path-tracing denoising (sr_scale=1, no VAE)
#
#   lq_in_channels=3         → noisy render [B,3,H,W] through image branch
#   lq_latent_channels=13    → G-buffers [B,13,H,W] through latent branch
#   sr_scale=1               → no upsampling; output == input resolution
#   latent_spatial_down_factor=1 → no extra spatial downsampling in latent branch
#   pit_lq_inject=True       → also inject into the PiT pixel pathway
#
# With patch_size=16, sr_scale=1, latent_spatial_down_factor=1:
#   image branch:  PixelUnshuffle(16) → [B, 768, H/16, W/16]
#   latent branch: fold(16)           → [B, 3328, H/16, W/16]
# =============================================================================

PIXELDIT_CKPT = "checkpoints/PixelDiT_finetune_2kto4k/model_ema_bf16.pth"

_PT_NET_OVERRIDES = dict(
    lq_inject_mode="controlnet",
    lq_in_channels=3,
    lq_latent_channels=13,
    lq_hidden_dim=512,
    lq_gate_type="fixed",
    lq_interval=2,
    lq_num_res_blocks=2,
    lq_aux_rgb_head=False,
    lq_conv_padding_mode="replicate",
    pit_lq_inject=True,
    sr_scale=1,
    latent_spatial_down_factor=1,
    zero_init_lq=True,
    train_lq_proj_only=True,
    rope_mode="ntk_aware",
    rope_ref_h=2048,
    rope_ref_w=2048,
)

# =============================================================================
# Model config overrides (PidModelPTConfig fields)
# =============================================================================

_PT_MODEL_CONFIG = dict(
    precision="bfloat16",
    input_data_key="image",
    input_caption_key="caption",
    noisy_input_key="noisy_image",
    buffers_key="buffers",
    buffers_channels=13,
    max_spp=16.0,
    # Flow matching
    shift=1.0,
    logit_mean=0.0,
    logit_std=1.0,
    prediction_type="velocity",
    # Inference defaults
    cfg_scale=1.0,
    num_sample_steps=20,
    # EMA
    ema=dict(enabled=True, rate=0.9999),
    # No VAE-related settings
    state_ch=3,
    # No degradation pipeline, no latent noising
    train_degradation_config=None,
    latent_noising=dict(enabled=False),
    # caption_channels must match txt_embed_dim in PidNet
    caption_channels=2304,
    # Mild dynamic shift for 512-crop training
    dynamic_shift=dict(
        base_shift=1.0,
        base_image_size_for_shift_calc=512,
    ),
    # RGB alignment head disabled (not available with no-VAE path)
    lq_latent_image_align_config=None,
    # Net-level overrides (forwarded to PidNet constructor)
    net=_PT_NET_OVERRIDES,
)

# =============================================================================
# Full experiment config
# =============================================================================

PID_PT_TEACHER_512CROP_4BS: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /data_train": "pt_zarr_gfxr_cp_4bs_512crop"},
            {"override /data_val": "pt_zarr_gfxr_cp_2bs_512crop"},
            {"override /model": "ddp_pid_pt_teacher"},
            {"override /net": "pid_sr4x_v1pt5"},
            {"override /conditioner": "pid_pt_noisy_buffers"},
            {"override /optimizer": "adamw"},
            {"override /callbacks": ["basic", "wandb"]},
            {"override /ckpt_type": "dcp"},
            {"override /checkpoint": "local"},
            {"override /tokenizer": None},
            "_self_",
        ],
        job=dict(
            group="pid_pt_training",
            name="pid_pt_teacher_512crop_4bs",
        ),
        optimizer=dict(
            lr=1e-4,
            weight_decay=0.01,
        ),
        scheduler=dict(
            f_max=[1.0],
            f_min=[1e-2],
            f_start=[1e-6],
            warm_up_steps=[1000],
            cycle_lengths=[500_000],
        ),
        model=dict(
            config=_PT_MODEL_CONFIG,
        ),
        checkpoint=dict(
            save_iter=2500,
            replicate_ema_to_reg_in_training=True,
            load_training_state=False,
            strict_resume=False,
            load_path=PIXELDIT_CKPT,
        ),
        trainer=dict(
            max_iter=200_000,
            logging_iter=25,
            run_validation=True,
            validation_iter=2500,
            max_val_iter=50,
            callbacks=dict(
                grad_clip=dict(clip_norm=1.0),
                # Visualize a training-set crop every 1000 steps
                every_n_sample_train_infer_20step_reg=dict(
                    _target_="pid._src.callbacks.every_n_draw_sample.EveryNDrawSample",
                    every_n=1000,
                    is_ema=False,
                    guidance=[1.0],
                    num_sampling_step=20,
                    resize_wandb_image=False,
                    name="train_infer_20step",
                ),
                every_n_sample_train_infer_20step_ema=dict(
                    _target_="pid._src.callbacks.every_n_draw_sample.EveryNDrawSample",
                    every_n=1000,
                    is_ema=True,
                    guidance=[1.0],
                    num_sampling_step=20,
                    resize_wandb_image=False,
                    name="train_infer_20step",
                ),
            ),
        ),
    ),
    flags={"allow_objects": True},
)


def _build_debug_run(job: LazyDict) -> dict:
    return dict(
        defaults=[
            f"/experiment/{job['job']['name']}",
            "_self_",
        ],
        job=dict(
            group=job["job"]["group"] + "_debug",
            name=f"{job['job']['name']}_debug",
            wandb_mode="disabled",
        ),
        checkpoint=dict(
            save_iter=2500,
            replicate_ema_to_reg_in_training=False,
            load_training_state=False,
            strict_resume=False,
            load_path=PIXELDIT_CKPT,
        ),
        trainer=dict(
            max_iter=20,
            logging_iter=2,
            validation_iter=10,
            max_val_iter=2,
            callbacks=dict(
                every_n_sample_train_infer_20step_reg=dict(every_n=10),
                every_n_sample_train_infer_20step_ema=dict(every_n=10),
            ),
        ),
    )


cs = ConfigStore.instance()

for _item in [PID_PT_TEACHER_512CROP_4BS]:
    cs.store(
        group="experiment",
        package="_global_",
        name=_item["job"]["name"],
        node=_item,
    )
    _debug = _build_debug_run(_item)
    cs.store(
        group="experiment",
        package="_global_",
        name=_debug["job"]["name"],
        node=_debug,
    )
