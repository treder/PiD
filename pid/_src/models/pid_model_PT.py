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
PiD Path-Tracing (PT) denoising model.

Conditions the diffusion model on a noisy path-traced render at variable SPP and
G-buffer channels (specular albedo, diffuse albedo, normals, depth, roughness,
motion vectors). No text conditioning; no VAE.

The noisy render is routed through PidNet's image branch (lq_video_or_image) and
the G-buffers through the latent branch (lq_latent). Both branches use sr_scale=1
and latent_spatial_down_factor=1 since there is no super-resolution.

The per-sample degrade_sigma is derived from the SPP of the noisy render:
    sigma = 1 - sqrt(spp / max_spp)
matching the 1/sqrt(SPP) Monte Carlo noise convergence rate.
"""

from __future__ import annotations

import logging
import math
from contextlib import nullcontext
from typing import Any, Optional

import attrs
import numpy as np
import torch
from torch import Tensor

from pid._ext.imaginaire.lazy_config import instantiate as lazy_instantiate
from pid._ext.imaginaire.model import ImaginaireModel
from pid._ext.imaginaire.utils import misc
from pid._ext.imaginaire.utils.ema import FastEmaModelUpdater
from pid._src.models.pid_model import PidModel, PidModelConfig
from pid._src.modules.conditioner import PTCondition
from pid._src.networks.flow_matching import FlowMatchingTrainer

logger = logging.getLogger(__name__)


# =============================================================================
# Config
# =============================================================================


@attrs.define(slots=False)
class PidModelPTConfig(PidModelConfig):
    # Data batch key for the noisy PT render
    noisy_input_key: str = "noisy_image"
    # Data batch key for the concatenated G-buffer tensor [B, 13, H, W]
    buffer_key: str = "buffer"
    # Number of G-buffer channels (specular_albedo=3, diffuse_albedo=3, normal=3,
    # depth=1, roughness=1, mv=2 → 13 total)
    buffer_channels: int = 13
    # Maximum SPP value — maps to degrade_sigma=0 (gates fully open).
    # sigma = 1 - sqrt(spp / max_spp), matching 1/sqrt(SPP) MC noise convergence.
    max_spp: float = 16.0


# =============================================================================
# Model
# =============================================================================


class PidModelPT(PidModel):
    """PixelDiT path-tracing denoising model.

    Extends PidModel with:
    - No text encoder (null embeddings pre-cached as zeros; Gemma-2 is NOT loaded)
    - No VAE (G-buffers and noisy render are native pixel-space tensors)
    - Conditioning: noisy PT image → lq_video_or_image, G-buffers → lq_latent
    - degrade_sigma derived from SPP: sigma = 1 - sqrt(spp / max_spp)

    Inherits optimizer, EMA, checkpoint, gradient clipping, and LoRA infrastructure
    from PidModel unchanged.
    """

    def __init__(self, config: PidModelPTConfig):
        # Bypass PidModel.__init__ and PixelDiTModel.__init__ entirely to avoid
        # loading the Gemma-2 text encoder (~2.6B params) and the VAE.
        # Reconstruct only the pieces we need, in the same order as PixelDiTModel.
        ImaginaireModel.__init__(self)
        self.config = config

        # 1. Precision setup
        _dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        requested_dtype = _dtype_map[config.precision]
        if requested_dtype != torch.float32:
            self.autocast_dtype = requested_dtype
            self.precision = torch.float32
        else:
            self.autocast_dtype = None
            self.precision = torch.float32
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}

        # 2. Build network
        with misc.timer("PidModelPT: build_net"):
            self.net = lazy_instantiate(config.net)
            self.net = self.net.to(device="cuda", dtype=torch.float32)
            self.net.requires_grad_(True)
            if hasattr(self.net, "init_weights"):
                self.net.init_weights()
            if getattr(self.net, "patch_blocks", None):
                last_patch_block = self.net.patch_blocks[-1]
                if hasattr(last_patch_block, "freeze_unused_text_output_branch"):
                    last_patch_block.freeze_unused_text_output_branch()
            logger.info(f"PidModelPT net params: {sum(p.numel() for p in self.net.parameters()):,}")

        # 3. Null text embeddings — zero tensor [1, 1, caption_channels].
        # Stored as a non-persistent buffer so it moves with the model to the right
        # device but is not saved in checkpoints (same avoidance as text_encoder).
        # caption_channels matches the network's txt_embed_dim (default 2304 for Gemma-2).
        object.__setattr__(self, "tokenizer", None)
        object.__setattr__(self, "text_encoder", None)
        self._chi_prompt_str = ""
        self._num_chi_tokens = 0
        null_emb = torch.zeros(1, 1, config.caption_channels, dtype=torch.float32)
        self.register_buffer("_null_caption_embs", null_emb, persistent=False)

        # 4. Flow matching trainer
        self.fm_trainer = FlowMatchingTrainer(
            timescale=config.fm_timescale,
            sigma_min=0.0,
            t_sampler_args={"t_mean": config.logit_mean, "t_std": config.logit_std},
            t_sampler_type="logit_normal",
            prediction_type=config.prediction_type,
        )

        # 5. No REPA loss
        self.repa_loss = None

        # 6. Conditioner (PTConditioner expected)
        self.conditioner = lazy_instantiate(config.conditioner)
        logger.info(f"PidModelPT conditioner: {self.conditioner}")

        # 7. Dynamic shift logging
        if config.dynamic_shift is not None:
            _ds = config.dynamic_shift
            logger.info(
                f"PidModelPT dynamic shift: base_shift={_ds['base_shift']} "
                f"base_image_size={_ds['base_image_size_for_shift_calc']}"
            )

        # 8. EMA
        if config.ema.enabled:
            self.net_ema = lazy_instantiate(config.net)
            self.net_ema = self.net_ema.to(device="cuda", dtype=torch.float32)
            self.net_ema.requires_grad_(False)
            self.net_ema_worker = FastEmaModelUpdater()
            s = config.ema.rate
            self.ema_exp_coefficient = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()
            self.net_ema_worker.copy_to(src_model=self.net, tgt_model=self.net_ema)

        # 9. Stubs expected by PidModel methods we inherit
        self.vae_encoder = None
        self.latent_noiser = None
        self.downscale = 1.0  # unused but referenced by PidModel.forward

        # 10. Fidelity LoRA (optional — same as PidModel)
        if config.lora_config.enabled:
            self._inject_fidelity_lora()

    # =========================================================================
    # Text encoding — returns pre-cached null embeddings, no Gemma-2 call
    # =========================================================================

    @torch.no_grad()
    def _encode_text_raw(self, captions) -> tuple[Tensor, Tensor]:
        B = len(captions) if isinstance(captions, (list, tuple)) else 1
        embs = self._null_caption_embs.expand(B, -1, -1)
        masks = torch.ones(B, 1, device=embs.device, dtype=torch.long)
        return embs, masks

    # =========================================================================
    # SPP → degrade_sigma
    # =========================================================================

    def _spp_to_sigma(self, spp) -> Tensor:
        """Convert SPP value(s) to degrade_sigma.

        sigma = 1 - sqrt(spp / max_spp), reflecting 1/sqrt(SPP) MC noise convergence.
        max_spp → 0.0 (gates fully open); 1spp → 0.75; 0.5spp → 0.82.
        Always in [0, 1].
        """
        spp_t = torch.as_tensor(spp, dtype=torch.float32, device="cuda").reshape(-1)
        sigma = 1.0 - torch.sqrt((spp_t / self.config.max_spp).clamp(min=0.0))
        return sigma.clamp(0.0, 1.0)

    # =========================================================================
    # Data preparation
    # =========================================================================

    def prepare_data_batch_for_training(self, data_batch: dict, training_iteration=None) -> dict:
        """Compute degrade_sigma from SPP; no VAE encoding or degradation pipeline."""
        if "degrade_sigma" not in data_batch:
            spp = data_batch.get("spp")
            if spp is not None:
                data_batch["degrade_sigma"] = self._spp_to_sigma(spp)
            else:
                B = data_batch[self.config.input_data_key].shape[0]
                data_batch["degrade_sigma"] = torch.zeros(B, device="cuda", dtype=torch.float32)
        return data_batch

    # =========================================================================
    # Training
    # =========================================================================

    def training_step(self, data_batch: dict, iteration: int) -> tuple[dict, Tensor]:
        self._maybe_enable_cp_on_nets([self.net])
        cp_group = self.get_context_parallel_group()

        if cp_group is not None and cp_group.size() > 1:
            for key in (
                self.config.input_data_key,
                self.config.noisy_input_key,
                self.config.buffer_key,
            ):
                if isinstance(data_batch.get(key), torch.Tensor):
                    data_batch[key] = self._broadcast_tensor_for_cp(data_batch[key])
            data_batch.pop("degrade_sigma", None)

        # Per-step flow shift
        _shift = self.config.shift
        if self.config.dynamic_shift is not None:
            _raw = data_batch[self.config.input_data_key]
            _h, _w = _raw.shape[-2], _raw.shape[-1]
            _ds = self.config.dynamic_shift
            _shift = _ds["base_shift"] * math.sqrt(
                math.sqrt(_h * _w) / _ds["base_image_size_for_shift_calc"]
            )

        # Clean target
        x0 = data_batch[self.config.input_data_key]
        x0 = self._normalize_image(x0).to(**self.tensor_kwargs)
        if x0.ndim == 5:
            x0 = x0[:, :, 0, :, :]

        # Compute degrade_sigma from SPP
        data_batch = self.prepare_data_batch_for_training(data_batch, training_iteration=iteration)
        if cp_group is not None and cp_group.size() > 1:
            data_batch["degrade_sigma"] = self._broadcast_tensor_for_cp(data_batch["degrade_sigma"])

        # Null text embeddings: [B, 1, caption_channels]
        B = x0.shape[0]
        caption_embs = self._null_caption_embs.expand(B, -1, -1).to(**self.tensor_kwargs)

        # Conditioning tensors (with optional dropout from conditioner)
        condition: PTCondition = self.conditioner(data_batch)
        noisy_image = condition.noisy_image
        buffers = condition.buffers
        if noisy_image is not None:
            noisy_image = noisy_image.to(**self.tensor_kwargs)
        if buffers is not None:
            buffers = buffers.to(**self.tensor_kwargs)

        degrade_sigma = data_batch["degrade_sigma"]

        # Sample timestep and noise
        t = self.fm_trainer.sample_t(B, device=x0.device)
        if _shift != 1.0:
            t = (_shift * t) / (1.0 + (_shift - 1.0) * t)
        noise = torch.randn_like(x0)
        if cp_group is not None and cp_group.size() > 1:
            t = self._broadcast_tensor_for_cp(t)
            noise = self._broadcast_tensor_for_cp(noise)

        # Flow matching loss
        autocast_ctx = (
            torch.autocast("cuda", dtype=self.autocast_dtype) if self.autocast_dtype else nullcontext()
        )

        def _net_fn(x_t, t, **kwargs):
            out = self.net(
                x_t,
                t,
                caption_embs,
                lq_video_or_image=noisy_image,
                lq_latent=buffers,
                degrade_sigma=degrade_sigma,
            )
            if self.config.prediction_type == "x0":
                return out
            elif self.config.prediction_type == "velocity":
                return -out
            else:
                raise ValueError(f"Invalid prediction_type: {self.config.prediction_type}")

        with autocast_ctx:
            diff_loss, _ = self.fm_trainer.loss(fn=_net_fn, x=x0, t=t, noise=noise)

        total_loss = self.config.loss_weights.get("diffusion", 1.0) * diff_loss
        loss_dict = {"diffusion_loss": diff_loss, "total_loss": total_loss}
        output_batch = {"fm_loss": total_loss.detach(), "loss_dict": loss_dict}
        backward_loss = total_loss * self._cp_loss_scale
        return output_batch, backward_loss

    def validation_step(self, data_batch: dict, iteration: int) -> tuple[dict, Tensor]:
        return self.training_step(data_batch, iteration)

    # =========================================================================
    # Inference
    # =========================================================================

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: dict,
        cfg_scale: float = None,
        num_steps: int = None,
        seed: int = 0,
        image_size=None,
        shift: float = None,
        **kwargs,
    ) -> Tensor:
        """Generate denoised images from a noisy PT render + G-buffers.

        data_batch must contain:
          - noisy_input_key: noisy PT render [B, 3, H, W]
          - buffer_key: G-buffers [B, 13, H, W]
          - "spp" (optional): SPP value for degrade_sigma; defaults to max_spp (sigma=0)
          - "degrade_sigma" (optional): overrides spp-derived sigma
        """
        from pid._src.modules.dpmsolver import DPMS

        if cfg_scale is None:
            cfg_scale = self.config.cfg_scale
        num_steps = num_steps if num_steps is not None else self.config.num_sample_steps

        self._maybe_enable_cp_on_nets([self.net])
        cp_group = self.get_context_parallel_group()

        noisy_image = data_batch[self.config.noisy_input_key].to(**self.tensor_kwargs)
        buffers = data_batch[self.config.buffer_key].to(**self.tensor_kwargs)

        B = noisy_image.shape[0]
        img_h, img_w = noisy_image.shape[-2], noisy_image.shape[-1]
        if image_size is not None:
            if isinstance(image_size, (list, tuple)):
                img_h, img_w = int(image_size[0]), int(image_size[1])
            else:
                img_h = img_w = int(image_size)

        # Resolve shift
        if shift is not None:
            _shift = shift
        elif self.config.dynamic_shift is not None:
            _ds = self.config.dynamic_shift
            _shift = _ds["base_shift"] * math.sqrt(
                math.sqrt(img_h * img_w) / _ds["base_image_size_for_shift_calc"]
            )
        else:
            _shift = self.config.shift

        # Resolve degrade_sigma
        if "degrade_sigma" in data_batch:
            sigma_val = data_batch["degrade_sigma"]
            if isinstance(sigma_val, torch.Tensor):
                degrade_sigma = sigma_val.to(device="cuda", dtype=torch.float32).reshape(-1)
                if degrade_sigma.numel() == 1:
                    degrade_sigma = degrade_sigma.expand(B).contiguous()
            else:
                degrade_sigma = self._spp_to_sigma(sigma_val).expand(B).contiguous()
        elif "spp" in data_batch:
            degrade_sigma = self._spp_to_sigma(data_batch["spp"])
            if degrade_sigma.numel() == 1:
                degrade_sigma = degrade_sigma.expand(B).contiguous()
        else:
            degrade_sigma = torch.zeros(B, device="cuda", dtype=torch.float32)

        if cp_group is not None and cp_group.size() > 1:
            noisy_image = self._broadcast_tensor_for_cp(noisy_image)
            buffers = self._broadcast_tensor_for_cp(buffers)
            degrade_sigma = self._broadcast_tensor_for_cp(degrade_sigma)

        net = self.net
        net.eval()

        # Null text for all paths (no CFG over text; only over conditioning if needed)
        caption_embs = self._null_caption_embs.expand(B, -1, -1).unsqueeze(1)  # [B, 1, 1, C]
        null_y = caption_embs  # same — PT model has no meaningful unconditioned path

        model_dtype = next(net.parameters()).dtype

        gen = torch.Generator(device="cuda").manual_seed(int(seed))
        z = torch.randn(B, 3, img_h, img_w, device="cuda", generator=gen)

        autocast_ctx = (
            torch.autocast("cuda", dtype=self.autocast_dtype) if self.autocast_dtype else nullcontext()
        )

        with autocast_ctx:
            def _forward_fn(x, timestep, y, mask=None, **kw):
                if y.dim() == 4:
                    y = y.squeeze(1)
                return net(
                    x.to(model_dtype),
                    timestep.to(model_dtype),
                    y.to(model_dtype),
                    lq_video_or_image=noisy_image,
                    lq_latent=buffers,
                    degrade_sigma=degrade_sigma,
                )

            if self.config.prediction_type == "x0":
                dpms_model_type = "x_start"
            else:
                dpms_model_type = "flow"

            dpm_solver = DPMS(
                _forward_fn,
                condition=caption_embs,
                uncondition=null_y,
                cfg_scale=1.0,  # no CFG for PT (both cond and uncond are null text)
                model_type=dpms_model_type,
                guidance_type="classifier-free",
                model_kwargs={},
                schedule="FLOW",
                interval_guidance=[0, 1],
            )
            samples = dpm_solver.sample(
                z,
                steps=num_steps,
                order=min(num_steps, 2),
                skip_type="time_uniform_flow",
                method="multistep",
                flow_shift=_shift,
            )

        return samples.clamp(-1, 1).unsqueeze(2)

    # =========================================================================
    # Callback interface
    # =========================================================================

    def get_data_and_condition(self, data_batch: dict, **kwargs):
        """Extract GT image and PT conditions for visualization callbacks."""
        x0 = data_batch[self.config.input_data_key]
        x0 = self._normalize_image(x0).to(**self.tensor_kwargs)
        if x0.ndim == 5:
            x0 = x0[:, :, 0, :, :]
        raw_data = x0.unsqueeze(2)
        condition = self.conditioner(
            data_batch, override_dropout_rate={n: 0.0 for n in self.conditioner.embedders}
        )
        return raw_data, x0, condition
