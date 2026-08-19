# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared model-to-callback contract for periodic sample generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class SamplingInputs:
    """Inputs and images prepared by a model for a sampling callback.

    ``inference_batch`` is passed directly to ``generate_samples_from_batch``.
    Conditioning images are rendered before generated samples, and the optional
    reference image is rendered after them. All visualization tensors use image
    layout ``[B, C, 1, H, W]``.
    """

    inference_batch: dict[str, Any]
    conditioning_images: tuple[Tensor, ...] = ()
    reference_image: Tensor | None = None


def as_image_visualization(
    image: Tensor,
    *,
    reference_image: Tensor | None = None,
    name: str = "conditioning image",
) -> Tensor:
    """Convert an image tensor to callback layout and match the reference size."""
    if image.ndim == 5:
        if image.shape[2] != 1:
            raise ValueError(f"Expected single-frame {name}, got {tuple(image.shape)}")
        image = image[:, :, 0]
    if image.ndim != 4:
        raise ValueError(f"Expected {name} tensor [B, C, H, W], got {tuple(image.shape)}")

    if reference_image is not None:
        target_h, target_w = reference_image.shape[-2:]
        if image.shape[-2:] != (target_h, target_w):
            image = F.interpolate(image.float(), size=(target_h, target_w), mode="bicubic", align_corners=False)

    return image.unsqueeze(2)
