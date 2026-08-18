# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from pid._src.models.pid_model import PidModel
from pid._src.models.pid_model_PT import PidModelPT


class _FakePid:
    config = SimpleNamespace(input_caption_key="caption")
    tensor_kwargs = {"device": "cpu", "dtype": torch.float32}

    def __init__(self):
        self.encoded = None

    def encode_lq_latent(self, image):
        self.encoded = image
        return torch.full((image.shape[0], 4, 2, 2), 7.0)

    def get_data_and_condition(self, batch, **kwargs):
        del kwargs
        assert isinstance(batch["LQ_latent"], torch.Tensor)
        reference = batch["image"].unsqueeze(2)
        return reference, batch["image"], None

    def _validate_inference_data_batch(self, batch):
        assert set(batch) == {"caption", "LQ_latent", "degrade_sigma"}


class _FakePT:
    config = SimpleNamespace(
        input_data_key="image",
        noisy_input_key="noisy_image",
        buffers_key="buffers",
        max_spp=16.0,
    )
    _spp_to_sigma = PidModelPT._spp_to_sigma
    _resolve_degrade_sigma = PidModelPT._resolve_degrade_sigma

    def encode_lq_latent(self, image):
        del image
        raise AssertionError("PT sampling must never invoke inherited PiD VAE logic")

    def get_data_and_condition(self, batch, **kwargs):
        del kwargs
        reference = batch["image"].unsqueeze(2)
        return reference, batch["image"], None


@pytest.mark.L0
def test_pid_live_batch_refreshes_latent_without_mutating_caller():
    model = _FakePid()
    original_latent = torch.full((2, 4, 2, 2), -3.0)
    batch = {
        "caption": ["a", "b"],
        "image": torch.zeros(2, 3, 8, 8),
        "LQ_video_or_image": torch.ones(2, 3, 4, 4),
        "LQ_latent": original_latent,
        "degrade_sigma": torch.ones(2),
    }

    result = PidModel.prepare_sampling_inputs(model, batch, from_model_step=True)

    assert model.encoded is batch["LQ_video_or_image"]
    assert result.inference_batch["LQ_latent"].eq(7).all()
    assert result.inference_batch["degrade_sigma"].eq(0).all()
    assert batch["LQ_latent"] is original_latent
    assert batch["degrade_sigma"].eq(1).all()
    assert result.conditioning_images[0].shape == (2, 3, 1, 8, 8)
    assert result.reference_image.shape == (2, 3, 1, 8, 8)


@pytest.mark.L0
def test_pid_fixed_batch_preserves_authored_latent_and_sigma():
    model = _FakePid()
    latent = torch.randn(1, 4, 2, 2)
    sigma = torch.tensor([0.35])
    batch = {
        "caption": ["fixed"],
        "image": torch.zeros(1, 3, 8, 8),
        "LQ_video_or_image": torch.ones(1, 3, 4, 4),
        "LQ_latent": latent,
        "degrade_sigma": sigma,
    }

    result = PidModel.prepare_sampling_inputs(model, batch, from_model_step=False)

    assert model.encoded is None
    assert result.inference_batch["LQ_latent"] is latent
    assert result.inference_batch["degrade_sigma"] is sigma


@pytest.mark.L0
def test_pt_sampling_uses_native_schema_and_derives_sigma_from_spp():
    model = _FakePT()
    batch = {
        "image": torch.zeros(2, 3, 8, 8),
        "noisy_image": torch.ones(2, 3, 8, 8),
        "buffers": torch.randn(2, 13, 8, 8),
        "spp": torch.tensor([1.0, 4.0]),
    }

    result = PidModelPT.prepare_sampling_inputs(model, batch, from_model_step=True)

    assert set(result.inference_batch) == {"noisy_image", "buffers", "degrade_sigma"}
    assert torch.allclose(result.inference_batch["degrade_sigma"], torch.tensor([0.75, 0.5]))
    assert result.conditioning_images[0].shape == (2, 3, 1, 8, 8)
    assert result.reference_image.shape == (2, 3, 1, 8, 8)
    assert "degrade_sigma" not in batch


@pytest.mark.L0
def test_pt_explicit_scalar_sigma_is_literal_and_target_is_optional():
    model = _FakePT()
    batch = {
        "noisy_image": torch.ones(2, 3, 8, 8),
        "buffers": torch.randn(2, 13, 8, 8),
        "degrade_sigma": 0.25,
    }

    result = PidModelPT.prepare_sampling_inputs(model, batch, from_model_step=False)

    assert torch.allclose(result.inference_batch["degrade_sigma"], torch.tensor([0.25, 0.25]))
    assert result.reference_image is None


@pytest.mark.L0
def test_pt_validation_returns_unscaled_objective():
    class _ValidationModel:
        def training_step(self, data_batch, iteration):
            del data_batch, iteration
            objective = torch.tensor(2.0)
            return {"loss_dict": {"total_loss": objective}}, objective * 4

    output, loss = PidModelPT.validation_step(_ValidationModel(), {}, 3)

    assert loss is output["loss_dict"]["total_loss"]
    assert loss.item() == 2.0
