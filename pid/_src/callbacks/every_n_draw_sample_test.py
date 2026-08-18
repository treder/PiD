# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

import pid._src.callbacks.every_n_draw_sample as draw_sample_module
from pid._src.callbacks.every_n_draw_sample import EveryNDrawSample
from pid._src.models.sampling import SamplingInputs


@pytest.mark.L0
def test_callback_delegates_schema_and_preserves_visualization_order(monkeypatch):
    conditioning = torch.full((1, 3, 1, 4, 4), -1.0)
    generated = torch.zeros(1, 3, 1, 4, 4)
    reference = torch.ones(1, 3, 1, 4, 4)
    inference_batch = {"noisy_image": torch.ones(1, 3, 4, 4), "buffers": torch.ones(1, 13, 4, 4)}

    class _Model:
        config = SimpleNamespace(dynamic_shift=None, shift=1.0)

        # This inherited-looking method triggered the former false-positive.
        def encode_lq_latent(self, image):
            del image
            raise AssertionError("the callback must not infer a schema from method presence")

        def prepare_sampling_inputs(self, batch, *, from_model_step):
            assert set(batch) == {"image", "noisy_image", "buffers"}
            assert from_model_step is True
            return SamplingInputs(
                inference_batch=inference_batch,
                conditioning_images=(conditioning,),
                reference_image=reference,
            )

        def generate_samples_from_batch(self, batch, **kwargs):
            assert batch is inference_batch
            assert kwargs["cfg_scale"] == 1.0
            return generated

    callback = EveryNDrawSample(every_n=1, guidance=[1.0], resize_wandb_image=False)
    callback.data_parallel_id = 0
    captured = {}

    def _capture_save(images, batch_size, base_path):
        captured["images"] = images
        captured["batch_size"] = batch_size
        captured["base_path"] = base_path
        return "sample.jpg"

    callback.run_save = _capture_save
    monkeypatch.setattr(draw_sample_module, "is_tp_cp_pp_rank0", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    result = callback.sample(
        trainer=None,
        model=_Model(),
        data_batch={
            "image": torch.zeros(1, 3, 4, 4),
            "noisy_image": torch.ones(1, 3, 4, 4),
            "buffers": torch.ones(1, 13, 4, 4),
        },
        output_batch={},
        loss=torch.tensor(0.0),
        iteration=10,
    )

    assert result == "sample.jpg"
    assert all(
        actual is expected
        for actual, expected in zip(captured["images"], (conditioning, generated, reference), strict=True)
    )
    assert captured["batch_size"] == 1
