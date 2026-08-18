# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import pid._ext.imaginaire.trainer as trainer_module
from pid._ext.imaginaire.trainer import ImaginaireTrainer


class _Callbacks:
    def __init__(self):
        self.events = []

    def on_validation_start(self, *args, **kwargs):
        model = args[0]
        assert model.training is False
        torch.rand(())
        self.events.append("start")

    def on_validation_step_start(self, *args, **kwargs):
        self.events.append("step_start")

    def on_validation_step_end(self, *args, **kwargs):
        self.events.append("step_end")

    def on_validation_end(self, *args, **kwargs):
        model = args[0]
        assert model.training is False
        torch.rand(())
        self.events.append("end")


class _ValidationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scope_events = []

    @contextmanager
    def ema_scope(self, context=None, is_cpu=False):
        del is_cpu
        self.scope_events.append(("enter", context))
        try:
            yield
        finally:
            self.scope_events.append(("exit", context))

    def validation_step(self, data_batch, iteration):
        assert self.training is False
        assert data_batch["image"].device.type == "cpu"
        loss = torch.rand(())
        return {"loss": loss, "iteration": iteration}, loss


@pytest.mark.L0
def test_validate_uses_model_ema_scope_and_restores_mode_and_rng(monkeypatch):
    trainer = object.__new__(ImaginaireTrainer)
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(max_val_iter=1, seed=123))
    trainer.callbacks = _Callbacks()
    model = _ValidationModel()
    model.train()
    rng_state = torch.random.get_rng_state()

    monkeypatch.setattr(trainer_module.misc, "to", lambda data, **kwargs: data)
    trainer.validate(model, [{"image": torch.zeros(1, 3, 4, 4)}], iteration=5)

    assert model.training is True
    assert model.scope_events == [("enter", "validation"), ("exit", "validation")]
    assert trainer.callbacks.events == ["start", "step_start", "step_end", "end"]
    assert torch.equal(torch.random.get_rng_state(), rng_state)
