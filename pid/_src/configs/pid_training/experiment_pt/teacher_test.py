# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from pid._src.configs.pid_training.experiment_pt.teacher import (
    PID_PT_TEACHER_512CROP_4BS,
    _build_debug_run,
)


@pytest.mark.L0
def test_pt_training_enables_bounded_validation():
    trainer = PID_PT_TEACHER_512CROP_4BS["trainer"]

    assert trainer["run_validation"] is True
    assert trainer["validation_iter"] == 2500
    assert trainer["max_val_iter"] == 50


@pytest.mark.L0
def test_pt_debug_exercises_sampling_and_validation():
    trainer = _build_debug_run(PID_PT_TEACHER_512CROP_4BS)["trainer"]

    assert trainer["validation_iter"] == 10
    assert trainer["max_val_iter"] == 2
    assert trainer["callbacks"]["every_n_sample_train_infer_20step_reg"]["every_n"] == 10
    assert trainer["callbacks"]["every_n_sample_train_infer_20step_ema"]["every_n"] == 10
