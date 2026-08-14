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

# Registers:
# - pid_pt_noisy_buffers: noisy PT render (no dropout) + G-buffers (no dropout)

from hydra.core.config_store import ConfigStore

from pid._ext.imaginaire.lazy_config import LazyCall as L
from pid._src.modules.conditioner import (
    BufferTensorDrop,
    LQTensorDrop,
    PTConditioner,
)

# Noisy PT image + G-buffers — no CFG dropout (both are always present)
Pid_PT_NoisyBuffers_Config = L(PTConditioner)(
    noisy_image=L(LQTensorDrop)(
        input_key="noisy_image",
        output_key="noisy_image",
        dropout_rate=0.0,
    ),
    buffers=L(BufferTensorDrop)(
        input_key="buffer",
        output_key="buffers",
        dropout_rate=0.0,
    ),
)


def register_conditioner_pid_pt():
    cs = ConfigStore.instance()
    cs.store(
        group="conditioner",
        package="model.config.conditioner",
        name="pid_pt_noisy_buffers",
        node=Pid_PT_NoisyBuffers_Config,
    )
