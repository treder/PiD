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
Registers zarr path-tracing dataloaders for the ConfigStore.

Each registered config is a LazyCall to get_cached_replay_dataloader wrapping
ZarrPathTracingDataset.  The names follow the pattern:
    pt_zarr_{batch_size}bs_{crop_size}crop

so they can be overridden in experiment configs with e.g.:
    {"override /data_train": "pt_zarr_4bs_512crop"}
"""

from hydra.core.config_store import ConfigStore

from pid._ext.imaginaire.dataloaders.cached_replay_dataloader import get_cached_replay_dataloader
from pid._ext.imaginaire.lazy_config import LazyCall as L
from pid._src.datasets.zarr_path_tracing_dataset import ZarrPathTracingDataset

_PT_ZARR_ROOT = "/Users/matthias.treder/data/zarr/gfxr_cp"

_BATCH_SIZES = [1, 2, 4, 8]
_CROP_SIZES = [512, 768, 1024]


def _make_pt_loader(zarr_root: str, batch_size: int, crop_size: int, num_workers: int = 4):
    return L(get_cached_replay_dataloader)(
        dataset=L(ZarrPathTracingDataset)(
            zarr_root=zarr_root,
            crop_size=(crop_size, crop_size),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        webdataset=False,
        cache_replay_name="pt_zarr_dataloader",
    )


def register_pt_data():
    cs = ConfigStore.instance()
    for batch_size in _BATCH_SIZES:
        for crop_size in _CROP_SIZES:
            node = _make_pt_loader(
                zarr_root=_PT_ZARR_ROOT,
                batch_size=batch_size,
                crop_size=crop_size,
            )
            cs.store(
                group="data_train",
                package="dataloader_train",
                name=f"pt_zarr_{batch_size}bs_{crop_size}crop",
                node=node,
            )
            cs.store(
                group="data_val",
                package="dataloader_val",
                name=f"pt_zarr_{batch_size}bs_{crop_size}crop",
                node=node,
            )
