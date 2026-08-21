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

Each registered config is a LazyCall to get_cached_replay_dataloader wrapping a
named Zarr dataset mixture. The names follow the pattern:
    pt_zarr_{mixture}_{batch_size}bs_{crop_size}crop

so they can be overridden in experiment configs with e.g.:
    {"override /data_train": "pt_zarr_gfxr_cp_4bs_512crop"}
"""

from hydra.core.config_store import ConfigStore

from pid._ext.imaginaire.dataloaders.cached_replay_dataloader import get_cached_replay_dataloader
from pid._ext.imaginaire.lazy_config import LazyCall as L
from pid._src.datasets.zarr_path_tracing_dataset import ZarrPathTracingMixtureDataset

PT_ZARR_BASE_ROOT = "/dlrs/home/data/ml_denoiser_master_data/datasets/Dropbox/Zarr"

# Physical sources for the current same-resolution training experiment. Inputs
# come from 1080p; the aligned 2160p TAA target is area-downsampled by 2x in the
# dataset. A native 2160p target requires a separate sr_scale=2 experiment.
PT_ZARR_SOURCES = {
    name: {
        "zarr_root": f"{PT_ZARR_BASE_ROOT}/{name}",
        "data_group": "1080p",
        "clean_group": "2160p_taa",
        "clean_key": "target",
        "clean_downscale_factor": 2,
    }
    for name in (
        "gfxr_aw2",
        "gfxr_aw2_be",
        "gfxr_cp",
        "gfxr_cp_be",
        "gfxr_ij",
        "gfxr_ij_be",
    )
}

# Select a single dataset by its own name, or edit/add a named mixture here.
# Weights are relative (0.7/0.3 is equivalent to 7/3); zero disables a source.
PT_ZARR_MIXTURES = {name: {name: 1.0} for name in PT_ZARR_SOURCES}
PT_ZARR_MIXTURES["mixed"] = {
    "gfxr_cp": 0.5,
    "gfxr_ij": 0.3,
    "gfxr_aw2": 0.2,
}

_BATCH_SIZES = [1, 2, 4, 8, 16]
_CROP_SIZES = [512, 768, 1024]
_VALIDATION_FRACTION = 0.1
_SPLIT_SEED = 0


def _make_pt_loader(
    mixture: dict[str, float],
    batch_size: int,
    crop_size: int,
    *,
    split: str,
    num_workers: int = 4,
):
    zarr_roots = {name: PT_ZARR_SOURCES[name] for name in mixture}
    return L(get_cached_replay_dataloader)(
        dataset=L(ZarrPathTracingMixtureDataset)(
            zarr_roots=zarr_roots,
            sample_weights=mixture,
            crop_size=(crop_size, crop_size),
            split=split,
            validation_fraction=_VALIDATION_FRACTION,
            split_seed=_SPLIT_SEED,
            random_seed=0 if split == "val" else None,
        ),
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=True,
        webdataset=False,
        cache_replay_name="pt_zarr_dataloader",
    )


def register_pt_data():
    cs = ConfigStore.instance()
    for mixture_name, mixture in PT_ZARR_MIXTURES.items():
        for batch_size in _BATCH_SIZES:
            for crop_size in _CROP_SIZES:
                train_node = _make_pt_loader(
                    mixture=mixture,
                    batch_size=batch_size,
                    crop_size=crop_size,
                    split="train",
                )
                val_node = _make_pt_loader(
                    mixture=mixture,
                    batch_size=batch_size,
                    crop_size=crop_size,
                    split="val",
                )
                config_name = f"pt_zarr_{mixture_name}_{batch_size}bs_{crop_size}crop"
                cs.store(
                    group="data_train",
                    package="dataloader_train",
                    name=config_name,
                    node=train_node,
                )
                cs.store(
                    group="data_val",
                    package="dataloader_val",
                    name=config_name,
                    node=val_node,
                )
