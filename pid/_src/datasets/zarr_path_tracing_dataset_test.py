# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from pid._src.datasets.zarr_path_tracing_dataset import ZarrPathTracingDataset


def _make_sequence(root, scene_name: str, sequence_length: int = 2):
    shard = root / scene_name / "00001_00002"
    data_group = shard / "1080p"
    clean_group = shard / "2160p_taa"
    clean_array = clean_group / "target"
    data_group.mkdir(parents=True)
    clean_array.mkdir(parents=True)
    metadata = json.dumps({"attributes": {"sequence_length": sequence_length}})
    (data_group / "zarr.json").write_text(metadata)
    (clean_group / "zarr.json").write_text(metadata)
    (clean_array / "zarr.json").write_text("{}")


@pytest.mark.L0
def test_scene_split_is_disjoint_complete_and_groups_capture_variants(tmp_path):
    for scene_index in range(40):
        _make_sequence(tmp_path, f"game_Scene_{scene_index:04d}_Data_0001_fps30")
    _make_sequence(tmp_path, "game_Scene_0001_Data_0002_fps60")

    common = dict(
        zarr_root=str(tmp_path),
        crop_size=(16, 16),
        validation_fraction=0.2,
        split_seed=17,
    )
    all_data = ZarrPathTracingDataset(split="all", **common)
    train_data = ZarrPathTracingDataset(split="train", **common)
    val_data = ZarrPathTracingDataset(split="val", random_seed=0, **common)

    def groups(dataset):
        return {
            dataset._scene_group(scene_dir.rsplit("/", 1)[-1])
            for scene_dir, _, _ in dataset._index
        }

    all_groups = groups(all_data)
    train_groups = groups(train_data)
    val_groups = groups(val_data)
    assert train_groups
    assert val_groups
    assert train_groups.isdisjoint(val_groups)
    assert train_groups | val_groups == all_groups

    variant_group = "game_Scene_0001"
    assert (variant_group in train_groups) != (variant_group in val_groups)
    selected = train_data if variant_group in train_groups else val_data
    selected_variants = {
        scene_dir.rsplit("/", 1)[-1]
        for scene_dir, _, _ in selected._index
        if selected._scene_group(scene_dir.rsplit("/", 1)[-1]) == variant_group
    }
    assert selected_variants == {
        "game_Scene_0001_Data_0001_fps30",
        "game_Scene_0001_Data_0002_fps60",
    }
