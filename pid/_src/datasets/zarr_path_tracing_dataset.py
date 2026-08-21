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
Zarr v3 dataset for path-tracing denoising.

Reads TAA reference renders and noisy renders at variable SPP from zarr v3
stores, applies a consistent random crop, normalizes each buffer type, and
returns a dict suitable for PidModelPT training.

Requires zarr >= 3.0 (pip install "zarr>=3.0").

Directory layout:
    zarr_root/
        {scene}/
            {shard}/
                1080p/
                    zarr.json          (group metadata; attributes.sequence_length)
                    color/             (noisy render, [T, 3, H, W], float16)
                    color_hspp/        (half-SPP noisy render, [T, 3, H, W], float16)
                    color_1spp/        (1-SPP noisy render, optional)
                    color_4spp/        ...
                    color_8spp/        ...
                    color_16spp/       ...
                    specular_albedo/   ([T, 3, H, W], float16)
                    diffuse_albedo/    ([T, 3, H, W], float16)
                    normal/            ([T, 3, H, W], float16)
                    depth/             ([T, 1, H, W], float32)
                    roughness/         ([T, 1, H, W], float16)
                    mv/                ([T, 2, H, W], float32)
                2160p_taa/
                    zarr.json
                    target/            (clean TAA reference, [T, 3, H, W], float16)
"""

import hashlib
import json
import math
import os
from bisect import bisect_right
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.utils.data

# G-buffer arrays in concat order → 3+3+3+1+1+2 = 13 channels total
_BUFFER_KEYS: List[str] = [
    "specular_albedo",  # [3]
    "diffuse_albedo",   # [3]
    "normal",           # [3]
    "depth",            # [1]
    "roughness",        # [1]
    "mv",               # [2]
]

# Known SPP keys and their SPP values (used when the dataset randomly picks SPP)
_DEFAULT_NOISY_KEYS: Tuple[str, ...] = (
    "color",
    "color_hspp",
    "color_1spp",
    "color_4spp",
    "color_8spp",
    "color_16spp",
)
_DEFAULT_KEY_TO_SPP: Dict[str, float] = {
    "color": 1.0,
    "color_hspp": 0.5,
    "color_1spp": 1.0,
    "color_4spp": 4.0,
    "color_8spp": 8.0,
    "color_16spp": 16.0,
}


class ZarrPathTracingDataset(torch.utils.data.Dataset):
    """Dataset reading path-tracing renders from zarr v3 stores.

    Each __getitem__ call:
      1. Picks a (scene, shard, frame) from the pre-built index.
      2. Randomly selects one of the available noisy SPP arrays.
      3. Applies the same random crop to all arrays.
      4. Normalizes each channel group appropriately.
      5. Returns a dict with keys: image, noisy_image, buffer, spp, caption.

    Args:
        zarr_root: Root directory containing scene sub-directories.
        noisy_keys: Tuple of array names to consider as noisy inputs.
            At runtime only those present in the zarr group are used.
        noisy_key_to_spp: Mapping from array name to its SPP value.
        data_group: Shard-relative Zarr group containing noisy images and buffers.
        clean_group: Shard-relative Zarr group containing the clean reference.
        clean_key: Array name in clean_group for the clean reference image.
        clean_downscale_factor: Integer spatial ratio between the clean reference
            and noisy input. The aligned clean crop is area-downsampled by this
            factor before normalization.
        crop_size: (H, W) input/output crop. Both dimensions must be divisible by 16.
        log1p_max_clean: HDR normalisation ceiling for clean image (log1p scale).
        log1p_max_noisy: HDR normalisation ceiling for noisy image (log1p scale).
        max_depth: Linear depth normalisation ceiling (scene units).
        split: ``"all"``, ``"train"``, or ``"val"``. Train/val partition
            scene groups deterministically, keeping variants of one scene together.
        validation_fraction: Fraction of scene groups assigned to ``"val"``.
        split_seed: Stable seed for the scene-group partition.
        random_seed: If set, SPP and crop choices are deterministic per index.
    """

    BUFFER_KEYS: List[str] = _BUFFER_KEYS

    def __init__(
        self,
        zarr_root: str,
        noisy_keys: Sequence[str] = _DEFAULT_NOISY_KEYS,
        noisy_key_to_spp: Dict[str, float] = _DEFAULT_KEY_TO_SPP,
        data_group: str = "1080p",
        clean_group: str = "2160p_taa",
        clean_key: str = "target",
        clean_downscale_factor: int = 2,
        crop_size: Tuple[int, int] = (512, 512),
        log1p_max_clean: float = 50.0,
        log1p_max_noisy: float = 50.0,
        max_depth: float = 1000.0,
        split: str = "all",
        validation_fraction: float = 0.1,
        split_seed: int = 0,
        random_seed: Optional[int] = None,
    ):
        super().__init__()

        if crop_size[0] % 16 != 0 or crop_size[1] % 16 != 0:
            raise ValueError(f"crop_size must be divisible by 16 (patch_size), got {crop_size}")
        if clean_downscale_factor < 1:
            raise ValueError(
                f"clean_downscale_factor must be a positive integer, got {clean_downscale_factor}"
            )
        if split not in {"all", "train", "val"}:
            raise ValueError(f"split must be 'all', 'train', or 'val', got {split!r}")
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError(f"validation_fraction must be in (0, 1), got {validation_fraction}")

        self.zarr_root = zarr_root
        self.noisy_keys = list(noisy_keys)
        self.noisy_key_to_spp = dict(noisy_key_to_spp)
        self.data_group = data_group
        self.clean_group = clean_group
        self.clean_key = clean_key
        self.clean_downscale_factor = int(clean_downscale_factor)
        self.crop_size = crop_size
        self.log1p_max_clean = log1p_max_clean
        self.log1p_max_noisy = log1p_max_noisy
        self.max_depth = max_depth
        self.split = split
        self.validation_fraction = float(validation_fraction)
        self.split_seed = int(split_seed)
        self.random_seed = random_seed

        self._index: List[Tuple[str, str, int]] = self._build_index(zarr_root)
        # Lazily populated per DataLoader worker process to avoid fork-safety issues.
        self._group_cache: Dict[Tuple[str, str, str], object] = {}

        if len(self._index) == 0:
            raise RuntimeError(f"No valid zarr sequences found under {zarr_root!r}")

    # =========================================================================
    # Index construction
    # =========================================================================

    @staticmethod
    def _scene_group(scene_name: str) -> str:
        """Group capture variants from the same named scene into one split."""
        return scene_name.split("_Data_", 1)[0]

    def _include_scene(self, scene_name: str) -> bool:
        if self.split == "all":
            return True
        group = self._scene_group(scene_name)
        digest = hashlib.sha256(f"{self.split_seed}:{group}".encode()).digest()
        score = int.from_bytes(digest[:8], byteorder="big") / 2**64
        is_validation = score < self.validation_fraction
        return is_validation if self.split == "val" else not is_validation

    def _build_index(self, zarr_root: str) -> List[Tuple[str, str, int]]:
        """Walk zarr_root → scenes → shards, build list of (scene_dir, shard, frame_idx)."""
        index = []
        if not os.path.isdir(zarr_root):
            raise FileNotFoundError(f"zarr_root does not exist: {zarr_root!r}")

        for scene_name in sorted(os.listdir(zarr_root)):
            scene_dir = os.path.join(zarr_root, scene_name)
            if not os.path.isdir(scene_dir) or not self._include_scene(scene_name):
                continue
            for shard_name in sorted(os.listdir(scene_dir)):
                data_meta_path = os.path.join(scene_dir, shard_name, self.data_group, "zarr.json")
                clean_meta_path = os.path.join(scene_dir, shard_name, self.clean_group, "zarr.json")
                clean_array_path = os.path.join(
                    scene_dir, shard_name, self.clean_group, self.clean_key, "zarr.json"
                )
                if not (
                    os.path.isfile(data_meta_path)
                    and os.path.isfile(clean_meta_path)
                    and os.path.isfile(clean_array_path)
                ):
                    continue
                with open(data_meta_path) as f:
                    data_meta = json.load(f)
                with open(clean_meta_path) as f:
                    clean_meta = json.load(f)
                data_seq_len = data_meta.get("attributes", {}).get("sequence_length")
                clean_seq_len = clean_meta.get("attributes", {}).get("sequence_length")
                if data_seq_len is None or clean_seq_len is None:
                    continue
                if int(data_seq_len) != int(clean_seq_len):
                    raise ValueError(
                        f"Sequence length mismatch for {scene_dir}/{shard_name}: "
                        f"{self.data_group}={data_seq_len}, {self.clean_group}={clean_seq_len}"
                    )
                for frame_idx in range(int(data_seq_len)):
                    index.append((scene_dir, shard_name, frame_idx))

        return index

    # =========================================================================
    # Zarr group cache (lazy, per-worker)
    # =========================================================================

    def _get_group(self, scene_dir: str, shard: str, group_name: str):
        key = (scene_dir, shard, group_name)
        if key not in self._group_cache:
            import zarr  # requires zarr >= 3.0

            path = os.path.join(scene_dir, shard, group_name)
            self._group_cache[key] = zarr.open_group(path, mode="r")
        return self._group_cache[key]

    # =========================================================================
    # Normalisation helpers
    # =========================================================================

    @staticmethod
    def _log1p_norm(x: np.ndarray, max_val: float) -> np.ndarray:
        """HDR tonemapping: log1p(clip(x,0)) / log1p(max_val) → [0,1] → [-1,1]."""
        x = np.log1p(np.clip(x, 0.0, None)) / math.log1p(max_val)
        return np.clip(x, 0.0, 1.0) * 2.0 - 1.0

    @staticmethod
    def _linear_01_to_11(x: np.ndarray) -> np.ndarray:
        """Scale [0,1] → [-1,1]."""
        return np.clip(x, 0.0, 1.0) * 2.0 - 1.0

    def _normalize_buffers(
        self, buf: np.ndarray, crop_h: int, crop_w: int
    ) -> np.ndarray:
        """Normalize concatenated buffer tensor [13, H, W] to approximately [-1, 1].

        Channel layout (matches BUFFER_KEYS order):
          [0:3]   specular_albedo  — physically [0,1]; scale linearly
          [3:6]   diffuse_albedo   — physically [0,1]; scale linearly
          [6:9]   normal           — world-space unit vectors in [-1,1]; clamp
          [9:10]  depth            — linear scene units; normalize by max_depth
          [10:11] roughness        — [0,1]; scale linearly
          [11:13] mv               — motion vectors in pixels; divide by crop dims
        """
        out = buf.copy()
        # Albedos: linear [0,1] → [-1,1]
        out[0:6] = self._linear_01_to_11(buf[0:6])
        # Normals: already in [-1,1]
        out[6:9] = np.clip(buf[6:9], -1.0, 1.0)
        # Depth: linear normalize
        out[9:10] = np.clip(buf[9:10], 0.0, self.max_depth) / self.max_depth * 2.0 - 1.0
        # Roughness: linear [0,1] → [-1,1]
        out[10:11] = self._linear_01_to_11(buf[10:11])
        # Motion vectors: normalize by crop dimensions
        out[11:12] = np.clip(buf[11:12] / crop_w, -1.0, 1.0)
        out[12:13] = np.clip(buf[12:13] / crop_h, -1.0, 1.0)
        return out

    # =========================================================================
    # Dataset interface
    # =========================================================================

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        scene_dir, shard, frame_idx = self._index[idx]
        data_group = self._get_group(scene_dir, shard, self.data_group)
        clean_group = self._get_group(scene_dir, shard, self.clean_group)

        crop_h, crop_w = self.crop_size

        if self.random_seed is None:
            randint = np.random.randint
        else:
            rng = np.random.default_rng(self.random_seed + int(idx))
            randint = rng.integers

        # Pick a noisy render and ensure the input and reference grids align.
        available_noisy = [key for key in self.noisy_keys if key in data_group]
        if not available_noisy:
            raise RuntimeError(
                f"None of the requested noisy keys {self.noisy_keys} found in "
                f"{scene_dir}/{shard}/{self.data_group}"
            )
        noisy_key = available_noisy[randint(0, len(available_noisy))]
        spp_value = float(self.noisy_key_to_spp.get(noisy_key, 1.0))

        clean_shape = clean_group[self.clean_key].shape
        noisy_shape = data_group[noisy_key].shape
        scale = self.clean_downscale_factor
        expected_clean_shape = (
            noisy_shape[0],
            noisy_shape[1],
            noisy_shape[2] * scale,
            noisy_shape[3] * scale,
        )
        if clean_shape != expected_clean_shape:
            raise ValueError(
                f"Clean/noisy shape mismatch for {scene_dir}/{shard}: "
                f"{self.clean_group}/{self.clean_key}={clean_shape}, expected "
                f"{expected_clean_shape} from {self.data_group}/{noisy_key}={noisy_shape} "
                f"with clean_downscale_factor={scale}"
            )
        full_h, full_w = noisy_shape[2], noisy_shape[3]
        if crop_h > full_h or crop_w > full_w:
            raise ValueError(
                f"crop_size {self.crop_size} exceeds image size {(full_h, full_w)} "
                f"for {scene_dir}/{shard}"
            )

        # Random crop offsets (consistent across all arrays for this sample)
        crop_y = int(randint(0, full_h - crop_h + 1))
        crop_x = int(randint(0, full_w - crop_w + 1))

        def _read_input(key: str) -> np.ndarray:
            """Read an input-resolution crop as float32."""
            return np.asarray(
                data_group[key][
                    frame_idx,
                    :,
                    crop_y : crop_y + crop_h,
                    crop_x : crop_x + crop_w,
                ],
                dtype=np.float32,
            )

        # --- Clean image: aligned high-resolution crop, then area downsample ---
        clean_crop_h = crop_h * scale
        clean_crop_w = crop_w * scale
        clean_y = crop_y * scale
        clean_x = crop_x * scale
        clean_np = np.asarray(
            clean_group[self.clean_key][
                frame_idx,
                :,
                clean_y : clean_y + clean_crop_h,
                clean_x : clean_x + clean_crop_w,
            ],
            dtype=np.float32,
        )
        if scale > 1:
            channels = clean_np.shape[0]
            clean_np = clean_np.reshape(channels, crop_h, scale, crop_w, scale).mean(
                axis=(2, 4)
            )
        clean_t = torch.from_numpy(self._log1p_norm(clean_np, self.log1p_max_clean))

        # --- Noisy render ---
        noisy_np = _read_input(noisy_key)  # [3, H, W]
        noisy_t = torch.from_numpy(self._log1p_norm(noisy_np, self.log1p_max_noisy))

        # --- G-buffers: read and concatenate ---
        buffer_parts = [_read_input(bkey) for bkey in self.BUFFER_KEYS]
        buf_np = np.concatenate(buffer_parts, axis=0)  # [13, H, W]
        buf_np = self._normalize_buffers(buf_np, crop_h, crop_w)
        buf_t = torch.from_numpy(buf_np)

        return {
            "image": clean_t,        # [3, H, W] in [-1, 1]
            "noisy_image": noisy_t,  # [3, H, W] in [-1, 1]
            "buffers": buf_t,        # [13, H, W] in [-1, 1]
            "spp": spp_value,        # float scalar — SPP of the noisy render
            "caption": "",           # placeholder; not consumed by PTConditioner
        }


class ZarrPathTracingMixtureDataset(torch.utils.data.Dataset):
    """Combine Zarr datasets using explicit per-epoch sampling proportions.

    Sample weights are relative weights and do not need to sum to one. The
    mixture length defaults to the sum of the enabled source lengths. Smaller
    sources are repeated when their allocation exceeds their native length;
    larger sources are sampled at evenly spaced indices when undersampled.

    Args:
        zarr_roots: Mapping from a short source name to either its Zarr root or
            a mapping containing zarr_root and per-source dataset arguments.
        sample_weights: Mapping from source name to a non-negative relative
            sampling weight. A weight of zero disables that source.
        epoch_size: Number of samples in one mixture epoch. By default, use the
            sum of the enabled datasets' native lengths.
        **dataset_kwargs: Arguments forwarded to each ZarrPathTracingDataset.
    """

    def __init__(
        self,
        zarr_roots: Mapping[str, Union[str, Mapping[str, Any]]],
        sample_weights: Optional[Mapping[str, float]] = None,
        epoch_size: Optional[int] = None,
        **dataset_kwargs,
    ):
        super().__init__()

        roots = dict(zarr_roots)
        if not roots:
            raise ValueError("zarr_roots must contain at least one source")

        weights = (
            {name: 1.0 for name in roots}
            if sample_weights is None
            else {name: float(weight) for name, weight in sample_weights.items()}
        )
        unknown_sources = set(weights) - set(roots)
        missing_sources = set(roots) - set(weights)
        if unknown_sources or missing_sources:
            raise ValueError(
                "zarr_roots and sample_weights must have identical keys; "
                f"unknown={sorted(unknown_sources)}, missing={sorted(missing_sources)}"
            )
        if any(weight < 0 for weight in weights.values()):
            raise ValueError("sample weights must be non-negative")

        enabled_names = [name for name in roots if weights[name] > 0]
        if not enabled_names:
            raise ValueError("at least one sample weight must be greater than zero")

        self.datasets = {}
        for name in enabled_names:
            source = roots[name]
            if isinstance(source, str):
                source_root = source
                source_kwargs = {}
            else:
                source_kwargs = dict(source)
                try:
                    source_root = source_kwargs.pop("zarr_root")
                except KeyError as error:
                    raise ValueError(f"Source {name!r} is missing zarr_root") from error
            source_kwargs = {**dataset_kwargs, **source_kwargs}
            self.datasets[name] = ZarrPathTracingDataset(
                zarr_root=source_root,
                **source_kwargs,
            )
        native_size = sum(len(dataset) for dataset in self.datasets.values())
        self.epoch_size = native_size if epoch_size is None else int(epoch_size)
        if self.epoch_size <= 0:
            raise ValueError(f"epoch_size must be positive, got {self.epoch_size}")

        normalized_total = sum(weights[name] for name in enabled_names)
        exact_counts = [
            self.epoch_size * weights[name] / normalized_total for name in enabled_names
        ]
        sample_counts = [int(count) for count in exact_counts]
        remainder = self.epoch_size - sum(sample_counts)
        largest_fractions = sorted(
            range(len(enabled_names)),
            key=lambda index: exact_counts[index] - sample_counts[index],
            reverse=True,
        )
        for index in largest_fractions[:remainder]:
            sample_counts[index] += 1

        self.source_names = enabled_names
        self.sample_counts = dict(zip(enabled_names, sample_counts))
        self._cumulative_counts: List[int] = []
        running_total = 0
        for count in sample_counts:
            running_total += count
            self._cumulative_counts.append(running_total)

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx: int) -> dict:
        if idx < 0:
            idx += self.epoch_size
        if idx < 0 or idx >= self.epoch_size:
            raise IndexError(idx)

        source_index = bisect_right(self._cumulative_counts, idx)
        source_name = self.source_names[source_index]
        source_start = 0 if source_index == 0 else self._cumulative_counts[source_index - 1]
        position_in_source = idx - source_start
        allocation = self.sample_counts[source_name]
        dataset = self.datasets[source_name]

        if allocation <= len(dataset):
            dataset_index = position_in_source * len(dataset) // allocation
        else:
            dataset_index = position_in_source % len(dataset)
        return dataset[dataset_index]
