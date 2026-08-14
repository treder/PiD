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

Reads clean converged renders and noisy renders at variable SPP from zarr v3
stores, applies a consistent random crop, normalizes each buffer type, and
returns a dict suitable for PidModelPT training.

Requires zarr >= 3.0 (pip install "zarr>=3.0").

Directory layout:
    zarr_root/
        {scene}/
            {shard}/
                1080p/
                    zarr.json          (group metadata; attributes.sequence_length)
                    color/             (clean reference, [T, 3, H, W], float16)
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
"""

import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

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
    "color_hspp",
    "color_1spp",
    "color_4spp",
    "color_8spp",
    "color_16spp",
)
_DEFAULT_KEY_TO_SPP: Dict[str, float] = {
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
        clean_key: Array name for the clean/converged reference image.
        crop_size: (H, W) spatial crop. Both dimensions must be divisible by 16.
        log1p_max_clean: HDR normalisation ceiling for clean image (log1p scale).
        log1p_max_noisy: HDR normalisation ceiling for noisy image (log1p scale).
        max_depth: Linear depth normalisation ceiling (scene units).
    """

    BUFFER_KEYS: List[str] = _BUFFER_KEYS

    def __init__(
        self,
        zarr_root: str,
        noisy_keys: Sequence[str] = _DEFAULT_NOISY_KEYS,
        noisy_key_to_spp: Dict[str, float] = _DEFAULT_KEY_TO_SPP,
        clean_key: str = "color",
        crop_size: Tuple[int, int] = (512, 512),
        log1p_max_clean: float = 10.0,
        log1p_max_noisy: float = 10.0,
        max_depth: float = 1000.0,
    ):
        super().__init__()

        if crop_size[0] % 16 != 0 or crop_size[1] % 16 != 0:
            raise ValueError(f"crop_size must be divisible by 16 (patch_size), got {crop_size}")

        self.zarr_root = zarr_root
        self.noisy_keys = list(noisy_keys)
        self.noisy_key_to_spp = dict(noisy_key_to_spp)
        self.clean_key = clean_key
        self.crop_size = crop_size
        self.log1p_max_clean = log1p_max_clean
        self.log1p_max_noisy = log1p_max_noisy
        self.max_depth = max_depth

        self._index: List[Tuple[str, str, int]] = self._build_index(zarr_root)
        # Lazily populated per DataLoader worker process to avoid fork-safety issues.
        self._group_cache: Dict[Tuple[str, str], object] = {}

        if len(self._index) == 0:
            raise RuntimeError(f"No valid zarr sequences found under {zarr_root!r}")

    # =========================================================================
    # Index construction
    # =========================================================================

    def _build_index(self, zarr_root: str) -> List[Tuple[str, str, int]]:
        """Walk zarr_root → scenes → shards, build list of (scene_dir, shard, frame_idx)."""
        index = []
        if not os.path.isdir(zarr_root):
            raise FileNotFoundError(f"zarr_root does not exist: {zarr_root!r}")

        for scene_name in sorted(os.listdir(zarr_root)):
            scene_dir = os.path.join(zarr_root, scene_name)
            if not os.path.isdir(scene_dir):
                continue
            for shard_name in sorted(os.listdir(scene_dir)):
                meta_path = os.path.join(scene_dir, shard_name, "1080p", "zarr.json")
                if not os.path.isfile(meta_path):
                    continue
                with open(meta_path) as f:
                    meta = json.load(f)
                seq_len = meta.get("attributes", {}).get("sequence_length")
                if seq_len is None:
                    continue
                for frame_idx in range(int(seq_len)):
                    index.append((scene_dir, shard_name, frame_idx))

        return index

    # =========================================================================
    # Zarr group cache (lazy, per-worker)
    # =========================================================================

    def _get_group(self, scene_dir: str, shard: str):
        key = (scene_dir, shard)
        if key not in self._group_cache:
            import zarr  # requires zarr >= 3.0

            path = os.path.join(scene_dir, shard, "1080p")
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
        group = self._get_group(scene_dir, shard)

        crop_h, crop_w = self.crop_size

        # Determine image dimensions from the clean array shape
        arr_shape = group[self.clean_key].shape  # [T, C, H, W]
        full_h, full_w = arr_shape[2], arr_shape[3]

        # Random crop offsets (consistent across all arrays for this sample)
        crop_y = int(np.random.randint(0, full_h - crop_h + 1))
        crop_x = int(np.random.randint(0, full_w - crop_w + 1))

        def _read(key: str) -> np.ndarray:
            """Read [frame_idx, :, crop_y:y+H, crop_x:x+W] as float32."""
            return np.asarray(
                group[key][frame_idx, :, crop_y : crop_y + crop_h, crop_x : crop_x + crop_w],
                dtype=np.float32,
            )

        # --- Clean image ---
        clean_np = _read(self.clean_key)  # [3, H, W]
        clean_t = torch.from_numpy(self._log1p_norm(clean_np, self.log1p_max_clean))

        # --- Noisy render: pick randomly among available SPP arrays ---
        available_noisy = [k for k in self.noisy_keys if k in group]
        if not available_noisy:
            raise RuntimeError(
                f"None of the requested noisy keys {self.noisy_keys} found in "
                f"{scene_dir}/{shard}/1080p"
            )
        noisy_key = available_noisy[np.random.randint(0, len(available_noisy))]
        spp_value = float(self.noisy_key_to_spp.get(noisy_key, 1.0))

        noisy_np = _read(noisy_key)  # [3, H, W]
        noisy_t = torch.from_numpy(self._log1p_norm(noisy_np, self.log1p_max_noisy))

        # --- G-buffers: read and concatenate ---
        buffer_parts = [_read(bkey) for bkey in self.BUFFER_KEYS]
        buf_np = np.concatenate(buffer_parts, axis=0)  # [13, H, W]
        buf_np = self._normalize_buffers(buf_np, crop_h, crop_w)
        buf_t = torch.from_numpy(buf_np)

        return {
            "image": clean_t,        # [3, H, W] in [-1, 1]
            "noisy_image": noisy_t,  # [3, H, W] in [-1, 1]
            "buffer": buf_t,         # [13, H, W] in [-1, 1]
            "spp": spp_value,        # float scalar — SPP of the noisy render
            "caption": "",           # placeholder; not consumed by PTConditioner
        }
