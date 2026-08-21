#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run path-tracing denoising inference on deterministic Zarr crops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pid._src.configs.pid_training.defaults.dataloader_pt import PT_ZARR_SOURCES
from pid._src.datasets.zarr_path_tracing_dataset import ZarrPathTracingDataset
from pid._src.utils.model_loader import load_model_from_checkpoint


DEFAULT_EXPERIMENT = "pid_pt_teacher_512crop_4bs"
DEFAULT_SOURCE = "gfxr_cp"
DEFAULT_NUM_SAMPLES = 4
DEFAULT_NUM_STEPS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Denoise deterministic crops from the path-tracing Zarr dataset."
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="DCP iter_XXXXXXXXX directory or a consolidated .pth checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory in which inference images and metadata are written.",
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help=f"Registered model experiment (default: {DEFAULT_EXPERIMENT}).",
    )
    parser.add_argument(
        "--source",
        choices=sorted(PT_ZARR_SOURCES),
        default=DEFAULT_SOURCE,
        help=f"Path-tracing Zarr dataset source (default: {DEFAULT_SOURCE}).",
    )
    parser.add_argument(
        "--zarr_root",
        type=Path,
        help="Override the Zarr root configured for --source.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Maximum number of dataset samples to process (default: {DEFAULT_NUM_SAMPLES}).",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=DEFAULT_NUM_STEPS,
        help=f"Number of diffusion denoising steps (default: {DEFAULT_NUM_STEPS}).",
    )
    parser.add_argument(
        "--no_ema",
        action="store_true",
        help="Load the regular (non-EMA) network weights instead of EMA weights.",
    )
    parser.add_argument(
        "--warm_start_t",
        type=float,
        default=None,
        help=(
            "When set, initialise the diffusion chain from a partially-noised version of "
            "noisy_image rather than pure Gaussian noise. Uses the rectified-flow forward "
            "process z = (1-t)*noisy_image + t*noise and then runs the solver from "
            "t_start=warm_start_t down to ~0. Must be in (0, 1). "
            "Good starting values: 0.3 (light correction) to 0.7 (more freedom). "
            "Default: None (pure Gaussian noise, full reverse diffusion)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic crop/SPP selection and diffusion sampling.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "all"),
        default="val",
        help="Dataset split to sample (default: val).",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=512,
        help="Square crop size; must be divisible by 16 (default: 512).",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.crop_size <= 0 or args.crop_size % 16 != 0:
        raise ValueError(f"--crop_size must be a positive multiple of 16, got {args.crop_size}")
    if args.num_samples <= 0:
        raise ValueError(f"--num_samples must be positive, got {args.num_samples}")
    if args.num_steps <= 0:
        raise ValueError(f"--num_steps must be positive, got {args.num_steps}")
    if args.warm_start_t is not None and not (0.0 < args.warm_start_t < 1.0):
        raise ValueError(f"--warm_start_t must be in (0, 1), got {args.warm_start_t}")
    if args.zarr_root is not None and not args.zarr_root.is_dir():
        raise FileNotFoundError(f"Zarr root does not exist: {args.zarr_root}")
    if args.ckpt.suffix == ".pth":
        if not args.ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint file does not exist: {args.ckpt}")
    elif not args.ckpt.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {args.ckpt}")
    elif not (args.ckpt / "model").is_dir():
        raise ValueError(
            "A DCP checkpoint must point to the iter_XXXXXXXXX directory containing model/: "
            f"{args.ckpt}"
        )


def _to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a three-channel tensor in [-1, 1] to an RGB image."""
    array = ((tensor.detach().float().clamp(-1, 1) + 1.0) * 127.5).round()
    array = array.to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(np.asarray(array), mode="RGB")


def _save_sample(
    output_dir: Path,
    *,
    index: int,
    spp: float,
    noisy: torch.Tensor,
    denoised: torch.Tensor,
    target: torch.Tensor,
    scene_dir: str,
    shard: str,
    frame_index: int,
) -> None:
    sample_dir = output_dir / f"{index:04d}_spp{spp:g}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    images = {
        "noisy": _to_image(noisy),
        "denoised": _to_image(denoised),
        "target": _to_image(target),
    }
    for name, image in images.items():
        image.save(sample_dir / f"{name}.png")

    comparison = Image.new("RGB", (sum(image.width for image in images.values()), images["noisy"].height))
    x_offset = 0
    for image in images.values():
        comparison.paste(image, (x_offset, 0))
        x_offset += image.width
    comparison.save(sample_dir / "comparison_noisy_denoised_target.png")

    metadata = {
        "dataset_index": index,
        "scene_dir": scene_dir,
        "shard": shard,
        "frame_index": frame_index,
        "spp": spp,
    }
    with (sample_dir / "metadata.json").open("w") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("PT inference requires a CUDA-capable GPU")

    args.output.mkdir(parents=True, exist_ok=True)

    model, _ = load_model_from_checkpoint(
        experiment_name=args.experiment,
        checkpoint_path=str(args.ckpt),
        config_file="pid/_src/configs/pid_training/config.py",
        enable_fsdp=False,
        strict=False,
        load_ema_to_reg=not args.no_ema,
        seed=args.seed,
    )
    model.eval()

    source = dict(PT_ZARR_SOURCES[args.source])
    if args.zarr_root is not None:
        source["zarr_root"] = str(args.zarr_root)
    dataset = ZarrPathTracingDataset(
        **source,
        crop_size=(args.crop_size, args.crop_size),
        split=args.split,
        validation_fraction=0.1,
        split_seed=0,
        random_seed=args.seed,
    )

    num_samples = min(args.num_samples, len(dataset))
    print(
        f"Running {num_samples} samples from {source['zarr_root']} split={args.split!r}; "
        f"writing to {args.output}"
    )

    with torch.inference_mode():
        for index in range(num_samples):
            item = dataset[index]
            batch = {
                "noisy_image": item["noisy_image"].unsqueeze(0),
                "buffers": item["buffers"].unsqueeze(0),
                "spp": torch.tensor([item["spp"]], dtype=torch.float32),
            }
            denoised = model.generate_samples_from_batch(
                batch,
                num_steps=args.num_steps,
                seed=args.seed + index,
                warm_start_t=args.warm_start_t,
            )[0, :, 0]
            scene_dir, shard, frame_index = dataset._index[index]
            _save_sample(
                args.output,
                index=index,
                spp=float(item["spp"]),
                noisy=item["noisy_image"],
                denoised=denoised,
                target=item["image"],
                scene_dir=scene_dir,
                shard=shard,
                frame_index=frame_index,
            )
            print(f"[{index + 1}/{num_samples}] wrote sample {index:04d} (SPP={item['spp']:g})")


if __name__ == "__main__":
    main()
