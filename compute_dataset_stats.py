#!/usr/bin/env python3
"""
compute_dataset_stats.py
========================
TMDb 멀티모달 데이터셋 디렉토리에서 이미지 채널별 mean/std를 계산한다.

특징:
- 프로젝트 내부 모듈에 의존하지 않는 독립 실행 스크립트
- annotations.json + image 경로만 있으면 다른 PC에서도 실행 가능
- preprocess.py 와 같은 split 방식(70/15/15, seed=42)으로 train 통계 계산 가능
- 전체(all) 또는 train split 기준으로 선택 가능
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15


def parse_args():
    parser = argparse.ArgumentParser(description="데이터셋 이미지 mean/std 계산")
    parser.add_argument(
        "data_dir",
        help="데이터셋 루트 경로 (annotations.json 이 있어야 함)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="통계 계산 전에 적용할 Resize 크기 (기본: 224)",
    )
    parser.add_argument(
        "--split",
        choices=["all", "train"],
        default="train",
        help="all 전체 기준 또는 train split 기준 통계 (기본: train)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="train split 생성용 시드 (기본: 42)",
    )
    return parser.parse_args()


def load_annotations(data_dir: Path):
    ann_path = data_dir / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"annotations.json 이 없습니다: {ann_path}")
    with ann_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_indices(n_samples: int, split: str, seed: int):
    if split == "all":
        return list(range(n_samples))

    n_train = int(n_samples * TRAIN_RATIO)
    n_val = int(n_samples * VAL_RATIO)
    n_test = n_samples - n_train - n_val
    lengths = [n_train, n_val, n_test]

    subsets = torch.utils.data.random_split(
        list(range(n_samples)),
        lengths,
        generator=torch.Generator().manual_seed(seed),
    )
    train_subset = subsets[0]
    return list(train_subset.indices)


def compute_stats(data_dir: Path, samples: list[dict], indices: list[int], image_size: int):
    to_tensor = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sq_sum = torch.zeros(3, dtype=torch.float64)
    num_pixels = 0

    for idx in indices:
        item = samples[idx]
        img_path = data_dir / item["image"]
        image = Image.open(img_path).convert("RGB")
        tensor = to_tensor(image).to(dtype=torch.float64)  # [3, H, W], 0~1
        channel_sum += tensor.sum(dim=(1, 2))
        channel_sq_sum += (tensor ** 2).sum(dim=(1, 2))
        num_pixels += tensor.shape[1] * tensor.shape[2]

    mean = channel_sum / num_pixels
    var = (channel_sq_sum / num_pixels) - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=0.0))
    return mean, std


def fmt(values: torch.Tensor):
    return "[" + ", ".join(f"{float(v):.6f}" for v in values) + "]"


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    samples = load_annotations(data_dir)
    indices = select_indices(len(samples), args.split, args.seed)
    mean, std = compute_stats(data_dir, samples, indices, args.image_size)

    print(f"data_dir      : {data_dir.resolve()}")
    print(f"split         : {args.split}")
    print(f"image_size    : {args.image_size}")
    print(f"n_samples     : {len(indices)}")
    print(f"mean          : {fmt(mean)}")
    print(f"std           : {fmt(std)}")
    print("")
    print("preprocess.py 용:")
    print(f"transforms.Normalize(mean={fmt(mean)}, std={fmt(std)})")


if __name__ == "__main__":
    main()
