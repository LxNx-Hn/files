#!/usr/bin/env python3
"""
run_research_plan.py
====================
다음 실험 계획을 단계별로 자동 실행한다.

기본 단계:
1. single10_final_phase1_cnn : CNN 3퓨전 + CNN image_only + text_only
2. single10_final_phase2_res : RES 3퓨전 + RES image_only
"""

import argparse
import subprocess
from pathlib import Path


DEFAULT_STAGES = ["single10_final_phase1_cnn", "single10_final_phase2_res"]
ALL_STAGES = ["single10_final_phase1_cnn", "single10_final_phase2_res"]


def parse_args():
    parser = argparse.ArgumentParser(description="연구 계획 자동 실행")
    parser.add_argument("--python", default="python3", help="사용할 파이썬 실행파일")
    parser.add_argument("--base_data_dir", default="./data", help="원본 데이터셋 경로")
    parser.add_argument("--variants_root", default="./data_variants", help="variant 데이터셋 루트")
    parser.add_argument("--stages", nargs="+", default=DEFAULT_STAGES,
                        choices=ALL_STAGES, help="실행할 단계 목록")
    parser.add_argument("--gpus", nargs="+", type=int, default=None, help="launcher GPU override")
    parser.add_argument("--dry_run", action="store_true", help="명령만 출력")
    return parser.parse_args()


def run_cmd(cmd, dry_run=False):
    print("\n[cmd]", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def build_stage_commands(args, stage):
    py = args.python
    base = Path(args.base_data_dir)
    variants_root = Path(args.variants_root)
    gpu_args = []
    if args.gpus:
        gpu_args = ["--gpus", *map(str, args.gpus)]

    if stage == "single10_final_phase1_cnn":
        out_dir = variants_root / "single10_balanced"
        return [
            [
                py, "prepare_dataset_variant.py",
                "--source_dir", str(variants_root / "single10"),
                "--out_dir", str(out_dir),
                "--balance_to_min",
                "--force",
            ],
            [
                py, "preprocess.py", str(out_dir),
                "--max_text_len", "96",
            ],
            [
                py, "launcher.py",
                "--config", "experiments_single10_final_phase1_cnn.json",
                *gpu_args,
                "--dry_run",
            ],
            [
                py, "launcher.py",
                "--config", "experiments_single10_final_phase1_cnn.json",
                *gpu_args,
            ],
        ]

    if stage == "single10_final_phase2_res":
        out_dir = variants_root / "single10_balanced"
        return [
            [
                py, "preprocess.py", str(out_dir),
                "--max_text_len", "96",
            ],
            [
                py, "launcher.py",
                "--config", "experiments_single10_final_phase2_res.json",
                *gpu_args,
                "--dry_run",
            ],
            [
                py, "launcher.py",
                "--config", "experiments_single10_final_phase2_res.json",
                *gpu_args,
            ],
        ]

    raise ValueError(f"Unknown stage: {stage}")


def main():
    args = parse_args()

    for stage in args.stages:
        print("\n" + "=" * 72)
        print(f"[stage] {stage}")
        print("=" * 72)
        for cmd in build_stage_commands(args, stage):
            run_cmd(cmd, dry_run=args.dry_run)

    print("\n[done] research plan completed")


if __name__ == "__main__":
    main()
