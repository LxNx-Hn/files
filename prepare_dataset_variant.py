#!/usr/bin/env python3
"""
prepare_dataset_variant.py
==========================
기존 TMDb 데이터셋에서 실험용 variant 데이터셋을 생성한다.
원본 이미지는 재사용하고, annotations/class_map만 별도 디렉토리에 작성한다.
"""

import argparse
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="실험용 데이터셋 variant 생성")
    parser.add_argument("--source_dir", default="./data", help="원본 데이터셋 루트")
    parser.add_argument("--out_dir", required=True, help="variant 출력 디렉토리")
    parser.add_argument("--include", nargs="+", default=None, help="포함할 장르 목록")
    parser.add_argument("--exclude", nargs="+", default=None, help="제외할 장르 목록")
    parser.add_argument("--single_genre_only", action="store_true",
                        help="단일장르 영화만 유지")
    parser.add_argument("--balance_to_min", action="store_true",
                        help="필터링 후 클래스별 최소 개수에 맞춰 균형 샘플링")
    parser.add_argument("--balance_count", type=int, default=None,
                        help="클래스당 고정 개수로 균형 샘플링 (모든 클래스가 해당 개수 이상이어야 함)")
    parser.add_argument("--seed", type=int, default=42,
                        help="균형 샘플링용 랜덤 시드")
    parser.add_argument("--force", action="store_true",
                        help="out_dir가 있으면 삭제 후 재생성")
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    if args.balance_to_min and args.balance_count is not None:
        raise ValueError("--balance_to_min 과 --balance_count 는 동시에 사용할 수 없습니다.")
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    ann_path = source_dir / "annotations.json"
    class_map_path = source_dir / "class_map.json"

    annotations = load_json(ann_path)
    class_map = load_json(class_map_path)
    ordered_names = [name for name, _ in sorted(class_map.items(), key=lambda x: x[1])]

    include = list(args.include) if args.include else list(ordered_names)
    exclude = set(args.exclude or [])
    target_names = [name for name in include if name not in exclude]

    missing = [name for name in target_names if name not in class_map]
    if missing:
        raise ValueError(f"class_map에 없는 장르: {missing}")
    if not target_names:
        raise ValueError("최소 1개 이상의 장르가 필요합니다.")

    if out_dir.exists() and args.force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    old_to_name = {idx: name for name, idx in class_map.items()}
    new_class_map = {name: idx for idx, name in enumerate(target_names)}
    old_to_new = {
        old_idx: new_class_map[name]
        for old_idx, name in old_to_name.items()
        if name in new_class_map
    }

    filtered = []
    for item in annotations:
        top3 = [int(x) for x in item.get("labels_top3", [item["label"]])[:3]]
        if not top3:
            top3 = [int(item["label"])]

        kept_labels = [old_to_new[idx] for idx in top3 if idx in old_to_new]
        if not kept_labels:
            continue

        dedup = []
        seen = set()
        for idx in kept_labels:
            if idx not in seen:
                dedup.append(idx)
                seen.add(idx)

        if args.single_genre_only and len(dedup) != 1:
            continue

        filtered.append({
            "image": item["image"],
            "text": item["text"],
            "label": dedup[0],
            "labels_top3": dedup[:3],
        })

    pre_balance_counts = Counter(item["label"] for item in filtered)
    balance_mode = "none"
    target_count = None

    if args.balance_to_min or args.balance_count is not None:
        if not filtered:
            raise ValueError("필터링 결과가 비어 있어 균형 샘플링을 수행할 수 없습니다.")

        if args.balance_to_min:
            target_count = min(pre_balance_counts.values())
            balance_mode = "min"
        else:
            target_count = args.balance_count
            balance_mode = "fixed"
            lacking = [
                target_names[idx]
                for idx in range(len(target_names))
                if pre_balance_counts.get(idx, 0) < target_count
            ]
            if lacking:
                raise ValueError(
                    f"balance_count={target_count} 보다 샘플 수가 적은 클래스가 있습니다: {lacking}"
                )

        rng = random.Random(args.seed)
        per_label = {idx: [] for idx in range(len(target_names))}
        for item in filtered:
            per_label[item["label"]].append(item)

        balanced = []
        for idx in range(len(target_names)):
            items = per_label[idx]
            rng.shuffle(items)
            balanced.extend(items[:target_count])
        rng.shuffle(balanced)
        filtered = balanced

    save_json(out_dir / "annotations.json", filtered)
    save_json(out_dir / "class_map.json", new_class_map)

    src_mmimdb = source_dir / "mmimdb"
    dst_mmimdb = out_dir / "mmimdb"
    if dst_mmimdb.exists() or dst_mmimdb.is_symlink():
        if dst_mmimdb.is_symlink() or dst_mmimdb.is_file():
            dst_mmimdb.unlink()
        else:
            shutil.rmtree(dst_mmimdb)
    try:
        os.symlink(src_mmimdb.resolve(), dst_mmimdb)
    except OSError:
        shutil.copytree(src_mmimdb, dst_mmimdb)

    counts = Counter(item["label"] for item in filtered)
    metadata = {
        "source_dir": str(source_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "target_names": target_names,
        "single_genre_only": args.single_genre_only,
        "balance_mode": balance_mode,
        "balance_target_count": target_count,
        "n_classes": len(target_names),
        "n_samples": len(filtered),
        "pre_balance_label_counts": {target_names[idx]: pre_balance_counts.get(idx, 0) for idx in range(len(target_names))},
        "label_counts": {target_names[idx]: counts[idx] for idx in range(len(target_names))},
    }
    save_json(out_dir / "variant_meta.json", metadata)

    print(f"[done] out_dir={out_dir}")
    print(f"[done] n_classes={len(target_names)}")
    print(f"[done] n_samples={len(filtered)}")
    for idx, name in enumerate(target_names):
        print(f"  {name:<18} {counts[idx]}")


if __name__ == "__main__":
    main()
