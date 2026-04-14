#!/usr/bin/env python3
"""
launcher.py — Multi-GPU 병렬 실험 런처 (LPT 샤드 밸런싱)
==========================================================
experiments.json을 읽어 GPU당 N개 실험을 동시에 실행합니다.

샤드 밸런싱 전략 (LPT — Longest Processing Time First):
  - experiments.json 의 estimated_minutes 기준 내림차순 정렬
  - 각 실험을 현재 누적 부하가 가장 적은 GPU에 사전 할당
  - 결과: 두 GPU의 예상 완료 시간이 최대한 동일해짐

동작 흐름:
  1. LPT로 각 GPU에 실험 목록 사전 할당
  2. GPU별 스레드 풀(max_per_gpu 워커) 생성
  3. 각 워커: CUDA_VISIBLE_DEVICES=N 환경변수로 subprocess 실행
  4. 전체 완료 후 results/runs/*/summary.json 집계 → 최종 랭킹

사용법:
  python launcher.py                            # 기본 실행
  python launcher.py --max_per_gpu 2            # GPU당 2개 동시
  python launcher.py --gpus 0 1                 # 사용 GPU 지정
  python launcher.py --dry_run                  # 명령어/할당 계획만 출력
  python launcher.py --config custom.json       # 다른 설정 파일
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RANK_METRIC = "f1_weighted"


# ─── LPT 샤드 밸런싱 ─────────────────────────────────────────────
def lpt_assign(experiments: list, gpus: list) -> dict:
    """
    Longest Processing Time First 알고리즘으로 GPU별 실험 할당.
    estimated_minutes 내림차순 정렬 후, 현재 부하가 가장 적은 GPU에 순서대로 배정.
    반환: {gpu_id: [exp, ...], ...}
    """
    sorted_exps = sorted(
        experiments,
        key=lambda e: e.get("estimated_minutes", 30),
        reverse=True,
    )
    gpu_load = {g: 0.0 for g in gpus}
    gpu_queues = {g: [] for g in gpus}

    for exp in sorted_exps:
        target = min(gpu_load, key=gpu_load.get)
        gpu_queues[target].append(exp)
        gpu_load[target] += exp.get("estimated_minutes", 30)

    return gpu_queues, gpu_load


def print_plan(gpu_queues: dict, gpu_load: dict) -> None:
    """할당 계획 출력"""
    print("\n  ── LPT 샤드 할당 계획 ──────────────────────────────────")
    for g in sorted(gpu_queues):
        exps = gpu_queues[g]
        names = [e["name"] for e in exps]
        total = gpu_load[g]
        print(f"  GPU {g}  ({len(exps)}개 실험 / 예상 총 {total:.0f}분)")
        for i, e in enumerate(exps):
            mins = e.get("estimated_minutes", "?")
            print(f"    {'└' if i == len(exps)-1 else '├'} [{mins:>3}분] {e['name']}")
    max_load = max(gpu_load.values())
    min_load = min(gpu_load.values())
    imbalance = (max_load - min_load) / max_load * 100 if max_load else 0
    print(f"\n  예상 불균형 : {imbalance:.1f}%  (낮을수록 좋음)")
    print("  ─────────────────────────────────────────────────────\n")


# ─── 명령어 구성 ──────────────────────────────────────────────────
def build_cmd(python: str, exp: dict, common_args: dict,
              exp_results_dir: str) -> list:
    cmd = [
        python, "multimodal_experiment.py",
        "--img_encoders", exp["img_encoder"],
        "--txt_encoders", exp["txt_encoder"],
        "--fusions",      exp["fusion"],
        "--cnn_layers",   str(exp["cnn_layers"]),
        "--results_dir",  exp_results_dir,
    ]
    for k, v in common_args.items():
        if k == "results_dir":
            continue
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(v)])
    return cmd


# ─── 단일 실험 실행 ───────────────────────────────────────────────
def run_single(exp: dict, common_args: dict, gpu_id: int,
               results_dir: str, python: str) -> dict:
    safe_name = (exp["name"]
                 .replace("+", "_")
                 .replace("(", "L")
                 .replace(")", ""))
    exp_results_dir = str(Path(results_dir) / "runs" / safe_name)
    log_dir = Path(results_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{safe_name}.log"
    summary_path = Path(exp_results_dir) / "summary.json"

    if summary_path.exists():
        print(f"  [GPU {gpu_id}] SKIP {exp['name']}  (existing summary.json)", flush=True)
        return {
            "name":            exp["name"],
            "exp_results_dir": exp_results_dir,
            "returncode":      0,
            "time_sec":        0.0,
        }

    cmd = build_cmd(python, exp, common_args, exp_results_dir)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    t0 = time.time()
    print(f"  [GPU {gpu_id}] ▶  {exp['name']}", flush=True)
    with open(log_path, "w") as lf:
        ret = subprocess.run(
            cmd, env=env, stdout=lf, stderr=subprocess.STDOUT
        ).returncode
    elapsed = time.time() - t0
    icon = "✓" if ret == 0 else "✗"
    print(f"  [GPU {gpu_id}] {icon}  {exp['name']}  ({elapsed / 60:.1f}분)", flush=True)

    return {
        "name":            exp["name"],
        "exp_results_dir": exp_results_dir,
        "returncode":      ret,
        "time_sec":        elapsed,
    }


# ─── GPU별 샤드 실행 ─────────────────────────────────────────────
def run_gpu_shard(gpu_id: int, experiments: list, common_args: dict,
                  results_dir: str, python: str, max_concurrent: int) -> list:
    """단일 GPU에서 할당된 실험들을 max_concurrent개씩 병렬 실행"""
    results = []
    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = [
            pool.submit(run_single, exp, common_args, gpu_id, results_dir, python)
            for exp in experiments
        ]
        for fut in as_completed(futures):
            results.append(fut.result())
    return results


# ─── 결과 집계 ────────────────────────────────────────────────────
def aggregate(run_dirs: list, final_dir: str) -> None:
    all_results = []
    for d in run_dirs:
        sj = Path(d) / "summary.json"
        if sj.exists():
            with open(sj, encoding="utf-8") as f:
                s = json.load(f)
            all_results.extend(s.get("all_results", []))
        else:
            print(f"  ⚠  {sj} 없음 — 스킵")

    if not all_results:
        print("  ⚠  집계할 결과 없음")
        return

    all_results.sort(key=lambda x: x[RANK_METRIC], reverse=True)
    best = all_results[0]

    summary = {
        "rank_metric": RANK_METRIC,
        "best_config": best["config_name"],
        "best_scores": {
            "accuracy":        best["accuracy"],
            "precision_macro": best["precision_macro"],
            "recall_macro":    best["recall_macro"],
            "f1_macro":        best["f1_macro"],
            "f1_weighted":     best["f1_weighted"],
            "relaxed_top2_accuracy": best.get("relaxed_top2_accuracy", 0.0),
            "relaxed_top3_accuracy": best.get("relaxed_top3_accuracy", 0.0),
        },
        "all_results": all_results,
    }

    Path(final_dir).mkdir(parents=True, exist_ok=True)
    out_json = Path(final_dir) / "summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  Summary JSON   : {out_json}")

    out_txt = Path(final_dir) / "summary_report.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=" * 74 + "\n")
        f.write("  Multimodal Image+Text Experiment Summary (launcher aggregate)\n")
        f.write("=" * 74 + "\n\n")
        f.write(f"  Best Config  : {best['config_name']}\n")
        f.write(f"  F1 Weighted : {best['f1_weighted']:.4f}\n")
        f.write(f"  Accuracy    : {best['accuracy']:.4f}\n")
        f.write(f"  Relaxed@2   : {best.get('relaxed_top2_accuracy', 0.0):.4f}\n")
        f.write(f"  Relaxed@3   : {best.get('relaxed_top3_accuracy', 0.0):.4f}\n")
        f.write(f"  F1 Macro    : {best['f1_macro']:.4f}\n\n")
        hdr = (f"  {'Rank':<4} {'Config':48} "
               f"{'Acc':>6} {'F1_mac':>7} {'F1_wgt':>7} {'Rlx@2':>7} {'Rlx@3':>7} {'Time(m)':>8}")
        f.write(hdr + "\n")
        f.write("  " + "-" * (len(hdr) - 2) + "\n")
        for i, r in enumerate(all_results, 1):
            mins = r.get("train_time_sec", 0) / 60
            f.write(
                f"  {i:<4} {r['config_name']:48} "
                f"{r['accuracy']:>6.4f} {r['f1_macro']:>7.4f} "
                f"{r['f1_weighted']:>7.4f} {r.get('relaxed_top2_accuracy', 0.0):>7.4f} "
                f"{r.get('relaxed_top3_accuracy', 0.0):>7.4f} {mins:>8.1f}\n"
            )
    print(f"  Summary Report : {out_txt}")
    _plot_comparison(all_results, final_dir)


def _plot_comparison(all_results: list, results_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    names = [r["config_name"] for r in all_results]
    accs  = [r["accuracy"]    for r in all_results]
    f1m   = [r["f1_macro"]    for r in all_results]
    f1w   = [r["f1_weighted"] for r in all_results]

    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(14, len(names) * 0.95), 6))
    ax.bar(x - w, accs, w, label="Accuracy",    color="#4e79a7")
    ax.bar(x,     f1m,  w, label="F1 Macro",    color="#f28e2b")
    ax.bar(x + w, f1w,  w, label="F1 Weighted", color="#e15759")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Experiment Comparison (sorted by F1 Weighted)")
    plt.tight_layout()
    plots_dir = Path(results_dir) / "plots"
    plots_dir.mkdir(exist_ok=True)
    out = plots_dir / "f1_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Comparison Plot : {out}")


# ─── 별도 집계 명령 (--aggregate_only) ───────────────────────────
def aggregate_only(results_dir: str) -> None:
    """이미 완료된 results/runs/*/summary.json 를 재집계"""
    import glob
    run_dirs = [
        str(Path(p).parent)
        for p in glob.glob(str(Path(results_dir) / "runs" / "**" / "summary.json"),
                           recursive=True)
    ]
    if not run_dirs:
        print(f"  ⚠  {results_dir}/runs/ 아래에 summary.json 없음")
        return
    print(f"  {len(run_dirs)}개 실험 결과 발견 — 집계 중...")
    aggregate(run_dirs, results_dir)


# ─── 메인 ─────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-GPU 병렬 실험 런처 (LPT 샤드 밸런싱)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config",         default="experiments.json")
    parser.add_argument("--max_per_gpu",    type=int, default=None,
                        help="GPU당 최대 동시 실험 수 (JSON 설정 override)")
    parser.add_argument("--gpus",           nargs="+", type=int, default=None,
                        help="사용 GPU ID 목록 (JSON 설정 override)")
    parser.add_argument("--python",         default="python3")
    parser.add_argument("--dry_run",        action="store_true",
                        help="할당 계획과 명령어만 출력, 실제 실행 안 함")
    parser.add_argument("--aggregate_only", action="store_true",
                        help="실험 없이 기존 results/runs/ 재집계만 수행")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)

    gpus        = args.gpus        or cfg.get("gpus",        [0])
    max_per_gpu = args.max_per_gpu or cfg.get("max_per_gpu", 2)
    results_dir = cfg.get("results_dir", "./results")
    common_args = cfg.get("common_args", {})
    experiments = cfg["experiments"]

    # ── 집계만 수행 ──
    if args.aggregate_only:
        aggregate_only(results_dir)
        return

    # ── LPT 샤드 할당 ──
    gpu_queues, gpu_load = lpt_assign(experiments, gpus)

    print("\n" + "=" * 62)
    print(f"  GPU 목록      : {gpus}")
    print(f"  GPU당 최대    : {max_per_gpu}개 동시")
    print(f"  총 슬롯       : {len(gpus) * max_per_gpu}개")
    print(f"  총 실험 수    : {len(experiments)}개")
    print(f"  결과 디렉토리 : {results_dir}")
    print("=" * 62)
    print_plan(gpu_queues, gpu_load)

    if args.dry_run:
        for g in sorted(gpu_queues):
            print(f"\n  ── GPU {g} 명령어 ──")
            for exp in gpu_queues[g]:
                safe = exp["name"].replace("+", "_").replace("(", "L").replace(")", "")
                exp_dir = str(Path(results_dir) / "runs" / safe)
                cmd = build_cmd(args.python, exp, common_args, exp_dir)
                print(f"  {' '.join(cmd)}")
        return

    # ── GPU별 샤드를 상위 스레드 풀에서 병렬 실행 ──
    all_run_dirs: list = []

    with ThreadPoolExecutor(max_workers=len(gpus)) as shard_pool:
        shard_futures = {
            shard_pool.submit(
                run_gpu_shard,
                g, gpu_queues[g], common_args, results_dir, args.python, max_per_gpu
            ): g
            for g in gpus
        }
        for fut in as_completed(shard_futures):
            g = shard_futures[fut]
            shard_results = fut.result()
            for meta in shard_results:
                all_run_dirs.append(meta["exp_results_dir"])
            ok  = sum(1 for m in shard_results if m["returncode"] == 0)
            err = sum(1 for m in shard_results if m["returncode"] != 0)
            print(f"\n  [GPU {g}] 샤드 완료 — 성공 {ok} / 실패 {err}")

    print("\n" + "=" * 62)
    print("  모든 샤드 완료 — 결과 집계 중...")
    aggregate(all_run_dirs, results_dir)
    print("=" * 62)
    print(f"\n  완료!  결과 디렉토리: {results_dir}\n")


if __name__ == "__main__":
    main()
