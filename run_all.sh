#!/bin/bash
# ============================================================
#  run_all.sh  —  최종 2단계 실행 스크립트
#  data -> single10 -> single10_balanced -> phase1(CNN) -> phase2(RES)
# ============================================================

set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
VARIANTS_ROOT="${VARIANTS_ROOT:-./data_variants}"
RESULTS_DIR="${RESULTS_DIR:-./results_single10_final}"
PYTHON="${PYTHON:-python3}"

TMDB_API_KEY="${TMDB_API_KEY:-7335b880e3c8007b7beaa2e78dbd2a6c}"
TARGETS_FILE="${TARGETS_FILE:-./collection_targets_single10_boosted.json}"

MAX_PER_GPU="${MAX_PER_GPU:-1}"
GPU_IDS="${GPU_IDS:-0 1}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${RESULTS_DIR}/run_${TIMESTAMP}.log"

mkdir -p "${RESULTS_DIR}"

print_header() {
  echo ""
  echo "╔══════════════════════════════════════════════════════════╗"
  echo "║   TMDb 멀티모달 최종 실험                              ║"
  echo "║   single10 -> balanced -> phase1(CNN) -> phase2(RES) ║"
  echo "╚══════════════════════════════════════════════════════════╝"
  echo "  DATA_DIR        : ${DATA_DIR}"
  echo "  VARIANTS_ROOT   : ${VARIANTS_ROOT}"
  echo "  RESULTS_DIR     : ${RESULTS_DIR}"
  echo "  TARGETS_FILE    : ${TARGETS_FILE}"
  echo "  GPU_IDS         : ${GPU_IDS}"
  echo "  MAX_PER_GPU     : ${MAX_PER_GPU}"
  echo "  LOG_FILE        : ${LOG_FILE}"
  echo ""
}

check_gpu() {
  echo "[0/5] GPU 환경 확인..."
  if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free \
      --format=csv,noheader,nounits | \
      awk -F',' '{printf "  GPU[%s] %-30s | Total: %6s MB | Free: %6s MB\n",$1,$2,$3,$4}'
  else
    echo "  ⚠ GPU 감지 안됨"
  fi
  echo ""
}

install_packages() {
  echo "[1/5] 패키지 설치..."
  pip install --quiet --upgrade pip
  pip install --quiet -r requirements.txt
  echo "  ✓ 패키지 준비 완료"
  echo ""
}

download_dataset() {
  echo "[2/5] TMDb 데이터 수집..."
  if [ -f "${DATA_DIR}/annotations.json" ]; then
    echo "  ✓ annotations.json 이미 존재 — 수집 스킵"
    echo ""
    return
  fi

  ${PYTHON} download_mmimdb.py \
    --data_dir "${DATA_DIR}" \
    --api_key "${TMDB_API_KEY}" \
    --per_genre_overrides_file "${TARGETS_FILE}" \
    2>&1 | tee -a "${LOG_FILE}"
  echo ""
}

prepare_variants() {
  echo "[3/5] variant 준비..."
  ${PYTHON} prepare_dataset_variant.py \
    --source_dir "${DATA_DIR}" \
    --out_dir "${VARIANTS_ROOT}/single10" \
    --single_genre_only \
    --force 2>&1 | tee -a "${LOG_FILE}"

  ${PYTHON} prepare_dataset_variant.py \
    --source_dir "${VARIANTS_ROOT}/single10" \
    --out_dir "${VARIANTS_ROOT}/single10_balanced" \
    --balance_to_min \
    --force 2>&1 | tee -a "${LOG_FILE}"

  ${PYTHON} preprocess.py "${VARIANTS_ROOT}/single10_balanced" --max_text_len 96 2>&1 | tee -a "${LOG_FILE}"
  echo ""
}

run_experiments() {
  echo "[4/5] launcher.py 실행..."
  ${PYTHON} launcher.py \
    --config experiments_single10_final_phase1_cnn.json \
    --max_per_gpu "${MAX_PER_GPU}" \
    --gpus ${GPU_IDS} \
    --python "${PYTHON}" \
    2>&1 | tee -a "${LOG_FILE}"
  ${PYTHON} launcher.py \
    --config experiments_single10_final_phase2_res.json \
    --max_per_gpu "${MAX_PER_GPU}" \
    --gpus ${GPU_IDS} \
    --python "${PYTHON}" \
    2>&1 | tee -a "${LOG_FILE}"
  echo ""
}

print_summary() {
  echo "[5/5] 결과 확인"
  echo "  phase1 : ./results_single10_final_phase1_cnn/summary_report.txt"
  echo "  phase2 : ./results_single10_final_phase2_res/summary_report.txt"
  echo "  logs   : ${LOG_FILE}"
  echo ""
}

main() {
  print_header
  check_gpu
  install_packages
  download_dataset
  prepare_variants
  run_experiments
  print_summary
}

main "$@" 2>&1 | tee -a "${LOG_FILE}"
