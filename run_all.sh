#!/bin/bash
# ============================================================
#  run_all.sh  —  TMDb 멀티모달 실험 자동화 스크립트
#  포스터+줄거리 / 15개 장르 / 3 GPU 최소 병렬
# ============================================================

set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
PYTHON="${PYTHON:-python3}"

TMDB_API_KEY="${TMDB_API_KEY:-7335b880e3c8007b7beaa2e78dbd2a6c}"
PER_GENRE="${PER_GENRE:-1000}"

MAX_PER_GPU="${MAX_PER_GPU:-1}"
GPU_IDS="${GPU_IDS:-0 1 2}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${RESULTS_DIR}/run_${TIMESTAMP}.log"

mkdir -p "${RESULTS_DIR}"

print_header() {
  echo ""
  echo "╔══════════════════════════════════════════════════════════╗"
  echo "║   TMDb 멀티모달 자동화 실험                            ║"
  echo "║   포스터+줄거리 | 15장르 | 3 GPU 최소 병렬             ║"
  echo "╚══════════════════════════════════════════════════════════╝"
  echo "  DATA_DIR        : ${DATA_DIR}"
  echo "  RESULTS_DIR     : ${RESULTS_DIR}"
  echo "  PER_GENRE       : ${PER_GENRE}"
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
    --per_genre "${PER_GENRE}" \
    2>&1 | tee -a "${LOG_FILE}"
  echo ""
}

run_preprocess_check() {
  echo "[3/5] 전처리 검증..."
  ${PYTHON} preprocess.py "${DATA_DIR}" 2>&1 | tee -a "${LOG_FILE}"
  echo ""
}

run_experiments() {
  echo "[4/5] launcher.py 실행..."
  ${PYTHON} launcher.py \
    --config experiments.json \
    --max_per_gpu "${MAX_PER_GPU}" \
    --gpus ${GPU_IDS} \
    --python "${PYTHON}" \
    2>&1 | tee -a "${LOG_FILE}"
  echo ""
}

print_summary() {
  echo "[5/5] 결과 확인"
  echo "  summary: ${RESULTS_DIR}/summary.json"
  echo "  report : ${RESULTS_DIR}/summary_report.txt"
  echo "  logs   : ${LOG_FILE}"
  echo ""
}

main() {
  print_header
  check_gpu
  install_packages
  download_dataset
  run_preprocess_check
  run_experiments
  print_summary
}

main "$@" 2>&1 | tee -a "${LOG_FILE}"
