# RunPod 실행 가이드

3 GPU를 동시에 빌려 병렬 비교 실험을 수행하는 기준으로 정리했습니다. 기본 전략은 `GPU 3대 × GPU당 1실험`, `장르당 1000편`입니다.

## 빠른 시작

```bash
unzip tmdb_multimodal_runpod.zip
cd /workspace

pip install -r requirements.txt

# TMDb API Key 발급: https://www.themoviedb.org/settings/api
export TMDB_API_KEY=<your_key>

python3 download_mmimdb.py --data_dir ./data --api_key $TMDB_API_KEY
python3 preprocess.py ./data
python3 launcher.py --dry_run
python3 launcher.py
```

## 권장 설정

`experiments.json` 기본값:

```json
"gpus": [0, 1, 2],
"max_per_gpu": 1,
"common_args": { "n_classes": 15, "batch_size": 32, "epochs": 20, "txt_lr": 2e-5, "img_lr": 1e-4, "fusion_lr": 1e-3, "freeze_bert": true }
```

이 설정이면 7개 실험이 3 GPU에 LPT로 분배되어, fusion 계열과 단일모달 baseline을 함께 비교할 수 있습니다.

## 업로드용 압축

```bash
cd ~/Desktop/files
zip -r tmdb_multimodal_runpod.zip \
  run_all.sh launcher.py experiments.json \
  multimodal_experiment.py preprocess.py download_mmimdb.py \
  CLAUDE.md RUNPOD_GUIDE.md README.md requirements.txt
```

## 데이터 준비 확인

```bash
ls data/
# annotations.json  class_map.json  mmimdb

python3 preprocess.py ./data
```

기본 수집량은 `per_genre=1000`입니다. 더 줄이고 싶으면 `python3 download_mmimdb.py --data_dir ./data --per_genre 300`처럼 덮어쓸 수 있습니다.

## 실행/모니터링

```bash
python3 launcher.py --dry_run
python3 launcher.py
tail -f results/experiment.log
watch -n 5 nvidia-smi
```

## 중단 후 재시작

런처는 이미 `results/runs/<experiment>/summary.json` 이 있는 실험은 자동으로 건너뜁니다. 그래서 Pod가 끊기거나 SSH가 끊겨도 같은 명령을 다시 치면 됩니다.

```bash
cd /workspace
python3 launcher.py
```

데이터가 이미 받아진 상태라면 `annotations.json`이 있어서 수집도 자동 스킵됩니다. 전체 파이프라인을 다시 돌리고 싶으면:

```bash
cd /workspace
bash run_all.sh
```

기존 결과는 유지한 채 집계만 다시 하고 싶으면:

```bash
python3 launcher.py --aggregate_only
```

## 예상 시간

A40 3대 기준으로는 보통 5개 실험 전체가 약 30~45분대에 끝나는 구성이 가장 현실적입니다. 수집 시간은 별도로 10~15분 정도 잡으면 됩니다.

## 실패 시

```bash
python3 launcher.py --aggregate_only
```

특정 실험만 다시 돌리고 싶으면 `experiments.json`의 `experiments` 배열에서 남길 항목만 남기면 됩니다.
