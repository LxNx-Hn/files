# RunPod 실행 가이드

현재 기준의 최종 실험은 `data -> single10 -> single10_balanced -> final 9 runs` 흐름입니다.  
2 GPU에서 `CNN / RES` 각각 `early / late / joint` 3개 퓨전과 `image_only / text_only` baseline을 함께 비교합니다.

## 빠른 시작

```bash
cd /workspace
unzip -o tmdb_multimodal_runpod.zip

pip install -r requirements.txt

# TMDb API Key 발급: https://www.themoviedb.org/settings/api
export TMDB_API_KEY=<your_key>

python3 download_mmimdb.py \
  --data_dir ./data \
  --api_key $TMDB_API_KEY \
  --per_genre_overrides_file ./collection_targets_single10_boosted.json

python3 prepare_dataset_variant.py \
  --source_dir ./data \
  --out_dir ./data_variants/single10 \
  --single_genre_only \
  --force

python3 prepare_dataset_variant.py \
  --source_dir ./data_variants/single10 \
  --out_dir ./data_variants/single10_balanced \
  --balance_to_min \
  --force

python3 preprocess.py ./data_variants/single10_balanced --max_text_len 96
python3 launcher.py --config experiments_single10_final_balanced_full.json --dry_run
python3 launcher.py --config experiments_single10_final_balanced_full.json
```

## 최종 구성

- 데이터셋: `single10_balanced`
- 장르 수: `10`
- 실험 수: `9`
- GPU: `2`
- 에폭: `50`
- 비교 모델:
  - `CNN + early / late / joint`
  - `RES + early / late / joint`
  - `CNN + image_only`
  - `RES + image_only`
  - `transformer + text_only`

## 업로드용 압축

```bash
cd ~/Desktop/files
zip -r tmdb_multimodal_runpod.zip \
  run_all.sh launcher.py experiments_single10_final_balanced_full.json \
  multimodal_experiment.py preprocess.py download_mmimdb.py \
  prepare_dataset_variant.py run_research_plan.py \
  collection_targets_single10_boosted.json \
  CLAUDE.md RUNPOD_GUIDE.md README.md requirements.txt
```

## 데이터 준비 확인

```bash
cat data_variants/single10/variant_meta.json
cat data_variants/single10_balanced/variant_meta.json
```

## 실행/모니터링

```bash
python3 launcher.py --config experiments_single10_final_balanced_full.json --dry_run
python3 launcher.py --config experiments_single10_final_balanced_full.json
tail -f results_single10_final_balanced_full/experiment.log
watch -n 5 nvidia-smi
```

## 중단 후 재시작

런처는 이미 `results_single10_final_balanced_full/runs/<experiment>/summary.json` 이 있는 실험은 자동으로 건너뜁니다. 그래서 Pod가 끊기거나 SSH가 끊겨도 같은 명령을 다시 치면 됩니다.

```bash
cd /workspace
python3 launcher.py --config experiments_single10_final_balanced_full.json
```

데이터가 이미 받아진 상태라면 전체 파이프라인을 다시 돌리고 싶을 때:

```bash
cd /workspace
bash run_all.sh
```

기존 결과는 유지한 채 집계만 다시 하고 싶으면:

```bash
python3 launcher.py --config experiments_single10_final_balanced_full.json --aggregate_only
```

## 예상 시간

데이터 수집까지 포함하면 꽤 오래 걸릴 수 있습니다.  
이미 `data`가 있는 상태에서 `single10 -> single10_balanced -> final 9 runs`만 다시 돌리면 보통 몇 시간 단위입니다.

## 실패 시

```bash
python3 launcher.py --config experiments_single10_final_balanced_full.json --aggregate_only
```
