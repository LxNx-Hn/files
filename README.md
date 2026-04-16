# TMDb 멀티모달 영화 장르 분류

영화 포스터 이미지와 줄거리 요약 텍스트를 함께 사용해 10개 장르를 분류하는 실험 코드입니다. 현재 메인 실험은 `single10 -> single10_balanced`를 만든 뒤, `CNN / RES` 각각에 대해 `early / late / joint` 3개 퓨전과 `image_only / text_only` baseline을 함께 비교하는 최종 런 기준입니다.

## 현재 메인 실험

- 원본 데이터: `./data`
- 단일장르 variant: `./data_variants/single10`
- 균형 variant: `./data_variants/single10_balanced`
- 최종 설정 파일: `experiments_single10_final_balanced_full.json`
- 결과 폴더: `results_single10_final_balanced_full`

총 실험 수는 `9런`입니다.

- `CNN + early / late / joint`
- `RES + early / late / joint`
- `CNN + image_only`
- `RES + image_only`
- `transformer + text_only`

## 실행

```bash
pip install -r requirements.txt
export TMDB_API_KEY=<your_key>
python3 download_mmimdb.py --data_dir ./data --api_key $TMDB_API_KEY
python3 prepare_dataset_variant.py --source_dir ./data --out_dir ./data_variants/single10 --single_genre_only --force
python3 prepare_dataset_variant.py --source_dir ./data_variants/single10 --out_dir ./data_variants/single10_balanced --balance_to_min --force
python3 preprocess.py ./data_variants/single10_balanced --max_text_len 96
python3 launcher.py --config experiments_single10_final_balanced_full.json
```

결과는 `results_single10_final_balanced_full/summary_report.txt`에 저장됩니다.

## 기본 설정

| 항목 | 기본값 | 설명 |
|---|---|---|
| `gpus` | `[0, 1]` | 2 GPU 동시 사용 |
| `max_per_gpu` | `1` | GPU당 1실험만 실행 |
| `n_classes` | `10` | 기본 장르 수 |
| `batch_size` | `32` | 전체 배치 크기 |
| `epochs` | `50` | 학습 에폭 수 |
| `txt_lr` | `2e-5` | Transformer 텍스트 인코더 학습률 |
| `img_lr` | `5e-4` | 이미지 인코더 학습률 |
| `fusion_lr` | `1e-3` | 퓨전 및 분류기 학습률 |
| `freeze_bert` | `false` | 최종 런은 fine-tuning 사용 |

## 파일 구조

```text
├── launcher.py
├── experiments_single10_final_balanced_full.json
├── multimodal_experiment.py
├── preprocess.py
├── download_mmimdb.py
├── prepare_dataset_variant.py
├── run_all.sh
└── requirements.txt
```

## 빠른 확인

```bash
python3 launcher.py --config experiments_single10_final_balanced_full.json --dry_run
python3 launcher.py --config experiments_single10_final_balanced_full.json --aggregate_only
```

## 기본 10개 장르

- `Action`
- `Animation`
- `Comedy`
- `Crime`
- `Documentary`
- `Family`
- `Horror`
- `Music`
- `Romance`
- `Science Fiction`
