# TMDb 멀티모달 영화 장르 분류

영화 포스터 이미지와 줄거리 요약 텍스트를 함께 사용해 15개 장르를 분류하는 실험 코드입니다. 기본 비교 대상은 `image_only`, `text_only`, `early`, `late`, `weighted_late`, `gated`, `cross_attention` 구성이고, 실행 기본값은 3 GPU 동시 사용 + GPU당 1개 실험, 장르당 최대 1000편 수집입니다.

## 실행

```bash
pip install -r requirements.txt
export TMDB_API_KEY=<your_key>
python3 download_mmimdb.py --data_dir ./data --api_key $TMDB_API_KEY
python3 preprocess.py ./data
python3 launcher.py
```

결과는 `results/summary_report.txt`에 저장됩니다.

## 기본 설정

| 항목 | 기본값 | 설명 |
|---|---|---|
| `gpus` | `[0, 1, 2]` | 3 GPU 동시 사용 |
| `max_per_gpu` | `1` | GPU당 1실험만 실행 |
| `n_classes` | `15` | TMDb 장르 수 |
| `batch_size` | `32` | 전체 배치 크기 |
| `epochs` | `20` | 학습 에폭 수 |
| `txt_lr` | `2e-5` | Transformer 텍스트 인코더 학습률 |
| `img_lr` | `1e-4` | CNN 이미지 인코더 학습률 |
| `fusion_lr` | `1e-3` | 퓨전 및 분류기 학습률 |
| `freeze_bert` | `true` | 텍스트 인코더 동결 |

## 파일 구조

```text
├── launcher.py
├── experiments.json
├── multimodal_experiment.py
├── preprocess.py
├── download_mmimdb.py
├── run_all.sh
└── requirements.txt
```

## 빠른 확인

```bash
python3 launcher.py --dry_run
python3 launcher.py --aggregate_only
```
