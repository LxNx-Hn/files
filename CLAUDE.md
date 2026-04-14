# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## 프로젝트 개요

TMDb 기반 영화 포스터+줄거리 멀티모달 장르 분류 실험 프로젝트다. 기본 실험은 pretrained ResNet50 이미지 인코더와 Transformer 텍스트 인코더를 고정하고 `image_only`, `text_only`, `early`, `late`, `weighted_late`, `gated`, `cross_attention` 구성을 비교한다. 기본 실행 환경은 RunPod의 3 GPU 동시 사용 + GPU당 1실험이다.

## 기본 실행

```bash
pip install -r requirements.txt
export TMDB_API_KEY=<your_key>
python3 download_mmimdb.py --data_dir ./data --api_key $TMDB_API_KEY
python3 preprocess.py ./data
python3 launcher.py
```

## 전체 파이프라인

```bash
bash run_all.sh
```

## 주요 환경변수

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `TMDB_API_KEY` | (없음) | TMDb API 키 |
| `PER_GENRE` | `1000` | 장르당 최대 수집 편수 |
| `DATA_DIR` | `./data` | 데이터 출력 경로 |
| `RESULTS_DIR` | `./results` | 결과 경로 |
| `GPU_IDS` | `0 1 2` | 사용할 GPU 목록 |
| `MAX_PER_GPU` | `1` | GPU당 동시 실험 수 |

## 파일 역할

| 파일 | 역할 |
|---|---|
| `download_mmimdb.py` | TMDb 포스터/줄거리 수집 및 `annotations.json` 생성 |
| `preprocess.py` | 공통 전처리 및 DataLoader 구성 |
| `multimodal_experiment.py` | 모델 정의, 학습, 평가, 시각화 |
| `launcher.py` | LPT 기반 다중 GPU 실험 분배 |
| `experiments.json` | 3 GPU 기본 실험 설정 |
| `run_all.sh` | 설치부터 실행까지 자동화 |

## 데이터 형식

`annotations.json`:

```json
[
  {
    "image": "mmimdb/images/123.jpg",
    "text": "Movie overview...",
    "label": 0
  }
]
```
