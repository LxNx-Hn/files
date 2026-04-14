#!/usr/bin/env python3
"""
download_mmimdb.py
==================
TMDb API에서 영화 포스터 + 줄거리 요약을 수집해
멀티모달 장르 분류용 annotations.json 을 생성한다.
"""

import argparse
import json
import logging
import os
import random
import time
from collections import Counter

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [MMIMDB] %(message)s")
log = logging.getLogger(__name__)

API_BASE = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
DEFAULT_TMDB_API_KEY = "7335b880e3c8007b7beaa2e78dbd2a6c"
TARGET_GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "Horror",
    "Music",
    "Mystery",
    "Romance",
    "Science Fiction",
    "Thriller",
]


class TMDbClient:
    def __init__(self, api_key: str, sleep_sec: float = 0.25):
        self.sleep_sec = sleep_sec
        self.session = requests.Session()
        self.session.params = {"api_key": api_key, "language": "en-US"}

    def get(self, path: str, **params):
        resp = self.session.get(f"{API_BASE}{path}", params=params, timeout=30)
        resp.raise_for_status()
        time.sleep(self.sleep_sec)
        return resp.json()

    def download_file(self, url: str, dest: str):
        resp = self.session.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def build_parser():
    parser = argparse.ArgumentParser(description="TMDb 포스터+줄거리 데이터 수집")
    parser.add_argument("--data_dir", default="./data", help="출력 데이터 디렉토리")
    parser.add_argument("--api_key", default=None, help="TMDb API Key")
    parser.add_argument("--per_genre", type=int, default=1000, help="장르당 최대 수집 편수")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    return parser


def get_api_key(cli_value):
    api_key = cli_value or os.environ.get("TMDB_API_KEY") or DEFAULT_TMDB_API_KEY
    if not api_key:
        raise ValueError("TMDb API Key가 필요합니다. --api_key 또는 TMDB_API_KEY를 설정하세요.")
    return api_key


def fetch_genre_map(client: TMDbClient) -> dict:
    data = client.get("/genre/movie/list")
    genre_map = {row["name"]: row["id"] for row in data["genres"]}
    missing = [name for name in TARGET_GENRES if name not in genre_map]
    if missing:
        raise RuntimeError(f"TMDb 장르 목록에서 누락된 항목: {missing}")
    return genre_map


def discover_movies_for_genre(client: TMDbClient, genre_id: int, per_genre: int, seen_ids: set) -> list:
    collected = []
    page = 1

    while len(collected) < per_genre:
        payload = client.get(
            "/discover/movie",
            with_genres=genre_id,
            sort_by="popularity.desc",
            include_adult="false",
            include_video="false",
            page=page,
            vote_count_gte=10,
        )
        results = payload.get("results", [])
        if not results:
            break

        for movie in results:
            if len(collected) >= per_genre:
                break
            movie_id = movie["id"]
            genre_ids = movie.get("genre_ids", [])
            if not genre_ids or genre_ids[0] != genre_id:
                continue
            if movie_id in seen_ids:
                continue
            if not movie.get("overview") or not movie.get("poster_path"):
                continue
            seen_ids.add(movie_id)
            collected.append(movie)

        total_pages = payload.get("total_pages", page)
        if page >= total_pages:
            break
        page += 1

    return collected


def save_posters(client: TMDbClient, movies_by_genre: dict, image_dir: str) -> list:
    annotations = []
    for genre_name, info in movies_by_genre.items():
        label = info["label"]
        movies = info["movies"]
        for movie in tqdm(movies, desc=f"{genre_name:>16}", unit="movie"):
            movie_id = movie["id"]
            rel_path = f"mmimdb/images/{movie_id}.jpg"
            abs_path = os.path.join(image_dir, f"{movie_id}.jpg")
            if not os.path.exists(abs_path):
                client.download_file(f"{IMAGE_BASE}{movie['poster_path']}", abs_path)
            top3_labels = build_top3_labels(movie.get("genre_ids", []))
            if not top3_labels:
                top3_labels = [label]
            annotations.append({
                "image": rel_path,
                "text": movie["overview"].strip(),
                "label": label,
                "labels_top3": top3_labels,
            })
    return annotations


def build_top3_labels(genre_ids: list) -> list:
    genre_name_to_idx = {name: idx for idx, name in enumerate(TARGET_GENRES)}
    genre_id_to_name = GENRE_ID_TO_NAME
    labels = []
    for genre_id in genre_ids:
        genre_name = genre_id_to_name.get(genre_id)
        if genre_name in genre_name_to_idx:
            labels.append(genre_name_to_idx[genre_name])
        if len(labels) == 3:
            break
    return labels[:3]


GENRE_ID_TO_NAME = {}


def write_json(path: str, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main():
    args = build_parser().parse_args()
    random.seed(args.seed)

    client = TMDbClient(api_key=get_api_key(args.api_key))
    os.makedirs(args.data_dir, exist_ok=True)
    mmimdb_dir = os.path.join(args.data_dir, "mmimdb")
    image_dir = os.path.join(mmimdb_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    log.info("TMDb 장르 목록 조회 중...")
    genre_map = fetch_genre_map(client)
    global GENRE_ID_TO_NAME
    GENRE_ID_TO_NAME = {genre_id: genre_name for genre_name, genre_id in genre_map.items()}
    class_map = {name: idx for idx, name in enumerate(TARGET_GENRES)}
    movies_by_genre = {}
    seen_ids = set()

    for genre_name in TARGET_GENRES:
        genre_id = genre_map[genre_name]
        log.info(f"{genre_name} 수집 시작 (genre_id={genre_id})")
        movies = discover_movies_for_genre(client, genre_id, args.per_genre, seen_ids)
        movies_by_genre[genre_name] = {"label": class_map[genre_name], "movies": movies}
        log.info(f"{genre_name} 수집 완료: {len(movies)}편")

    annotations = save_posters(client, movies_by_genre, image_dir)
    random.shuffle(annotations)

    write_json(os.path.join(args.data_dir, "annotations.json"), annotations)
    write_json(os.path.join(args.data_dir, "class_map.json"), class_map)

    counts = Counter({genre_name: len(info["movies"]) for genre_name, info in movies_by_genre.items()})
    min_count = min(counts.values()) if counts else 0
    max_count = max(counts.values()) if counts else 0
    imbalance = (max_count / min_count) if min_count else 0.0

    log.info("")
    log.info("수집 완료")
    for genre_name in TARGET_GENRES:
        log.info(f"  {genre_name:<16} {counts[genre_name]:>4}")
    log.info(f"총 샘플 수: {len(annotations)}")
    log.info(f"불균형 비율(max/min): {imbalance:.2f}")
    log.info(f"annotations.json → {os.path.join(args.data_dir, 'annotations.json')}")
    log.info(f"class_map.json   → {os.path.join(args.data_dir, 'class_map.json')}")


if __name__ == "__main__":
    main()
