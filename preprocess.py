"""
preprocess.py
=============
공통 전처리 모듈 - 이미지 + 텍스트 멀티모달 데이터셋 준비
모든 실험 레이어에서 동일하게 사용됨
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
from PIL import Image
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [PREPROCESS] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
IMAGE_SIZE = 224
MAX_TEXT_LEN = 64
TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # 경량 임베딩 토크나이저
BATCH_SIZE = 32
NUM_WORKERS = 4
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
# TEST_RATIO = 0.15


# ─────────────────────────────────────────────
# 이미지 전처리 파이프라인
# ─────────────────────────────────────────────
IMAGE_TRANSFORM_TRAIN = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

IMAGE_TRANSFORM_EVAL = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# 멀티모달 데이터셋
# ─────────────────────────────────────────────
class MultimodalDataset(Dataset):
    """
    data_dir 구조 예시:
        data/
          annotations.json   ← [{"image": "img/001.jpg", "text": "...", "label": 0}, ...]
          img/
    """

    def __init__(self, data_dir: str, split: str = "train",
                 tokenizer_name: str = TOKENIZER_NAME):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.transform = IMAGE_TRANSFORM_TRAIN if split == "train" else IMAGE_TRANSFORM_EVAL

        ann_path = os.path.join(data_dir, "annotations.json")
        with open(ann_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        log.info(f"[{split}] 샘플 수: {len(self.samples)}")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # 이미지 경로는 annotations.json 기준으로 그대로 해석
        img_path = item["image"]
        img_path = os.path.join(self.data_dir, img_path)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)          # (3, H, W)

        # 텍스트 토크나이즈
        enc = self.tokenizer(
            item["text"],
            max_length=MAX_TEXT_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)         # (L,)
        attention_mask = enc["attention_mask"].squeeze(0)

        label = torch.tensor(item["label"], dtype=torch.long)
        top3 = [int(x) for x in item.get("labels_top3", [item["label"]])[:3]]
        if not top3:
            top3 = [int(item["label"])]
        while len(top3) < 3:
            top3.append(top3[-1])
        top2 = top3[:2]

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
            "labels_top2": torch.tensor(top2, dtype=torch.long),
            "labels_top3": torch.tensor(top3, dtype=torch.long),
            "image_path": item["image"],
            "text": item["text"],
        }


# ─────────────────────────────────────────────
# 데이터로더 빌더
# ─────────────────────────────────────────────
def build_dataloaders(data_dir: str,
                      tokenizer_name: str = TOKENIZER_NAME,
                      batch_size: int = BATCH_SIZE,
                      num_workers: int = NUM_WORKERS):
    """
    전체 데이터셋을 train / val / test 로 분리 후 DataLoader 반환
    """
    train_full = MultimodalDataset(data_dir, split="train", tokenizer_name=tokenizer_name)
    eval_full  = MultimodalDataset(data_dir, split="val",   tokenizer_name=tokenizer_name)

    n = len(train_full)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    n_test  = n - n_train - n_val

    # 동일한 시드로 인덱스 분리 — train은 augmentation용, val/test는 eval용 데이터셋 사용
    train_ds, val_base, test_base = random_split(
        train_full, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    val_ds  = Subset(eval_full, val_base.indices)
    test_ds = Subset(eval_full, test_base.indices)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
        "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True),
    }

    log.info(f"DataLoader 준비 완료 | train={n_train} val={n_val} test={n_test}")
    return loaders


# ─────────────────────────────────────────────
# 단독 실행 시 검증
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    loaders = build_dataloaders(data_dir)
    batch = next(iter(loaders["train"]))
    log.info(f"image shape : {batch['image'].shape}")
    log.info(f"input_ids   : {batch['input_ids'].shape}")
    log.info(f"labels      : {batch['label'][:8]}")
    log.info("전처리 검증 완료 ✓")
