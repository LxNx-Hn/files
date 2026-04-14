"""
multimodal_experiment.py
========================
Poster + overview multimodal genre classification experiments.
"""

import os
import json
import csv
import time
import logging
import itertools
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
try:
    from torch.amp import GradScaler, autocast   # PyTorch 2.0+
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # PyTorch < 2.0
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from transformers import AutoModel

# ── 시각화 (matplotlib/seaborn 없으면 스킵) ─────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")   # RunPod headless 환경용 non-interactive 백엔드
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

from preprocess import build_dataloaders, IMAGE_SIZE, MAX_TEXT_LEN

os.makedirs("results", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("results/experiment.log", mode="a"),
    ],
)
log = logging.getLogger("EXPERIMENT")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS  = torch.cuda.device_count()
log.info(f"Device: {DEVICE} | 사용 가능한 GPU 수: {N_GPUS}")
if not HAS_PLOT:
    log.warning("matplotlib/seaborn 없음 — 시각화 비활성화 (pip install matplotlib seaborn)")


# ══════════════════════════════════════════════════════════════════
# 1. 이미지 인코더
# ══════════════════════════════════════════════════════════════════

class CNNImageEncoder(nn.Module):
    """ImageNet pretrained ResNet50 backbone + projection head"""

    def __init__(self, out_dim: int = 256, dropout: float = 0.0, n_layers: int = 4):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.net = nn.Sequential(*list(backbone.children())[:-1])  # 2048 x 1 x 1
        self.proj = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(2048, out_dim),
        )

    def forward(self, x):
        return self.proj(self.net(x).flatten(1))


class PatchEmbedding(nn.Module):
    """ViT 스타일 Patch Embedding"""
    def __init__(self, img_size=IMAGE_SIZE, patch_size=16, in_ch=3, emb_dim=256):
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.proj      = nn.Conv2d(in_ch, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_emb   = nn.Parameter(torch.zeros(1, n_patches + 1, emb_dim))

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        return torch.cat([cls, x], dim=1) + self.pos_emb


class TransformerImageEncoder(nn.Module):
    """Patch Embedding + Transformer Encoder → CLS 토큰"""
    def __init__(self, out_dim: int = 256, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.patch_emb = PatchEmbedding(emb_dim=out_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=n_heads, dim_feedforward=out_dim * 4,
            batch_first=True, dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, x):
        return self.encoder(self.patch_emb(x))[:, 0]


# ══════════════════════════════════════════════════════════════════
# 2. 텍스트 인코더
# ══════════════════════════════════════════════════════════════════

class DNNTextEncoder(nn.Module):
    """임베딩 → MLP (Bag-of-Embedding) 텍스트 인코더"""
    def __init__(self, vocab_size: int = 30522, emb_dim: int = 64,
                 out_dim: int = 256, seq_len: int = MAX_TEXT_LEN,
                 dropout: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * seq_len, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, out_dim), nn.ReLU(),
        )

    def forward(self, input_ids, attention_mask=None):
        return self.mlp(self.emb(input_ids).flatten(1))


class TransformerTextEncoder(nn.Module):
    """사전학습 SentenceBERT 계열 임베딩 모델 (CLS pooling)"""
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, out_dim: int = 256, freeze: bool = False, dropout: float = 0.0):
        super().__init__()
        self.bert = AutoModel.from_pretrained(self.MODEL_NAME)
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False
        hidden = self.bert.config.hidden_size   # 384
        self.proj = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.proj(out.last_hidden_state[:, 0])


# ══════════════════════════════════════════════════════════════════
# 3. 퓨전 레이어
# ══════════════════════════════════════════════════════════════════

class EarlyFusion(nn.Module):
    """Concatenation → MLP"""
    def __init__(self, feat_dim: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feat_dim * 2, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, img_feat, txt_feat):
        return self.head(torch.cat([img_feat, txt_feat], dim=-1))


class LateFusion(nn.Module):
    """각 모달 독립 헤드 → 로짓 평균"""
    def __init__(self, feat_dim: int, n_classes: int, dropout: float = 0.0):
        super().__init__()
        self.img_head = nn.Linear(feat_dim, n_classes)
        self.txt_head = nn.Linear(feat_dim, n_classes)

    def forward(self, img_feat, txt_feat):
        return (self.img_head(img_feat) + self.txt_head(txt_feat)) / 2


class WeightedLateFusion(nn.Module):
    """학습 가능한 가중합 late fusion"""
    def __init__(self, feat_dim: int, n_classes: int, dropout: float = 0.0):
        super().__init__()
        self.img_head = nn.Linear(feat_dim, n_classes)
        self.txt_head = nn.Linear(feat_dim, n_classes)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, img_feat, txt_feat):
        img_logits = self.img_head(img_feat)
        txt_logits = self.txt_head(txt_feat)
        alpha = torch.sigmoid(self.alpha)
        return alpha * img_logits + (1 - alpha) * txt_logits


class CrossAttentionFusion(nn.Module):
    """Cross-Attention: 이미지(Q) → 텍스트(K/V)"""
    def __init__(self, feat_dim: int, n_classes: int,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(feat_dim, n_heads,
                                          dropout=dropout, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, n_classes),
        )

    def forward(self, img_feat, txt_feat):
        q = img_feat.unsqueeze(1)
        k = v = txt_feat.unsqueeze(1)
        out, _ = self.attn(q, k, v)
        return self.head(out.squeeze(1))


class GatedFusion(nn.Module):
    """모달별 중요도를 gate로 조절하는 fusion"""
    def __init__(self, feat_dim: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, img_feat, txt_feat):
        concat = torch.cat([img_feat, txt_feat], dim=-1)
        gate = self.gate(concat)
        fused = gate * img_feat + (1 - gate) * txt_feat
        return self.head(fused)


class ImageOnlyFusion(nn.Module):
    """이미지 단독 분류 baseline"""
    def __init__(self, feat_dim: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, img_feat, txt_feat):
        return self.head(img_feat)


class TextOnlyFusion(nn.Module):
    """텍스트 단독 분류 baseline"""
    def __init__(self, feat_dim: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, img_feat, txt_feat):
        return self.head(txt_feat)


# ══════════════════════════════════════════════════════════════════
# 4. 전체 멀티모달 모델
# ══════════════════════════════════════════════════════════════════

class MultimodalModel(nn.Module):
    def __init__(self, img_encoder, txt_encoder, fusion):
        super().__init__()
        self.img_enc = img_encoder
        self.txt_enc = txt_encoder
        self.fusion  = fusion

    def forward(self, image, input_ids, attention_mask):
        return self.fusion(
            self.img_enc(image),
            self.txt_enc(input_ids, attention_mask),
        )


# ══════════════════════════════════════════════════════════════════
# 5. 학습 / 평가 루프
# ══════════════════════════════════════════════════════════════════

@dataclass
class ExperimentResult:
    config: Dict
    accuracy:              float = 0.0
    precision_macro:       float = 0.0
    recall_macro:          float = 0.0
    f1_macro:              float = 0.0
    precision_weighted:    float = 0.0
    recall_weighted:       float = 0.0
    f1_weighted:           float = 0.0
    relaxed_top2_accuracy: float = 0.0
    relaxed_top3_accuracy: float = 0.0
    per_class_f1:          List  = field(default_factory=list)
    confusion_matrix:      List  = field(default_factory=list)
    best_val_loss:         float = 999.0
    train_time_sec:        float = 0.0
    classification_report_str: str = ""
    predictions: List = field(default_factory=list)
    # 학습 히스토리 (시각화용)
    train_losses: List = field(default_factory=list)
    val_losses:   List = field(default_factory=list)
    val_accs:     List = field(default_factory=list)


def train_epoch(model, loader, optimizer, criterion, scaler, use_amp: bool = True):
    model.train()
    total_loss = 0
    for batch in loader:
        img   = batch["image"].to(DEVICE)
        ids   = batch["input_ids"].to(DEVICE)
        mask  = batch["attention_mask"].to(DEVICE)
        label = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        with autocast('cuda', enabled=use_amp):
            loss = criterion(model(img, ids, mask), label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, n_classes, use_amp: bool = True):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0
    all_prediction_rows = []
    relaxed_top2_hits = 0
    relaxed_top3_hits = 0
    total_examples = 0
    for batch in loader:
        img   = batch["image"].to(DEVICE)
        ids   = batch["input_ids"].to(DEVICE)
        mask  = batch["attention_mask"].to(DEVICE)
        label = batch["label"].to(DEVICE)

        with autocast('cuda', enabled=use_amp):
            logits = model(img, ids, mask)
            total_loss += criterion(logits, label).item()
        preds = logits.argmax(-1).cpu().numpy()
        labels = label.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

        image_paths = batch["image_path"]
        texts = batch["text"]
        labels_top2 = batch["labels_top2"]
        labels_top3 = batch["labels_top3"]
        for image_path, text, true_label, pred_label, true_top2, true_top3 in zip(
            image_paths, texts, labels, preds, labels_top2, labels_top3
        ):
            top2_list = [int(x) for x in true_top2]
            top3_list = [int(x) for x in true_top3]
            hit_top2 = int(int(pred_label) in top2_list)
            hit_top3 = int(int(pred_label) in top3_list)
            relaxed_top2_hits += hit_top2
            relaxed_top3_hits += hit_top3
            total_examples += 1
            all_prediction_rows.append({
                "image_path": image_path,
                "text": text,
                "true_label": int(true_label),
                "true_top2_labels": top2_list,
                "true_top3_labels": top3_list,
                "pred_label": int(pred_label),
                "correct": int(true_label == pred_label),
                "relaxed_top2_correct": hit_top2,
                "relaxed_top3_correct": hit_top3,
            })

    avg_loss = total_loss / len(loader)
    return {
        "loss":               avg_loss,
        "accuracy":           accuracy_score(all_labels, all_preds),
        "precision_macro":    precision_score(all_labels, all_preds, average="macro",    zero_division=0),
        "recall_macro":       recall_score   (all_labels, all_preds, average="macro",    zero_division=0),
        "f1_macro":           f1_score       (all_labels, all_preds, average="macro",    zero_division=0),
        "precision_weighted": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall_weighted":    recall_score   (all_labels, all_preds, average="weighted", zero_division=0),
        "f1_weighted":        f1_score       (all_labels, all_preds, average="weighted", zero_division=0),
        "confusion_matrix":   confusion_matrix(all_labels, all_preds).tolist(),
        "per_class_f1":       f1_score(all_labels, all_preds, average=None,
                                       zero_division=0).tolist(),
        "classification_report": classification_report(all_labels, all_preds, zero_division=0),
        "predictions":        all_prediction_rows,
        "relaxed_top2_accuracy": relaxed_top2_hits / max(total_examples, 1),
        "relaxed_top3_accuracy": relaxed_top3_hits / max(total_examples, 1),
    }


def save_prediction_csvs(predictions: List[Dict], save_dir: str) -> None:
    if not predictions:
        return

    os.makedirs(save_dir, exist_ok=True)
    all_path = os.path.join(save_dir, "all_predictions.csv")
    mis_path = os.path.join(save_dir, "misclassified_samples.csv")
    fieldnames = [
        "image_path", "text", "true_label", "true_top2_labels", "true_top3_labels",
        "pred_label", "correct", "relaxed_top2_correct", "relaxed_top3_correct"
    ]

    with open(all_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    mis_rows = [row for row in predictions if row["correct"] == 0]
    with open(mis_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mis_rows)

    log.info(f"  Saved predictions: {all_path}")
    log.info(f"  Saved misclassified samples: {mis_path}")


def run_experiment(
    config:        Dict,
    loaders:       Dict,
    n_classes:     int,
    epochs:        int   = 50,
    lr:            float = 1e-3,
    img_lr:        float = 1e-4,
    txt_lr:        float = 2e-5,
    fusion_lr:     float = 1e-3,
    weight_decay:  float = 1e-4,
    warmup_epochs: int   = 3,
    feat_dim:      int   = 256,
    dropout:       float = 0.3,
    cnn_layers:    int   = 3,
    freeze_bert:   bool  = False,
    results_dir:   str   = "./results",
    save_models:   bool  = False,
    use_amp:       bool  = True,
) -> ExperimentResult:
    """
    config = {
        "img_encoder": "cnn" | "transformer",
        "txt_encoder": "dnn" | "transformer",
        "fusion":      "image_only" | "text_only" | "early" | "late" | "weighted_late" | "gated" | "cross_attention",
    }
    """
    cnn_suffix = f"(L{cnn_layers})" if config["img_encoder"] == "cnn" else ""
    cfg_name = "{img_encoder}+{txt_encoder}+{fusion}".format(**config) + cnn_suffix
    log.info(f"━━━━ 실험 시작: {cfg_name} ━━━━")

    # ── 인코더 / 퓨전 생성 ──
    if config["img_encoder"] == "cnn":
        img_enc = CNNImageEncoder(out_dim=feat_dim, dropout=dropout * 0.5,
                                  n_layers=cnn_layers)
    else:
        img_enc = TransformerImageEncoder(out_dim=feat_dim, n_heads=4,
                                          n_layers=2, dropout=dropout * 0.3)

    if config["txt_encoder"] == "dnn":
        txt_enc = DNNTextEncoder(out_dim=feat_dim, dropout=dropout * 0.5)
    else:
        txt_enc = TransformerTextEncoder(out_dim=feat_dim,
                                         freeze=freeze_bert, dropout=dropout * 0.3)

    if config["fusion"] == "image_only":
        fusion = ImageOnlyFusion(feat_dim, n_classes, dropout=dropout)
    elif config["fusion"] == "text_only":
        fusion = TextOnlyFusion(feat_dim, n_classes, dropout=dropout)
    elif config["fusion"] == "early":
        fusion = EarlyFusion(feat_dim, n_classes, dropout=dropout)
    elif config["fusion"] == "late":
        fusion = LateFusion(feat_dim, n_classes)
    elif config["fusion"] == "weighted_late":
        fusion = WeightedLateFusion(feat_dim, n_classes)
    elif config["fusion"] == "gated":
        fusion = GatedFusion(feat_dim, n_classes, dropout=dropout)
    else:
        fusion = CrossAttentionFusion(feat_dim, n_classes, n_heads=4,
                                      dropout=dropout * 0.3)

    model = MultimodalModel(img_enc, txt_enc, fusion).to(DEVICE)

    # ── 멀티 GPU (DataParallel) ──
    if N_GPUS > 1:
        model = nn.DataParallel(model)
        log.info(f"  DataParallel 활성화: {N_GPUS}개 GPU 병렬 처리")

    # ── 옵티마이저 ──
    param_groups = [
        {"params": img_enc.parameters(), "lr": img_lr},
        {"params": fusion.parameters(), "lr": fusion_lr},
    ]
    if config["txt_encoder"] == "transformer":
        param_groups.append({"params": txt_enc.bert.parameters(), "lr": txt_lr})
        if hasattr(txt_enc, "proj"):
            param_groups.append({"params": txt_enc.proj.parameters(), "lr": fusion_lr})
    else:
        param_groups.append({"params": txt_enc.parameters(), "lr": fusion_lr})

    optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

    # ── LR 스케줄러: 웜업 + 코사인 감쇠 ──
    if warmup_epochs > 0 and epochs > warmup_epochs:
        warmup_sched = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                total_iters=warmup_epochs)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        scheduler = SequentialLR(optimizer,
                                 schedulers=[warmup_sched, cosine_sched],
                                 milestones=[warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    criterion = nn.CrossEntropyLoss()
    scaler    = GradScaler('cuda', enabled=use_amp)

    best_val_loss = float("inf")
    best_state    = None
    train_losses, val_losses, val_accs = [], [], []
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss     = train_epoch(model, loaders["train"], optimizer, criterion,
                                  scaler, use_amp)
        val_metrics = evaluate(model, loaders["val"], criterion, n_classes, use_amp)
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(val_metrics["loss"])
        val_accs.append(val_metrics["accuracy"])

        log.info(
            f"[{cfg_name}] epoch {epoch:03d}/{epochs} | "
            f"train={tr_loss:.4f} val={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} "
            f"rlx2={val_metrics['relaxed_top2_accuracy']:.4f} "
            f"rlx3={val_metrics['relaxed_top3_accuracy']:.4f} "
            f"f1={val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            _m = model.module if isinstance(model, nn.DataParallel) else model
            best_state = {k: v.detach().clone() for k, v in _m.state_dict().items()}

    # ── 최적 가중치 복원 ──
    _m = model.module if isinstance(model, nn.DataParallel) else model
    _m.load_state_dict(best_state)

    # ── 모델 저장 (선택) ──
    if save_models:
        models_dir = os.path.join(results_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        safe = cfg_name.replace("+", "_")
        torch.save(best_state, os.path.join(models_dir, f"{safe}_best.pt"))
        log.info(f"  모델 저장: models/{safe}_best.pt")

    test_metrics = evaluate(model, loaders["test"], criterion, n_classes, use_amp)
    elapsed = time.time() - t_start

    result = ExperimentResult(
        config=config,
        accuracy=test_metrics["accuracy"],
        precision_macro=test_metrics["precision_macro"],
        recall_macro=test_metrics["recall_macro"],
        f1_macro=test_metrics["f1_macro"],
        precision_weighted=test_metrics["precision_weighted"],
        recall_weighted=test_metrics["recall_weighted"],
        f1_weighted=test_metrics["f1_weighted"],
        relaxed_top2_accuracy=test_metrics["relaxed_top2_accuracy"],
        relaxed_top3_accuracy=test_metrics["relaxed_top3_accuracy"],
        per_class_f1=test_metrics["per_class_f1"],
        confusion_matrix=test_metrics["confusion_matrix"],
        best_val_loss=best_val_loss,
        train_time_sec=round(elapsed, 2),
        classification_report_str=test_metrics["classification_report"],
        predictions=test_metrics["predictions"],
        train_losses=train_losses,
        val_losses=val_losses,
        val_accs=val_accs,
    )

    log.info(
        f"[{cfg_name}] TEST → acc={result.accuracy:.4f} "
        f"rlx2={result.relaxed_top2_accuracy:.4f} "
        f"rlx3={result.relaxed_top3_accuracy:.4f} "
        f"f1_macro={result.f1_macro:.4f} f1_weighted={result.f1_weighted:.4f} "
        f"({elapsed / 60:.1f}분)"
    )

    # ── 개별 학습 곡선 저장 ──
    if HAS_PLOT:
        _save_training_curve(result, results_dir)
    save_prediction_csvs(result.predictions, results_dir)

    return result


# ══════════════════════════════════════════════════════════════════
# 6. 시각화 함수
# ══════════════════════════════════════════════════════════════════

def _cfg_name(result: ExperimentResult) -> str:
    fusion_name = result.config["fusion"]
    if fusion_name == "cross_attention":
        fusion_name = "joint"
    name = f"{result.config['img_encoder']}+{result.config['txt_encoder']}+{fusion_name}"
    if result.config.get("img_encoder") == "cnn":
        name += f"+L{result.config.get('cnn_layers', 3)}"
    return name


def _save_training_curve(result: ExperimentResult, save_dir: str):
    """에폭별 학습 곡선 (Loss + Accuracy) 저장"""
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    name   = _cfg_name(result)
    epochs = range(1, len(result.train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(name, fontsize=12, fontweight="bold")

    ax1.plot(epochs, result.train_losses, label="Train Loss",  color="#2196F3")
    ax1.plot(epochs, result.val_losses,   label="Val Loss",    color="#FF5722")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, result.val_accs, label="Val Accuracy", color="#4CAF50")
    ax2.axhline(result.accuracy, linestyle="--", color="#9C27B0",
                label=f"Test Acc={result.accuracy:.4f}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    safe = name.replace("+", "_")
    fig.savefig(os.path.join(plots_dir, f"{safe}_curve.png"), dpi=100,
                bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved training curve: plots/{safe}_curve.png")


def plot_f1_comparison(results: List[ExperimentResult], save_dir: str):
    """전체 실험 F1 비교 가로 바 차트"""
    if not HAS_PLOT:
        return
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    names  = [_cfg_name(r) for r in results]
    f1_wgt = [r.f1_weighted for r in results]
    f1_mac = [r.f1_macro    for r in results]
    accs   = [r.accuracy    for r in results]

    order  = sorted(range(len(f1_wgt)), key=lambda i: f1_wgt[i], reverse=True)
    names  = [names[i]  for i in order]
    f1_wgt = [f1_wgt[i] for i in order]
    f1_mac = [f1_mac[i] for i in order]
    accs   = [accs[i]   for i in order]

    y = np.arange(len(names))
    w = 0.28

    fig, ax = plt.subplots(figsize=(14, max(6, len(names) * 0.7)))
    b1 = ax.barh(y + w, f1_wgt, w, label="F1 Weighted", color="#2196F3")
    ax.barh(y,       f1_mac, w, label="F1 Macro",    color="#FF5722")
    ax.barh(y - w,   accs,   w, label="Accuracy",    color="#4CAF50")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Score")
    ax.set_title("Experiment Comparison (sorted by F1 Weighted)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, 1.1)

    for bar in b1:
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.4f}", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "f1_comparison.png"), dpi=120,
                bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved F1 comparison plot: plots/f1_comparison.png")


def plot_per_class_f1(result: ExperimentResult, save_dir: str, top_n: int = 30):
    """클래스별 F1 — 상위/하위 top_n 표시"""
    if not HAS_PLOT or not result.per_class_f1:
        return
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    f1s        = result.per_class_f1
    n          = len(f1s)
    idx_sorted = sorted(range(n), key=lambda i: f1s[i], reverse=True)
    show       = min(top_n, n)
    top_idx    = idx_sorted[:show]
    bot_idx    = idx_sorted[-show:] if n > show else []

    ncols = 2 if bot_idx else 1
    fig, axes = plt.subplots(1, ncols, figsize=(16, max(6, show * 0.35)))
    if ncols == 1:
        axes = [axes]

    def _bar(ax, indices, title):
        vals   = [f1s[i] for i in indices]
        ys     = np.arange(len(indices))
        colors = ["#4CAF50" if v >= 0.5 else "#FF9800" if v >= 0.3 else "#F44336"
                  for v in vals]
        ax.barh(ys, vals, color=colors)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"Class {i}" for i in indices], fontsize=8)
        ax.set_xlabel("F1 Score")
        ax.set_xlim(0, 1.05)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
        ax.axvline(0.5, linestyle="--", color="gray", alpha=0.5)

    _bar(axes[0], top_idx, f"Top {show} Classes by F1")
    if bot_idx:
        _bar(axes[1], bot_idx, f"Bottom {show} Classes by F1")

    fig.suptitle(f"Per-class F1 - {_cfg_name(result)}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "per_class_f1_best.png"), dpi=100,
                bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved per-class F1 plot: plots/per_class_f1_best.png")


def plot_confusion_matrix(result: ExperimentResult, save_dir: str, n_classes: int):
    """정규화 혼동 행렬 히트맵 (클래스 수 <= 30일 때만 생성)"""
    if not HAS_PLOT:
        return
    if n_classes > 30:
        log.info(f"  Skip confusion matrix plot because n_classes={n_classes} > 30")
        return
    plots_dir = os.path.join(save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    cm   = np.array(result.confusion_matrix)
    norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(max(8, n_classes * 0.45),
                                    max(6, n_classes * 0.45)))
    sns.heatmap(norm, annot=(n_classes <= 15), fmt=".2f", cmap="Blues",
                ax=ax, cbar=True, square=True, linewidths=0.3)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Normalized Confusion Matrix - {_cfg_name(result)}")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "confusion_matrix_best.png"), dpi=100,
                bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved confusion matrix: plots/confusion_matrix_best.png")


# ══════════════════════════════════════════════════════════════════
# 7. 최종 요약 리포트 생성
# ══════════════════════════════════════════════════════════════════

RANK_METRIC = "f1_weighted"


def generate_summary(results: List[ExperimentResult],
                     save_dir: str = "results",
                     n_classes: int = 101):
    os.makedirs(save_dir, exist_ok=True)

    rows = []
    for r in results:
        name = _cfg_name(r)
        rows.append({
            "config_name":        name,
            "img_encoder":        r.config["img_encoder"],
            "txt_encoder":        r.config["txt_encoder"],
            "fusion":             r.config["fusion"],
            "accuracy":           round(r.accuracy, 4),
            "precision_macro":    round(r.precision_macro, 4),
            "recall_macro":       round(r.recall_macro, 4),
            "f1_macro":           round(r.f1_macro, 4),
            "precision_weighted": round(r.precision_weighted, 4),
            "recall_weighted":    round(r.recall_weighted, 4),
            "f1_weighted":        round(r.f1_weighted, 4),
            "relaxed_top2_accuracy": round(r.relaxed_top2_accuracy, 4),
            "relaxed_top3_accuracy": round(r.relaxed_top3_accuracy, 4),
            "per_class_f1":       [round(v, 4) for v in r.per_class_f1],
            "confusion_matrix":   r.confusion_matrix,
            "best_val_loss":      round(r.best_val_loss, 4),
            "train_time_sec":     r.train_time_sec,
            "classification_report": r.classification_report_str,
        })

    rows_sorted = sorted(rows, key=lambda x: x[RANK_METRIC], reverse=True)
    best = rows_sorted[0]

    summary = {
        "rank_metric": RANK_METRIC,
        "best_config": best["config_name"],
        "best_scores": {
            "accuracy":        best["accuracy"],
            "precision_macro": best["precision_macro"],
            "recall_macro":    best["recall_macro"],
            "f1_macro":        best["f1_macro"],
            "f1_weighted":     best["f1_weighted"],
            "relaxed_top2_accuracy": best["relaxed_top2_accuracy"],
            "relaxed_top3_accuracy": best["relaxed_top3_accuracy"],
        },
        "all_results": rows_sorted,
    }

    json_path = os.path.join(save_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    txt_path = os.path.join(save_dir, "summary_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  Multimodal Image+Text Experiment Summary Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Metric: {RANK_METRIC}\n")
        f.write(f"Best Config: {best['config_name']}\n\n")
        header = (f"{'Rank':<4} {'Config':40} "
                  f"{'Acc':>6} {'P_mac':>6} {'R_mac':>6} {'F1_mac':>7} "
                  f"{'F1_wgt':>7} {'Rlx@2':>7} {'Rlx@3':>7} {'Time(s)':>8}")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for rank, row in enumerate(rows_sorted, 1):
            f.write(
                f"{rank:<4} {row['config_name']:40} "
                f"{row['accuracy']:>6.4f} {row['precision_macro']:>6.4f} "
                f"{row['recall_macro']:>6.4f} {row['f1_macro']:>7.4f} "
                f"{row['f1_weighted']:>7.4f} {row['relaxed_top2_accuracy']:>7.4f} "
                f"{row['relaxed_top3_accuracy']:>7.4f} {row['train_time_sec']:>8.1f}\n"
            )
        f.write("\n" + "=" * 70 + "\n")
        f.write("  Detailed Classification Report (test split)\n")
        f.write("=" * 70 + "\n\n")
        for row in rows_sorted:
            f.write(f"-- {row['config_name']} --\n")
            f.write(row["classification_report"])
            f.write("\n")

    log.info(f"요약 저장 완료 → {json_path}")
    log.info(f"리포트 저장 완료 → {txt_path}")

    # ── 비교 시각화 ──
    if HAS_PLOT:
        plot_f1_comparison(results, save_dir)
        best_result = next(r for r in results if _cfg_name(r) == best["config_name"])
        plot_per_class_f1(best_result, save_dir)
        plot_confusion_matrix(best_result, save_dir, n_classes)

    return summary


# ══════════════════════════════════════════════════════════════════
# 8. 메인 실행
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="포스터+줄거리 멀티모달 실험")

    # ── 데이터 / 출력 ──
    parser.add_argument("--data_dir",    default="./data",    help="데이터 루트")
    parser.add_argument("--results_dir", default="./results", help="결과 저장 경로")
    parser.add_argument("--n_classes",   type=int, default=15, help="number of classes")

    # ── 학습 하이퍼파라미터 ──
    parser.add_argument("--epochs",        type=int,   default=20,
                        help="학습 에폭 수 (BERT동결 50-100, BERT파인튜닝 30-50 권장)")
    parser.add_argument("--batch_size",    type=int,   default=32,
                        help="전체 배치 크기 (DataParallel 시 GPU당 batch_size/N_GPUS)")
    parser.add_argument("--lr",            type=float, default=1e-3, help="초기 학습률")
    parser.add_argument("--img_lr",        type=float, default=1e-4, help="image encoder learning rate")
    parser.add_argument("--txt_lr",        type=float, default=2e-5, help="transformer text encoder learning rate")
    parser.add_argument("--fusion_lr",     type=float, default=1e-3, help="fusion/classifier learning rate")
    parser.add_argument("--weight_decay",  type=float, default=1e-4, help="AdamW 가중치 감쇠")
    parser.add_argument("--warmup_epochs", type=int,   default=3,
                        help="LR 웜업 에폭 수 (0=비활성화)")

    # ── 모델 아키텍처 ──
    parser.add_argument("--feat_dim",    type=int,   default=256,
                        help="인코더 공통 출력 차원")
    parser.add_argument("--dropout",     type=float, default=0.3,
                        help="기본 드롭아웃 비율 (인코더/퓨전에 비례 적용)")
    parser.add_argument("--cnn_layers",  nargs="+",  type=int, default=[4],
                        choices=[3, 4, 5],
                        help="CNN 레이어 수 목록 (예: --cnn_layers 3 4 5)")
    parser.add_argument("--freeze_bert", action="store_true",
                        help="SentenceBERT 파라미터 동결 (학습 속도 약 2배 향상, 성능 소폭 하락)")

    # ── 실험 조합 선택 (일부만 실행 가능) ──
    parser.add_argument("--img_encoders", nargs="+",
                        default=["cnn"],
                        choices=["cnn", "transformer"],
                        help="실험할 이미지 인코더 (공백 구분, 예: --img_encoders cnn)")
    parser.add_argument("--txt_encoders", nargs="+",
                        default=["transformer"],
                        choices=["dnn", "transformer"],
                        help="실험할 텍스트 인코더")
    parser.add_argument("--fusions", nargs="+",
                        default=["image_only", "text_only", "early", "late", "weighted_late", "gated", "cross_attention"],
                        choices=["image_only", "text_only", "early", "late", "weighted_late", "gated", "cross_attention"],
                        help="실험할 퓨전 전략")

    # ── 기타 ──
    parser.add_argument("--no_amp",      action="store_true",
                        help="Mixed Precision 비활성화 (CPU 또는 구형 GPU)")
    parser.add_argument("--save_models", action="store_true",
                        help="각 실험 최적 가중치를 results/models/ 에 저장")

    args = parser.parse_args()

    use_amp = not args.no_amp and DEVICE.type == "cuda"

    log.info("=" * 60)
    log.info(f"  에폭        : {args.epochs}")
    log.info(f"  배치 크기   : {args.batch_size}  (GPU당 {args.batch_size // max(N_GPUS, 1)})")
    log.info(f"  학습률      : base={args.lr} img={args.img_lr} txt={args.txt_lr} fusion={args.fusion_lr}")
    log.info(f"  웜업        : {args.warmup_epochs} epochs")
    log.info(f"  weight_decay: {args.weight_decay}")
    log.info(f"  feat_dim    : {args.feat_dim}  |  dropout: {args.dropout}  |  cnn_layers: {args.cnn_layers}")
    log.info(f"  freeze_bert : {args.freeze_bert}")
    log.info(f"  Mixed Prec  : {'ON' if use_amp else 'OFF'}")
    log.info(f"  Multi-GPU   : {'ON — DataParallel' if N_GPUS > 1 else 'OFF'} ({N_GPUS}개 GPU)")
    log.info("=" * 60)

    loaders = build_dataloaders(args.data_dir, batch_size=args.batch_size)

    # CNN 레이어 수는 CNN 이미지 인코더에만 의미 있음
    # img_encoder가 transformer일 때는 cnn_layers=[3]으로 고정해 중복 실험 방지
    cnn_layers_list = sorted(set(args.cnn_layers))
    configs = []
    for ie, te, fu in itertools.product(args.img_encoders, args.txt_encoders, args.fusions):
        if ie == "cnn":
            for cl in cnn_layers_list:
                configs.append({"img_encoder": ie, "txt_encoder": te, "fusion": fu, "cnn_layers": cl})
        else:
            configs.append({"img_encoder": ie, "txt_encoder": te, "fusion": fu, "cnn_layers": 3})
    log.info(f"총 실험 수: {len(configs)}")

    all_results = []
    for cfg in configs:
        try:
            result = run_experiment(
                config=cfg,
                loaders=loaders,
                n_classes=args.n_classes,
                epochs=args.epochs,
                lr=args.lr,
                img_lr=args.img_lr,
                txt_lr=args.txt_lr,
                fusion_lr=args.fusion_lr,
                weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs,
                feat_dim=args.feat_dim,
                dropout=args.dropout,
                cnn_layers=cfg["cnn_layers"],
                freeze_bert=args.freeze_bert,
                results_dir=args.results_dir,
                save_models=args.save_models,
                use_amp=use_amp,
            )
            all_results.append(result)
        except Exception as e:
            log.error(f"실험 실패 {cfg}: {e}", exc_info=True)

    summary = generate_summary(all_results,
                               save_dir=args.results_dir,
                               n_classes=args.n_classes)

    log.info("\n" + "=" * 60)
    log.info(f"  ✅ 최적 구성  : {summary['best_config']}")
    log.info(f"  📊 기준 지표  : {RANK_METRIC} = {summary['best_scores']['f1_weighted']:.4f}")
    log.info(f"  🎯 Accuracy   : {summary['best_scores']['accuracy']:.4f}")
    log.info(f"  🧪 Relaxed@2  : {summary['best_scores']['relaxed_top2_accuracy']:.4f}")
    log.info(f"  🧪 Relaxed@3  : {summary['best_scores']['relaxed_top3_accuracy']:.4f}")
    log.info(f"  📌 F1 Macro   : {summary['best_scores']['f1_macro']:.4f}")
    log.info("=" * 60)
