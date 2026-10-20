# -*- coding: utf-8 -*-
"""
真实域混合微调（Plan A）：以 best_resnet18_38.pth（PlantVillage 99.27%）为起点，
在「PlantVillage + PlantDoc 真实域」混合数据上低学习率继续训练，提升真实田间照片精度。

与 train.py 的区别：
- 起点是当前 best 权重（不是 ImageNet 随机头），避免从零训丢实验室域精度；
- 学习率更低（默认 5e-5，可用 FINETUNE_LR 覆盖）；
- 数据来自 dataset_index_mixed.json（PV + 真实域，由 prepare_plantdoc_mix.py 生成）；
- 早停监控 val（PV）F1，保证实验室域精度不退化；
- 每轮额外在真实域测试集上打印 acc（粗监控，不用于早停）；
- 产物 finetuned_resnet18_38.pth，不覆盖原 best。

用法：
    python -m scripts.finetune_domain
前置：
    python -m scripts.prepare_plantdoc_mix   # 生成混合索引 + 真实域图片
"""

import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    MODEL_DIR, NUM_CLASSES, BATCH_SIZE, PATIENCE, NUM_WORKERS,
    USE_AMP, GRAD_CLIP, BEST_MODEL_PATH,
)
from src.model.resnet import build_resnet18
from src.data.datasets import PlantVillageDataset, load_index
from src.data.transforms import get_train_transform, get_val_transform

MIXED_INDEX = ROOT / "outputs" / "index" / "dataset_index_mixed.json"
REALTEST_INDEX = ROOT / "outputs" / "index" / "dataset_index_realtest.json"
FINETUNE_PATH = MODEL_DIR / "finetuned_resnet18_38.pth"

LR = float(os.getenv("FINETUNE_LR", "5e-5"))
EPOCHS = int(os.getenv("FINETUNE_EPOCHS", "20"))
WEIGHT_DECAY = 1e-4


def compute_class_weights(index):
    counts = [0] * NUM_CLASSES
    for _, label in index["train"]:
        counts[label] += 1
    total = sum(counts)
    weights = [total / (NUM_CLASSES * max(c, 1)) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        if USE_AMP:
            with autocast(device_type="cuda"):
                out = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, correct, total = [], [], 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        out = model(imgs)
        preds = out.argmax(1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    acc = correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, macro_f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | 微调学习率: {LR} | 轮数: {EPOCHS}")
    if not MIXED_INDEX.exists():
        raise SystemExit(f"混合索引不存在：{MIXED_INDEX}\n请先运行 python -m scripts.prepare_plantdoc_mix")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    mix = load_index(MIXED_INDEX)

    # 从训练真实图里按类分层切 15% 作「真实域 val」，用于早停/选模型。
    # real_test（225 张独立评估集）绝不能参与选模型，否则评估泄漏。
    real_idx = [i for i, (p, _l) in enumerate(mix["train"]) if "real_mix" in str(p)]
    by_cls = defaultdict(list)
    for i in real_idx:
        by_cls[mix["train"][i][1]].append(i)
    rng = random.Random(42)
    val_idx = set()
    for _lbl, idxs in by_cls.items():
        rng.shuffle(idxs)
        val_idx.update(idxs[:max(1, int(len(idxs) * 0.15))])
    train_items = [it for i, it in enumerate(mix["train"]) if i not in val_idx]
    real_val_items = [mix["train"][i] for i in sorted(val_idx)]

    train_loader = DataLoader(PlantVillageDataset(train_items, get_train_transform()),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(PlantVillageDataset(mix["val"], get_val_transform()),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    real_val_loader = DataLoader(PlantVillageDataset(real_val_items, get_val_transform()),
                                 batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    rt = load_index(REALTEST_INDEX)
    rt_loader = DataLoader(PlantVillageDataset(rt["test"], get_val_transform()),
                           batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"混合训练: {len(train_items)} | PV val: {len(mix['val'])} | "
          f"真实域 val: {len(real_val_items)} | 真实域 test(仅监控): {len(rt['test'])}")

    # 从 best 起点
    model = build_resnet18(pretrained=False).to(device)
    ckpt = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"已加载起点权重: {BEST_MODEL_PATH} (val_f1={ckpt.get('val_f1', '?')})")

    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(mix).to(device))
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = GradScaler(enabled=USE_AMP and device.type == "cuda")

    best_score, best_f1, no_improve, real_best_acc = 0.0, 0.0, 0, 0.0
    for epoch in range(EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_acc, val_f1 = evaluate(model, val_loader, device)        # PV val：守实验室域精度
        rv_acc, rv_f1 = evaluate(model, real_val_loader, device)     # 真实域 val：选模型用
        rt_acc, rt_f1 = evaluate(model, rt_loader, device)           # 真实域 test：仅监控
        scheduler.step()
        real_best_acc = max(real_best_acc, rt_acc)
        # 综合分数：PV 精度 + 真实域泛化（0.5 权重），避免只看 PV 早停把真实域提升截断
        score = val_f1 + 0.5 * rv_acc
        print(f"Ep {epoch+1:2d}/{EPOCHS} | loss {tr_loss:.4f} | tr_acc {tr_acc:.4f} | "
              f"PV_val {val_acc:.4f}/{val_f1:.4f} | REAL_val {rv_acc:.4f} | "
              f"REAL_test {rt_acc:.4f} | score {score:.4f} | {time.time()-t0:.1f}s")
        if score > best_score + 1e-4:
            best_score, best_f1, no_improve = score, val_f1, 0
            torch.save({"model": model.state_dict(), "val_f1": val_f1,
                        "real_val_acc": rv_acc, "real_test_acc": rt_acc, "epoch": epoch}, FINETUNE_PATH)
            print(f"  -> 保存微调权重 (PV_f1={val_f1:.4f}, REAL_val={rv_acc:.4f}, REAL_test={rt_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"综合分数连续 {PATIENCE} 轮无提升，早停。")
                break
    print(f"微调结束。最佳 PV val F1={best_f1:.4f}；真实域测试最佳 acc={real_best_acc:.4f}")
    print(f"产物: {FINETUNE_PATH}")
    print("下一步：用带作物先验的评估脚本对比 PlantDoc test 真实域 acc（目标 > 现状 52.5%）。")


if __name__ == "__main__":
    main()
