# -*- coding: utf-8 -*-
"""
ResNet18 训练脚本（对齐任务书“训练”与“优化”章节）。
- 迁移学习：ImageNet 预训练 + 末端替换为 38 分类
- 类别不均衡：逆频率加权交叉熵（USE_CLASS_WEIGHT）
- 优化：Adam + 余弦退火；混合精度 AMP（适配 8GB 显存）
- 早停：监控验证集宏平均 F1，连续 PATIENCE 轮无提升则停止
- 断点续训：自动加载最新 checkpoint

用法：
    python -m src.model.train
"""

import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim import Adam, lr_scheduler
from sklearn.metrics import f1_score

from src.config import (
    MODEL_DIR, INDEX_DIR, NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS,
    LEARNING_RATE, WEIGHT_DECAY, PATIENCE, NUM_WORKERS, USE_AMP,
    USE_CLASS_WEIGHT, GRAD_CLIP, BEST_MODEL_PATH, LAST_MODEL_PATH,
)
from src.data.datasets import build_loaders, load_index
from src.model.resnet import build_resnet18

# 模型权重路径集中到 config（见 #7），本地仅作别名保持下文兼容
CKPT_PATH = LAST_MODEL_PATH
BEST_PATH = BEST_MODEL_PATH


def compute_class_weights(index):
    """按训练集逆频率计算类别权重。"""
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
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        out = model(imgs)
        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    acc = correct / total
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, macro_f1, all_preds, all_labels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    index_path = INDEX_DIR / "dataset_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"索引不存在，请先运行 python -m src.data.build_index：{index_path}")
    index = load_index(index_path)
    train_loader, val_loader, _ = build_loaders(index_path, BATCH_SIZE, NUM_WORKERS)

    model = build_resnet18(pretrained=True, freeze_backbone=False).to(device)

    criterion = nn.CrossEntropyLoss()
    if USE_CLASS_WEIGHT:
        w = compute_class_weights(index).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)
    scaler = GradScaler(enabled=USE_AMP and device.type == "cuda")

    start_epoch = 0
    best_f1 = 0.0
    no_improve = 0
    if CKPT_PATH.exists():
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("best_f1", 0.0)
        no_improve = ckpt.get("no_improve", 0)   # 恢复早停计数器，避免 resume 后多跑 PATIENCE 轮
        print(f"已加载断点，从第 {start_epoch} 轮继续；历史最佳 F1={best_f1:.4f}；早停计数={no_improve}")


    for epoch in range(start_epoch, NUM_EPOCHS):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - t0

        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | loss {train_loss:.4f} | "
              f"train_acc {train_acc:.4f} | val_acc {val_acc:.4f} | val_f1 {val_f1:.4f} | "
              f"lr {lr_now:.2e} | {epoch_time:.1f}s")

        # 保存断点（原子操作：先写 .tmp 再 os.replace，避免写盘中途被杀导致文件损坏）
        ckpt_tmp = CKPT_PATH.with_name(CKPT_PATH.name + ".tmp")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_f1": best_f1,
            "no_improve": no_improve,   # 持久化早停计数器，断点续训时不丢失
        }, ckpt_tmp)
        os.replace(ckpt_tmp, CKPT_PATH)

        # 早停判定（以宏平均 F1 为准，更关注小类）
        if val_f1 > best_f1 + 1e-4:
            best_f1 = val_f1
            no_improve = 0
            torch.save({"model": model.state_dict(), "val_f1": val_f1, "epoch": epoch},
                       BEST_PATH)
            print(f"  -> 新的最佳模型已保存 (val_f1={best_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"验证 F1 连续 {PATIENCE} 轮无提升，早停。")
                break

    print("训练结束。最佳验证宏平均 F1 =", round(best_f1, 4))
    print(f"最佳模型: {BEST_PATH}")


if __name__ == "__main__":
    main()
