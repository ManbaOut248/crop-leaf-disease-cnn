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
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    MODEL_DIR, NUM_CLASSES, BATCH_SIZE, PATIENCE, NUM_WORKERS,
    USE_AMP, GRAD_CLIP, BEST_MODEL_PATH,
)

# 数据加载是 4.7 万张混合集的主要瓶颈：NUM_WORKERS=4 时 GPU 利用率长期趴着，
# 单轮 ~5.8 min 里绝大部分时间在等 IO/解码，而不是在算。这里允许环境变量覆盖
# （16 核机器上放到 10 个 worker），不动全局 config —— 那个值训练/推理共用。
NUM_WORKERS = int(os.getenv("FINETUNE_WORKERS", str(NUM_WORKERS)))
from src.model.resnet import build_resnet18
from src.data.datasets import PlantVillageDataset, load_index
from src.data.transforms import get_train_transform, get_val_transform

MIXED_INDEX = ROOT / "outputs" / "index" / os.getenv("FINETUNE_INDEX", "dataset_index_mixed.json")
REALTEST_INDEX = ROOT / "outputs" / "index" / os.getenv("FINETUNE_REALTEST", "dataset_index_realtest.json")
FINETUNE_PATH = MODEL_DIR / os.getenv("FINETUNE_OUT", "finetuned_resnet18_38.pth")
# 起点权重。默认 config.BEST_MODEL_PATH；做真实域 A/B 实验时**必须**显式指定
# 10-20 微调前的纯 PlantVillage 基座，否则起点已见过真实域数据 → 留出集泄漏。
INIT_PATH = Path(os.getenv("FINETUNE_INIT", str(BEST_MODEL_PATH)))

LR = float(os.getenv("FINETUNE_LR", "5e-5"))
EPOCHS = int(os.getenv("FINETUNE_EPOCHS", "20"))
WEIGHT_DECAY = 1e-4
# 真实域样本的额外损失权重（PV 样本恒为 1.0）。默认 1.0 = 与旧行为完全一致，
# 便于做「只改这一个变量」的 A/B 对照。见 docs/22。
REAL_DOMAIN_WEIGHT = float(os.getenv("REAL_DOMAIN_WEIGHT", "1.0"))


def is_real(path):
    """真实域样本判定：索引里的路径含 real_mix（PV 图片在 PlantVillage 下）。"""
    return "real_mix" in str(path).replace("\\", "/")


def sample_weights_of(items, real_weight):
    """与 items 同序的样本权重：PV=1.0，真实域=real_weight。

    为什么需要「样本级」权重：类权重只按类别频次算，无法区分同类的 PV 图与真实图。
    番茄在 PV 里 14,537 张，类权重被压到约 0.078；真实域番茄图因此也被同样压低，
    而它恰恰是唯一能教会模型「田间长什么样」的信号。这里对真实域样本单独乘系数。
    """
    return [real_weight if is_real(p) else 1.0 for p, _ in items]


class DomainWeightedDataset(Dataset):
    """在原数据集外包一层，额外返回样本权重。

    不改 src/data/datasets.py：那个数据集被训练/评估/推理共用，加第三个返回值会波及
    所有调用方（包括 val_loader 的 unpack）。这里只在微调入口局部包装，风险最小。
    """

    def __init__(self, items, transform, weights):
        from src.data.datasets import PlantVillageDataset
        self.base = PlantVillageDataset(items, transform)
        self.weights = weights

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return img, label, self.weights[idx]



def compute_class_weights(index):
    counts = [0] * NUM_CLASSES
    for _, label in index["train"]:
        counts[label] += 1
    total = sum(counts)
    weights = [total / (NUM_CLASSES * max(c, 1)) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, class_weight, optimizer, scaler, device):
    """一个 epoch。

    损失 = Σ(w_i · ce_i) / Σ(w_i)，其中 w_i = class_weight[label_i] × sample_weight_i
    （sample_weight：PV=1，真实域=REAL_DOMAIN_WEIGHT）。

    ⚠️ 归一化分母必须是「权重之和」而非 batch 样本数：PyTorch 的
    `nn.CrossEntropyLoss(weight=w, reduction='mean')` 内部就是除以目标类权重之和
    （实测 4.2085 = Σ(w·ce)/Σ(w)，而非 Σ(w·ce)/N=13.17）。
    所以当所有 sample_weight=1 时，本实现与旧代码逐位一致，A/B 对照只差一个变量。
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels, sw in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        sw = sw.to(device, non_blocking=True)

        def _loss(out):
            ce = F.cross_entropy(out, labels, reduction="none")   # 未加权的逐样本 NLL
            w = class_weight[labels] * sw
            return (ce * w).sum() / w.sum()

        optimizer.zero_grad()
        if USE_AMP:
            with autocast(device_type="cuda"):
                out = model(imgs)
                loss = _loss(out)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs)
            loss = _loss(out)
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
    # 重定向到文件时 stdout 默认是全缓冲，进程被 kill 会丢掉全部进度行 ——
    # 昨天 ctrl 组跑了 51 分钟，日志里只剩 290 行 stderr 的 warning，一行 epoch 都没有。
    # 强制逐行 flush，之后无论是 Ctrl+C 还是 taskkill 都能留下已完成轮次。
    sys.stdout.reconfigure(line_buffering=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | 微调学习率: {LR} | 轮数: {EPOCHS} | DataLoader workers: {NUM_WORKERS}")
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

    # 各 loader 共用的加载参数：常驻 worker 避免每轮重新 fork（4.7 万张要重建 4 次迭代器），
    # prefetch_factor=4 让每个 worker 预取 4 个 batch，把 GPU 喂饱。
    LOADER_KW = dict(num_workers=NUM_WORKERS, pin_memory=True,
                     persistent_workers=NUM_WORKERS > 0)
    if NUM_WORKERS > 0:
        LOADER_KW["prefetch_factor"] = 4

    train_sw = sample_weights_of(train_items, REAL_DOMAIN_WEIGHT)
    train_loader = DataLoader(DomainWeightedDataset(train_items, get_train_transform(), train_sw),
                              batch_size=BATCH_SIZE, shuffle=True, **LOADER_KW)
    val_loader = DataLoader(PlantVillageDataset(mix["val"], get_val_transform()),
                            batch_size=BATCH_SIZE, shuffle=False, **LOADER_KW)
    real_val_loader = DataLoader(PlantVillageDataset(real_val_items, get_val_transform()),
                                 batch_size=BATCH_SIZE, shuffle=False, **LOADER_KW)
    rt = load_index(REALTEST_INDEX)
    rt_loader = DataLoader(PlantVillageDataset(rt["test"], get_val_transform()),
                           batch_size=BATCH_SIZE, shuffle=False, **LOADER_KW)
    n_real = sum(1 for p, _ in train_items if is_real(p))
    w_real_sum = sum(w for w in train_sw)
    print(f"混合训练: {len(train_items)} | PV val: {len(mix['val'])} | "
          f"真实域 val: {len(real_val_items)} | 真实域 test(仅监控): {len(rt['test'])}")
    print(f"真实域样本: {n_real} 张（占 {n_real / len(train_items):.1%}）| "
          f"真实域额外权重 REAL_DOMAIN_WEIGHT={REAL_DOMAIN_WEIGHT} "
          f"-> 权重占比 {1 - (len(train_items) - n_real) / w_real_sum:.1%}")
    print(f"索引: {MIXED_INDEX.name} | 监控评估集: {REALTEST_INDEX.name} | 产物: {FINETUNE_PATH.name}")

    # 从 best 起点
    model = build_resnet18(pretrained=False).to(device)
    # 起点：默认从 BEST_MODEL_PATH；设 FINETUNE_RESUME=1 则从已有 finetuned 权重继续（断点续训）
    resume = os.getenv("FINETUNE_RESUME", "0") == "1"
    if resume and FINETUNE_PATH.exists():
        ckpt = torch.load(FINETUNE_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"续训：从 {FINETUNE_PATH} 加载 (epoch={ckpt.get('epoch')}, "
              f"PV_f1={ckpt.get('val_f1', 0):.4f}, REAL_val={ckpt.get('real_val_acc', 0):.4f})")
    else:
        ckpt = torch.load(INIT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"已加载起点权重: {INIT_PATH} (val_f1={ckpt.get('val_f1', '?')}, "
              f"epoch={ckpt.get('epoch', '?')})")
        if "real_test_acc" in ckpt:
            print("  ⚠️ 该起点含 real_test_acc 字段，说明它已用真实域训练过；"
                  "做留出集对照实验前请确认不会泄漏。")

    class_weight = compute_class_weights(mix).to(device)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = GradScaler(enabled=USE_AMP and device.type == "cuda")

    # 续训时从已有 checkpoint 的分数起步，避免第一轮无条件覆盖掉更好的旧权重
    init_f1 = float(ckpt.get("val_f1") or 0.0) if resume else 0.0
    init_rv = float(ckpt.get("real_val_acc") or 0.0) if resume else 0.0
    best_score, best_f1, no_improve, real_best_acc = init_f1 + 0.5 * init_rv, init_f1, 0, 0.0
    for epoch in range(EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, class_weight, optimizer, scaler, device)
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
