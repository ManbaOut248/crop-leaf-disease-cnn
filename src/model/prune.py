# -*- coding: utf-8 -*-
"""
结构化通道剪枝（ResNet18 专用，残差对齐）+ 短微调 —— 任务书 3.3「模型剪枝」步骤。

安全策略：仅剪每个 BasicBlock 内部的「中间通道」（conv1.out = conv2.in 同步裁剪），
保持每层的输入/输出通道数不变。这样：
  - 残差跳连（main 路径 out_c == shortcut out_c）天然对齐，无需改 downsample；
  - 各 layer 间通道数不变，fc(512->38) / stem(3->64) 维度不受影响，无需裁切。
因此剪枝不会引发维度不匹配，风险低；代价是只减小「块内」计算量（而非跨层），
但体积与推理速度仍明显下降，且精度经微调可恢复。

产出：models/pruned_resnet18_38.pth（state_dict）。不改动原 best_resnet18_38.pth。
"""
import json
import numpy as np
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

from src.config import MODEL_DIR, NUM_CLASSES, INDEX_DIR, BEST_MODEL_PATH, PRUNED_MODEL_PATH
from src.data.transforms import get_val_transform
from src.model.resnet import build_resnet18

PRUNE_RATIO = 0.2        # 中间通道裁剪比例（保守，可下调到 0.1 若精度掉多）
FINETUNE_EPOCHS = 4     # 剪枝后微调轮数（短训恢复精度）


# ---------- 1) 数据 ----------
class IndexImageDataset(Dataset):
    """按 dataset_index.json 的某个切分构建数据集。"""

    def __init__(self, split, transform):
        idx = json.loads((INDEX_DIR / "dataset_index.json").read_text(encoding="utf-8"))
        self.samples = idx[split]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, label = self.samples[i]
        img = Image.open(p).convert("RGB")
        return self.transform(img), label


def build_loaders(batch_size=64, num_workers=2):
    """返回 (train_loader, val_loader, test_loader)，使用验证/训练 transform。"""
    from src.data.transforms import get_train_transform
    train_ds = IndexImageDataset("train", get_train_transform())
    val_ds = IndexImageDataset("val", get_val_transform())
    test_ds = IndexImageDataset("test", get_val_transform())
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


def class_weighted_criterion(device):
    """按训练集类别频率构造加权交叉熵（应对不均衡）。"""
    idx = json.loads((INDEX_DIR / "dataset_index.json").read_text(encoding="utf-8"))
    counts = Counter(l for _, l in idx["train"])
    weights = torch.tensor(
        [1.0 / max(counts[i], 1) for i in range(NUM_CLASSES)], dtype=torch.float32, device=device
    )
    weights = weights * NUM_CLASSES / weights.sum()
    return nn.CrossEntropyLoss(weight=weights)


# ---------- 2) 剪枝核心 ----------
def prune_block(block, pr):
    """就地裁剪一个 BasicBlock 的内部中间通道（conv1.out = conv2.in），保持 out_c 不变。"""
    in_c = block.conv1.in_channels
    out_c = block.conv1.out_channels            # == block.conv2.out_channels
    mid = max(1, int(round(out_c * (1 - pr))))  # 剪枝后的中间通道数

    # conv1 输出通道的 L1 重要性
    w1 = block.conv1.weight.detach()
    imp1 = w1.abs().sum(dim=(1, 2, 3))          # [out_c]
    keep = torch.topk(imp1, mid).indices.sort().values  # [mid]，被选中的通道

    # 新 conv1: in_c -> mid
    new_conv1 = nn.Conv2d(
        in_c, mid, kernel_size=block.conv1.kernel_size,
        stride=block.conv1.stride, padding=block.conv1.padding, bias=False,
    )
    new_conv1.weight.data = w1[keep].clone()

    # 新 bn1: 通道数 mid，参数按 keep 取
    new_bn1 = nn.BatchNorm2d(mid)
    new_bn1.weight = nn.Parameter(block.bn1.weight.data[keep].clone())
    new_bn1.bias = nn.Parameter(block.bn1.bias.data[keep].clone())
    new_bn1.running_mean = block.bn1.running_mean.data[keep].clone()
    new_bn1.running_var = block.bn1.running_var.data[keep].clone()

    # 新 conv2: mid -> out_c，输入维度按 keep 取（输出维度全保留，保持 block 输出不变）
    w2 = block.conv2.weight.detach()
    new_conv2 = nn.Conv2d(
        mid, out_c, kernel_size=block.conv2.kernel_size,
        stride=block.conv2.stride, padding=block.conv2.padding, bias=False,
    )
    new_conv2.weight.data = w2[:, keep, :, :].clone()

    # 替换（bn2 / downsample 不变：block 输出通道 out_c 未动，残差天然对齐）
    block.conv1, block.bn1, block.conv2 = new_conv1, new_bn1, new_conv2


def build_pruned_model(pr):
    """加载 best 模型并就地剪枝所有 BasicBlock 的中间通道。"""
    model = build_resnet18(pretrained=False)
    sd = torch.load(BEST_MODEL_PATH, map_location="cpu")
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
        for block in layer:
            prune_block(block, pr)
    return model.eval()


# ---------- 3) 微调 + 评估 ----------
def finetune(model, epochs, device):
    train_loader, val_loader, _ = build_loaders()
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = class_weighted_criterion(device)
    best_f1, best_sd = 0.0, None
    for ep in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
        # 验证
        model.eval()
        P, T = [], []
        with torch.no_grad():
            for x, y in val_loader:
                P.append(model(x.to(device)).argmax(1).cpu().numpy())
                T.append(y.numpy())
        f1 = f1_score(np.concatenate(T), np.concatenate(P), average="macro")
        print(f"[微调] epoch {ep+1}/{epochs} val_macro_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1, best_sd = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.train()
    return best_sd, best_f1


def evaluate(model, device):
    _, _, test_loader = build_loaders()
    model.to(device).eval()
    P, T = [], []
    with torch.no_grad():
        for x, y in test_loader:
            P.append(model(x.to(device)).argmax(1).cpu().numpy())
            T.append(y.numpy())
    p, t = np.concatenate(P), np.concatenate(T)
    return accuracy_score(t, p), f1_score(t, p, average="macro")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[剪枝] 比例={PRUNE_RATIO}，设备={device}")
    model = build_pruned_model(PRUNE_RATIO)
    best_sd, best_f1 = finetune(model, FINETUNE_EPOCHS, device)

    # best_sd 已是「剪枝后架构」的最优权重，直接载入已剪枝的 model 并保存。
    # 注意：不可再新建一个未剪枝的 build_resnet18 去 load，否则通道数对不上会崩。
    model.load_state_dict(best_sd)

    out = PRUNED_MODEL_PATH
    # 保存时写入 prune_ratio 元数据：推理端据此复现同一剪枝架构（#6），
    # 避免改 PRUNE_RATIO 常量只重跑推理、不重剪枝导致通道数对不上崩溃。
    torch.save({"model": model.state_dict(), "prune_ratio": PRUNE_RATIO}, out)
    print(f"[完成] 剪枝模型已保存: {out} (微调最佳 val_f1={best_f1:.4f}, prune_ratio={PRUNE_RATIO})")

    acc, f1 = evaluate(model, device)
    print(f"[评估] 剪枝后 test 准确率={acc:.4f} | 宏F1={f1:.4f}")
    print(f"[对比] 原 fp32 准确率=0.9927 | 宏F1=0.9923")

    # 体积对比
    fp32 = BEST_MODEL_PATH.stat().st_size
    pr = out.stat().st_size
    print(f"[体积] fp32 {fp32/1e6:.1f}MB -> 剪枝 {pr/1e6:.1f}MB ({(1-pr/fp32)*100:.0f}% 减小)")


if __name__ == "__main__":
    main()
