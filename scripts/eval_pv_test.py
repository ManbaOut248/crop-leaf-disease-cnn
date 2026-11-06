# -*- coding: utf-8 -*-
"""
在 PlantVillage 标准 test split 上评估任意权重（实验室域精度守门）。

为什么需要
---------------------------------------------------------------
换部署权重的前提是「实验室域精度不掉」。训练日志里的 PV_val 来自混合索引的
val 划分（4921 张），与任务书口径（99.27%，来自 dataset_index.json 的 test split）
不是同一批图。部署前必须用同一口径复核。

用法
    python backups/_pv_test.py models/finetuned_v2_w4.pth models/finetuned_v2_ctrl.pth
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from src.config import BATCH_SIZE, NUM_CLASSES
from src.model.resnet import build_resnet18
from src.data.datasets import PlantVillageDataset, load_index
from src.data.transforms import get_val_transform

INDEX = ROOT / "outputs" / "index" / "dataset_index.json"


def main(weight_paths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_items = load_index(INDEX)["test"]
    loader = DataLoader(PlantVillageDataset(test_items, get_val_transform()),
                        batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=8, pin_memory=True)
    print(f"PlantVillage test: {len(test_items)} 张 | 设备 {device}")
    print(f"{'权重':<34}{'Top-1':>9}{'macro-F1':>11}")
    print("-" * 56)
    for wp in weight_paths:
        model = build_resnet18(pretrained=False).to(device)
        ck = torch.load(wp, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"] if "model" in ck else ck)
        model.eval()
        correct, total = 0, 0
        preds_all, labels_all = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device, non_blocking=True)
                preds = model(imgs).argmax(1).cpu()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                preds_all.extend(preds.tolist())
                labels_all.extend(labels.tolist())
        f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
        print(f"{Path(wp).name:<34}{correct / total:>9.2%}{f1:>11.4f}")


if __name__ == "__main__":
    args = sys.argv[1:] or ["models/best_resnet18_38.pth"]
    main([str(Path(a)) if Path(a).exists() else str(ROOT / a) for a in args])
