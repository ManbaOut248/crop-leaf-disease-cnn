# -*- coding: utf-8 -*-
"""
自定义数据集：从 build_index.py 生成的索引（JSON）读取图片路径与标签。
优点：原地引用 F:\实训产品 数据集，不做 54k 图片的拷贝，节省磁盘。
索引格式：{"train": [[path, label], ...], "val": [...], "test": [...]}
label 为 0..37（与 config.CLASS_DIRS 排序一致）。
"""

import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class PlantVillageDataset(Dataset):
    """按索引加载图片与标签，应用传入的 transform。"""

    def __init__(self, samples, transform=None):
        self.samples = samples          # list of [abs_path, label]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # 容错：个别损坏图片回退为纯黑图，避免整批崩溃
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label)


def load_index(index_path):
    """读取索引 JSON，返回 dict。"""
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_loaders(index_path, batch_size, num_workers=4):
    """根据索引构造 train/val/test 三个 DataLoader。"""
    from torch.utils.data import DataLoader
    from src.data.transforms import get_train_transform, get_val_transform

    index = load_index(index_path)
    train_ds = PlantVillageDataset(index["train"], get_train_transform())
    val_ds = PlantVillageDataset(index["val"], get_val_transform())
    test_ds = PlantVillageDataset(index["test"], get_val_transform())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader
