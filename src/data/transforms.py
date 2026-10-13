# -*- coding: utf-8 -*-
"""
图像预处理与数据增强。
- 训练集：随机裁剪/翻转/色彩抖动 + ImageNet 归一化（应对样本不均衡与过拟合）
- 验证/测试集：中心裁剪 + 归一化（不做随机增强，保证评估一致）
必须与原 ImageNet 预训练权重使用相同的 MEAN/STD。
"""

from torchvision import transforms
from src.config import IMAGE_SIZE, MEAN, STD


def get_train_transform():
    """训练集增强变换。"""
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_val_transform():
    """验证/测试集变换（确定性）。"""
    return transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE * 1.14)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
