# -*- coding: utf-8 -*-
"""
ResNet18 迁移学习骨干：加载 ImageNet 预训练权重，替换末端全连接层为 38 分类。
冻结浅层、微调末端策略由 train.py 控制。
"""

import torch.nn as nn
from torchvision import models
from src.config import NUM_CLASSES


def build_resnet18(pretrained=True, freeze_backbone=False):
    """构造 38 分类 ResNet18。"""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    # 替换分类头
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, NUM_CLASSES),
    )
    if freeze_backbone:
        # 仅训练分类头（warmup 阶段可选）
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
    return model
