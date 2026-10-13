# -*- coding: utf-8 -*-
"""
静态 INT8 量化（CPU, qnnpack 后端）—— 任务书 3.3「模型量化」步骤。

实现：加载防泄漏 best_resnet18_38.pth，去掉分类头中的 Dropout（推理时无效，且便于
FX 量化），用训练子集做校准（calibration），经 quantize_fx 静态量化产出 CPU INT8 模型。
输出：models/quantized_resnet18_38.pt（整体量化模型，CPU 推理）。

注意：
- PyTorch 的 INT8 量化推理仅在 CPU 真正生效（qnnpack）。GPU(CUDA) 上跑量化卷积
  基本不支持、无加速。故量化模型专用于「轻量 / CPU 部署」场景，体积约缩至 1/4。
- 不改动原 best_resnet18_38.pth（安全保留 fp32 模型）。
"""
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.ao.quantization import quantize_fx, get_default_qconfig_mapping

from src.config import MODEL_DIR, NUM_CLASSES, INDEX_DIR, BEST_MODEL_PATH
from src.data.transforms import get_val_transform
from src.model.resnet import build_resnet18

CALIB_N = 1024          # 校准集规模（取训练集前 N 张，足够统计激活分布）
CALIB_BATCH = 32


def load_best_state():
    """读取防泄漏 best 模型的权重 state_dict。"""
    p = BEST_MODEL_PATH
    sd = torch.load(p, map_location="cpu")
    return sd["model"] if "model" in sd else sd


def build_quantizable_model():
    """构造便于 FX 量化的模型：分类头改用纯 Linear（去掉 Dropout），并映射 best 权重。"""
    model = build_resnet18(pretrained=False)
    # 原 fc = Sequential(Dropout(0.3), Linear(512,38))；改为纯 Linear
    in_features = model.fc[1].in_features if isinstance(model.fc, nn.Sequential) else model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    # 映射 best 权重：fc.1.weight/bias -> fc.weight/bias（fc.0 是 Dropout，无参数跳过）
    sd = load_best_state()
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("fc.1."):
            new_sd["fc." + k[5:]] = v        # 'fc.1.weight' -> 'fc.weight'
        elif k.startswith("fc.0."):
            continue                         # Dropout 参数，丢弃
        else:
            new_sd[k] = v
    model.load_state_dict(new_sd)
    return model.eval()


def get_calib_tensor(n=CALIB_N):
    """从训练集取前 n 张图，按验证 transform 预处理成 tensor 批次。"""
    idx = json.loads((INDEX_DIR / "dataset_index.json").read_text(encoding="utf-8"))
    paths = [p for p, _ in idx["train"]]
    random.seed(0)
    random.shuffle(paths)
    tf = get_val_transform()
    tensors = []
    for p in paths[:n]:
        try:
            tensors.append(tf(Image.open(p).convert("RGB")))
        except Exception:
            pass
    return torch.stack(tensors)


def load_quantized_model():
    """重建并加载 CPU INT8 量化模型（供推理 / 接入使用）。"""
    model = build_quantizable_model()
    qconfig_mapping = get_default_qconfig_mapping("qnnpack")
    example = torch.randn(1, 3, 224, 224)
    qm = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs=example)
    qm = quantize_fx.convert_fx(qm)
    sd = torch.load(MODEL_DIR / "quantized_resnet18_38.pt", map_location="cpu")
    qm.load_state_dict(sd)
    return qm.eval()


def main():
    model = build_quantizable_model()

    # 1) 配置 CPU qnnpack 默认量化配置
    qconfig_mapping = get_default_qconfig_mapping("qnnpack")
    example = torch.randn(1, 3, 224, 224)
    qmodel = quantize_fx.prepare_fx(model, qconfig_mapping, example_inputs=example)

    # 2) 校准：仅前向，统计各层激活分布以定量化参数（不反传）
    qmodel.eval()
    calib = get_calib_tensor().to("cpu")
    print(f"[校准] 使用 {len(calib)} 张训练图做 INT8 校准...")
    with torch.no_grad():
        for i in range(0, len(calib), CALIB_BATCH):
            qmodel(calib[i:i + CALIB_BATCH])

    # 3) 转换为 INT8 量化模型
    qmodel = quantize_fx.convert_fx(qmodel)
    out = MODEL_DIR / "quantized_resnet18_38.pt"
    torch.save(qmodel.state_dict(), out)   # 保存量化权重(state_dict)，加载时重建结构更稳
    print(f"[完成] 量化模型已保存: {out}")

    # 4) 体积对比
    fp32 = BEST_MODEL_PATH.stat().st_size
    int8 = out.stat().st_size
    print(f"[体积] fp32 {fp32 / 1e6:.1f}MB  ->  INT8 {int8 / 1e6:.1f}MB  "
          f"(减小 {(1 - int8 / fp32) * 100:.0f}%)")


if __name__ == "__main__":
    main()
