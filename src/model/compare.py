# -*- coding: utf-8 -*-
"""
三后端对比评测（D-4）：fp32 / INT8量化 / 结构化剪枝。

统一在 CPU 上评测，保证延迟对比公平（压缩的目标本就是 CPU/边缘部署）：
  - 精度：test 集 准确率 + 宏F1（与诚实评估口径一致，不使用 AMP）
  - 体积：模型文件大小（MB）及相对 fp32 的缩减比例
  - 延迟：固定 dummy 输入(1,3,224,224) 预热后平均单图推理耗时(ms)

产出：outputs/compare.json（供 docs/06_模型压缩对比.md 引用）。
"""
import json
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from src.config import MODEL_DIR, NUM_CLASSES, PRUNED_MODEL_PATH, QUANTIZED_MODEL_PATH, BEST_MODEL_PATH
from src.data.transforms import get_val_transform
from src.model.resnet import build_resnet18
from src.model.quantize import load_quantized_model, load_best_state
from src.model.prune import build_loaders, evaluate, build_pruned_model, PRUNE_RATIO


def build_fp32():
    model = build_resnet18(pretrained=False)
    sd = load_best_state()
    model.load_state_dict(sd)
    return model.cpu().eval()


def build_pruned():
    if not PRUNED_MODEL_PATH.exists():
        raise FileNotFoundError(f"未找到剪枝模型：{PRUNED_MODEL_PATH}")
    # 同 predict.py：必须用 build_pruned_model 复现「剪枝后架构」，再载微调权重。
    # 比例优先取 checkpoint 元数据，缺失回退常量（见 #6）。
    ckpt = torch.load(PRUNED_MODEL_PATH, map_location="cpu")
    pr = ckpt.get("prune_ratio", PRUNE_RATIO)
    model = build_pruned_model(pr)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    return model.cpu().eval()


def build_quantized():
    return load_quantized_model().cpu().eval()


def measure_latency(model, n_warm=5, n_run=50):
    """固定 dummy 输入测平均单图推理延迟(ms)。"""
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        for _ in range(n_warm):
            model(x)
        t0 = time.perf_counter()
        for _ in range(n_run):
            model(x)
        dt = (time.perf_counter() - t0) / n_run
    return dt * 1000.0


def file_size_mb(p):
    return p.stat().st_size / 1e6 if p.exists() else 0.0


def main():
    device = torch.device("cpu")
    rows = []

    # ---- fp32 ----
    m = build_fp32()
    acc, f1 = evaluate(m, device)
    rows.append({
        "backend": "fp32", "desc": "原始防泄漏 best（fp32）",
        "size_mb": file_size_mb(BEST_MODEL_PATH),
        "acc": round(acc, 4), "macro_f1": round(f1, 4),
        "latency_ms": round(measure_latency(m), 2),
    })
    del m

    # ---- 量化 ----
    if QUANTIZED_MODEL_PATH.exists():
        m = build_quantized()
        acc, f1 = evaluate(m, device)
        rows.append({
            "backend": "quantized", "desc": "INT8 静态量化（qnnpack/CPU）",
            "size_mb": file_size_mb(QUANTIZED_MODEL_PATH),
            "acc": round(acc, 4), "macro_f1": round(f1, 4),
            "latency_ms": round(measure_latency(m), 2),
        })
        del m

    # ---- 剪枝 ----
    if PRUNED_MODEL_PATH.exists():
        m = build_pruned()
        acc, f1 = evaluate(m, device)
        rows.append({
            "backend": "pruned", "desc": "结构化通道剪枝（fp32，20% 中间通道）",
            "size_mb": file_size_mb(PRUNED_MODEL_PATH),
            "acc": round(acc, 4), "macro_f1": round(f1, 4),
            "latency_ms": round(measure_latency(m), 2),
        })
        del m

    # 以 fp32 为基准算相对缩减
    base = next(r for r in rows if r["backend"] == "fp32")
    for r in rows:
        r["size_reduce_pct"] = round((1 - r["size_mb"] / base["size_mb"]) * 100, 1)
        r["speedup_x"] = round(base["latency_ms"] / r["latency_ms"], 2) if r["latency_ms"] > 0 else None

    out = {"device": "cpu", "backends": rows}
    (MODEL_DIR.parent / "outputs" / "compare.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 打印表格
    print(f"{'后端':<10}{'体积MB':>9}{'缩减%':>8}{'准确率':>9}{'宏F1':>9}{'延迟ms':>10}{'加速x':>8}")
    for r in rows:
        print(f"{r['backend']:<10}{r['size_mb']:>9.1f}{r['size_reduce_pct']:>7.0f}%"
              f"{r['acc']:>9.4f}{r['macro_f1']:>9.4f}{r['latency_ms']:>10.2f}{str(r['speedup_x']):>8}")
    print(f"\n[完成] 对比结果已写入: outputs/compare.json")


if __name__ == "__main__":
    main()
