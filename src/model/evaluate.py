# -*- coding: utf-8 -*-
"""
在测试集上评估最佳模型，产出评估报告（对齐任务书任务 3 交付物）：
- 整体准确率、宏平均/加权平均 F1
- 各类 Precision / Recall / F1（分类报告）
- 混淆矩阵图（PNG）
- 置信度校准：校准前/后 ECE（Expected Calibration Error）
- 温度缩放（Temperature Scaling）：在验证集拟合温度 T，供推理端使用（缓解过自信）
- 错误样本清单 CSV（image_path, true_label, pred_label, calibrated_confidence），供人工复核根因

输出文件（均位于 outputs/）：
- eval_report.json        整体 + 校准 + 每类指标（per_class，已修复 key 为空字典的 bug）
- classification_report.txt  文本版分类报告
- confusion_matrix.png    混淆矩阵图（中文标签）
- calibration.json        温度 T 与 ECE 前后对照（predict.py 读取）
- error_samples.csv       错误样本清单
"""

import csv
import json
from pathlib import Path
import os
import glob
import warnings

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# 注册中文字体，确保混淆矩阵图的中文标签正常显示（默认 DejaVu Sans 缺 CJK 字形）。
# 旧实现硬编码 Windows 绝对路径（C:/Windows/Fonts/...），换平台/无该字体会直接出豆腐块。
# 现改为三级回退：已知绝对路径 → 常见系统字体目录 glob 扫描 → matplotlib 已注册字体名匹配，
# 任一命中即用；全失败仅告警（图仍生成，标签可能为方块），绝不抛异常。
_CJK_KEYWORDS = (
    "hei", "yahei", "song", "simsun", "simhei", "ming",
    "wqy", "noto sans cjk", "noto serif cjk", "source han",
    "pingfang", "stheiti", "arphic", "uming", "ukai",
    "microsoft yahei", "cjk",
)


def setup_cjk_font():
    """注册一款支持中文的字体，返回生效字体名（str）或 None（未找到，仅告警）。"""
    # 1) 已知绝对路径（覆盖 Win / Linux / macOS 常见中文字体，均为 CJK 字体，命中即用）
    curated = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for fp in curated:
        if os.path.exists(fp):
            try:
                matplotlib.font_manager.fontManager.addfont(fp)
                name = matplotlib.font_manager.FontProperties(fname=fp).get_name()
                plt.rcParams["font.family"] = name
                plt.rcParams["axes.unicode_minus"] = False
                return name
            except Exception:
                continue

    # 2) 常见系统字体目录通配扫描（兜底：用户自定义 / 其他发行版），仅采纳 CJK 名
    patterns = [
        "C:/Windows/Fonts/*",
        "/usr/share/fonts/**/*",
        "/usr/local/share/fonts/**/*",
        "/System/Library/Fonts/**/*",
        str(Path.home() / ".fonts" / "**" / "*"),
    ]
    for pat in patterns:
        for fp in glob.glob(pat, recursive=True):
            if not fp.lower().endswith((".ttf", ".ttc", ".otf")):
                continue
            try:
                matplotlib.font_manager.fontManager.addfont(fp)
                name = matplotlib.font_manager.FontProperties(fname=fp).get_name()
            except Exception:
                continue
            if any(k in name.lower() for k in _CJK_KEYWORDS):
                plt.rcParams["font.family"] = name
                plt.rcParams["axes.unicode_minus"] = False
                return name

    # 3) 扫描 matplotlib 已加载字体列表，找 CJK 名（最后兜底，避免重复注册）
    for f in matplotlib.font_manager.fontManager.ttflist:
        if any(k in f.name.lower() for k in _CJK_KEYWORDS):
            plt.rcParams["font.family"] = f.name
            plt.rcParams["axes.unicode_minus"] = False
            return f.name

    warnings.warn(
        "未找到支持中文的字体，混淆矩阵等图表的中文标签可能显示为方块（tofu）。"
        "请安装中文字体（如 Windows 黑体 / Linux 文泉驿 / macOS 苹方）。"
    )
    plt.rcParams["axes.unicode_minus"] = False
    return None


setup_cjk_font()

from src.config import (
    MODEL_DIR, INDEX_DIR, REPORT_DIR, CALIBRATION_PATH, NUM_CLASSES, BATCH_SIZE,
    NUM_WORKERS, CLASS_DIRS, class_name_zh, BEST_MODEL_PATH,
)
from src.data.datasets import build_loaders, load_index
from src.model.resnet import build_resnet18


def compute_ece(logits, labels, n_bins=10):
    """期望校准误差（Expected Calibration Error）。

    logits: (N, C); labels: (N,) 长整型。
    按置信度（softmax 最大概率）分箱，加权求和 |箱平均置信度 - 箱平均准确率|。
    值越接近 0 表示校准越好；过自信模型（高置信但低准确）ECE 偏高。
    """
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    acc = (pred == labels).float()
    ece = 0.0
    n_total = len(labels)
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (conf > lo) & (conf <= hi)
        n = int(mask.sum().item())
        if n > 0:
            ece += (conf[mask].mean() - acc[mask].mean()).abs().item() * (n / n_total)
    return ece


def fit_temperature(val_logits, val_labels):
    """温度缩放：在验证集上网格搜索最小化 NLL 的温度 T。

    过自信模型（softmax 集中在错误类）需要 T > 1 让分布更平滑、校准更好。
    返回最优 T。
    """
    val_logits = val_logits.float()
    best_T, best_nll = 1.0, float("inf")
    for T in [0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        scaled = val_logits / T
        logp = F.log_softmax(scaled, dim=1)
        nll = F.nll_loss(logp, val_labels.long()).item()
        if nll < best_nll:
            best_nll, best_T = nll, T
    return best_T


@torch.no_grad()
def collect_logits(model, loader, device):
    """遍历 loader，收集 (logits, labels)。推理统一 fp32，不使用 AMP。"""
    all_logits, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_loaders(
        INDEX_DIR / "dataset_index.json", BATCH_SIZE, NUM_WORKERS
    )
    # 测试集路径顺序与 test_loader 输出一致（shuffle=False），用于错误样本 CSV 对齐
    index = load_index(INDEX_DIR / "dataset_index.json")
    test_samples = index["test"]

    model = build_resnet18(pretrained=False).to(device)
    best = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(best["model"])
    model.eval()

    # ===== 收集 logits（fp32，推理与评估一致）=====
    test_logits, test_labels = collect_logits(model, test_loader, device)
    val_logits, val_labels = collect_logits(model, val_loader, device)

    test_probs = torch.softmax(test_logits, dim=1)
    all_preds = test_probs.argmax(dim=1).numpy()
    all_labels = test_labels.numpy()

    acc = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    # 中文标签
    target_names = [f"{i}-{class_name_zh(c)}" for i, c in enumerate(CLASS_DIRS)]
    report_txt = classification_report(
        all_labels, all_preds, labels=list(range(NUM_CLASSES)),
        target_names=target_names, digits=4, zero_division=0,
    )
    print(report_txt)
    print(f"整体准确率: {acc:.4f} | 宏平均F1: {macro_f1:.4f} | 加权平均F1: {weighted_f1:.4f}")

    # 保存文本报告
    with open(REPORT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_txt)
        f.write(f"\n整体准确率: {acc:.4f}\n宏平均F1: {macro_f1:.4f}\n加权平均F1: {weighted_f1:.4f}\n")

    # 混淆矩阵图
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("混淆矩阵 (Confusion Matrix) - 38 类", fontsize=16)
    ax.set_xlabel("预测类别", fontsize=12)
    ax.set_ylabel("真实类别", fontsize=12)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(target_names, rotation=90, fontsize=7)
    ax.set_yticklabels(target_names, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)
    print(f"混淆矩阵图已保存: {REPORT_DIR / 'confusion_matrix.png'}")

    # ===== 置信度校准（ECE + Temperature Scaling）=====
    ece_before = compute_ece(test_logits, test_labels)
    # 温度 T 在验证集上拟合，再应用到测试集（避免污染测试指标）
    T = fit_temperature(val_logits, val_labels)
    cal_test_logits = test_logits / T
    ece_after = compute_ece(cal_test_logits, test_labels)
    cal_probs = torch.softmax(cal_test_logits, dim=1)
    cal_conf = cal_probs.max(dim=1).values.numpy()
    print(f"校准: T={T:.2f} | ECE(校准前)={ece_before:.4f} | ECE(校准后)={ece_after:.4f}")

    # 保存校准参数（供 predict.py 推理端使用）
    cal_payload = {
        "temperature": round(float(T), 4),
        "ece_before": round(float(ece_before), 4),
        "ece_after": round(float(ece_after), 4),
        "note": "Temperature Scaling 温度 T 在验证集拟合、测试集评估；predict.py 推理端用 T 缩放 logits。",
    }
    with open(CALIBRATION_PATH, "w", encoding="utf-8") as f:
        json.dump(cal_payload, f, ensure_ascii=False, indent=2)
    print(f"校准参数已保存: {CALIBRATION_PATH}")

    # ===== JSON 报告（修复 per_class bug：output_dict 的 key 是 target_names，不是 "0"）=====
    report_json = {
        "overall_accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "calibration": {
            "temperature": round(float(T), 4),
            "ece_before": round(float(ece_before), 4),
            "ece_after": round(float(ece_after), 4),
        },
        "per_class": {},
    }
    per_class = classification_report(
        all_labels, all_preds, labels=list(range(NUM_CLASSES)),
        target_names=target_names, digits=4, zero_division=0, output_dict=True,
    )
    for i, c in enumerate(CLASS_DIRS):
        tn = target_names[i]  # 关键修复：output_dict 的 key 是 target_names（"0-苹果黑星病"），不是 "0"
        if tn in per_class:
            m = per_class[tn]
            report_json["per_class"][tn] = {
                "en": c,
                "zh": class_name_zh(c),
                "precision": round(float(m["precision"]), 4),
                "recall": round(float(m["recall"]), 4),
                # 注意：sklearn 的 classification_report(output_dict=True) 每类键名为 "f1-score"（带连字符）
                "f1": round(float(m["f1-score"]), 4),
                "support": int(m["support"]),
            }
    with open(REPORT_DIR / "eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report_json, f, ensure_ascii=False, indent=2)
    print(f"JSON 报告已保存: {REPORT_DIR / 'eval_report.json'} "
          f"(per_class 类别数={len(report_json['per_class'])}/{NUM_CLASSES})")

    # ===== 错误样本清单 CSV（供人工复核根因）=====
    err_path = REPORT_DIR / "error_samples.csv"
    n_err = 0
    with open(err_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "true_label", "pred_label", "calibrated_confidence"])
        for i in range(len(all_preds)):
            if all_preds[i] != all_labels[i]:
                n_err += 1
                w.writerow([
                    test_samples[i][0],
                    class_name_zh(CLASS_DIRS[int(all_labels[i])]),
                    class_name_zh(CLASS_DIRS[int(all_preds[i])]),
                    round(float(cal_conf[i]), 4),
                ])
    print(f"错误样本清单已保存: {err_path} (共 {n_err} 条错误 / 测试集 {len(all_preds)} 张)")


if __name__ == "__main__":
    main()
