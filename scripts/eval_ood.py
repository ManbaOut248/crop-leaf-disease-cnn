# -*- coding: utf-8 -*-
"""评估「非叶片输入拒识」各指标的可分离性，并标定推荐阈值。

背景：
    模型把动漫插画判成「番茄叶霉病 96.87%」。根因是 38 类闭集 softmax
    没有"以上都不是"选项，且对分布外输入过度自信（实测插画 MSP=0.968、
    随机噪声 MSP=0.975，与真实叶片 0.994 几乎重叠）。

本脚本做的事：
    1. 取「分布内」样本（data/split/test 的真实叶片）与「分布外」样本
       （data/ood_set/ 下各类非叶片图）；
    2. 对每张图计算三个候选拒识指标：Energy / MaxLogit / MSP；
    3. 计算 AUROC，并做「操作点扫描」—— 列出不同误杀率下能达到的拦截率；
    4. 给出保守 / 平衡两档推荐操作点，并分来源报告拦截率；
    5. 输出 outputs/ood_eval.json 供文档与后续落地引用。

关于误杀率的取舍（重要）：
    两类错误的代价并不对称 ——
      · 误杀（把真实叶片判成"非叶片"）：用户重新拍一张即可，代价低；
      · 漏放（把非叶片判成某种病害）：农户可能据此施药，代价高。
    因此操作点可以偏向"多拦一点"，但要控制在用户可接受的范围内。

用法：
    py -3.12 scripts/eval_ood.py [--n-in 500]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.predict import Predictor  # noqa: E402

OOD_ROOT = PROJECT_ROOT / "data" / "ood_set"
TEST_ROOT = PROJECT_ROOT / "data" / "split" / "test"
OUT_JSON = PROJECT_ROOT / "outputs" / "ood_eval.json"

SCAN_FPRS = (0.01, 0.02, 0.05, 0.10, 0.20)


def list_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)


def collect(paths, model, tf, device, T):
    """计算三个拒识指标：energy（越大越像 OOD）、maxlogit 与 msp（越小越像 OOD）。"""
    energy, maxlogit, msp = [], [], []
    skipped = 0
    for p in paths:
        try:
            with Image.open(p) as im:
                x = tf(im.convert("RGB")).unsqueeze(0).to(device)
        except Exception:  # noqa: BLE001 —— 个别坏图不应中断评估
            skipped += 1
            continue
        with torch.no_grad():
            logits = model(x)[0] / T
        energy.append(-torch.logsumexp(logits, dim=0).item())
        maxlogit.append(logits.max().item())
        msp.append(torch.softmax(logits, dim=0).max().item())
    return (
        np.array(energy, dtype=np.float64),
        np.array(maxlogit, dtype=np.float64),
        np.array(msp, dtype=np.float64),
        skipped,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-in", type=int, default=500, help="分布内样本数（真实叶片）")
    args = ap.parse_args()

    predictor = Predictor()
    model, tf, device, T = (
        predictor.model, predictor.transform, predictor.device, predictor.temperature
    )
    print(f"模型已加载 | device={device} | 温度 T={T}\n")

    # ---------- 1. 收集样本 ----------
    in_files = list_images(TEST_ROOT)
    rng = np.random.default_rng(0)
    if len(in_files) > args.n_in:
        pick = rng.choice(len(in_files), size=args.n_in, replace=False)
        in_files = [in_files[i] for i in sorted(pick)]
    print(f"分布内（真实叶片）：{len(in_files)} 张")

    groups = {}
    if OOD_ROOT.is_dir():
        for d in sorted(p for p in OOD_ROOT.iterdir() if p.is_dir()):
            files = list_images(d)
            if files:
                groups[d.name] = files
    if not groups:
        print("！！ data/ood_set 为空，请先运行 scripts/build_ood_set.py")
        return
    for k, v in groups.items():
        print(f"分布外 · {k:<11}：{len(v)} 张")
    print(f"分布外合计：{sum(len(v) for v in groups.values())} 张\n")

    # ---------- 2. 推理 ----------
    print("推理中…")
    in_energy, in_maxlogit, in_msp, _ = collect(in_files, model, tf, device, T)
    per_group, ood_energy_all, ood_maxlogit_all, ood_msp_all = {}, [], [], []
    for k, files in groups.items():
        e, ml, m, skipped = collect(files, model, tf, device, T)
        per_group[k] = {"n": int(len(e)), "energy": e, "maxlogit": ml, "msp": m, "skipped": skipped}
        ood_energy_all.append(e)
        ood_maxlogit_all.append(ml)
        ood_msp_all.append(m)
    ood_energy = np.concatenate(ood_energy_all)
    ood_maxlogit = np.concatenate(ood_maxlogit_all)
    ood_msp = np.concatenate(ood_msp_all)

    # ---------- 3. AUROC ----------
    # 统一方向：分数越大越像「分布外」
    metrics = {
        "energy": (in_energy, ood_energy, "分布的 energy 越大越像非叶片"),
        "neg_maxlogit": (-in_maxlogit, -ood_maxlogit, "maxlogit 越小越像非叶片"),
        "neg_msp": (-in_msp, -ood_msp, "置信度越低越像非叶片"),
    }
    y = np.concatenate([np.zeros(len(in_energy)), np.ones(len(ood_energy))])
    aurocs = {}
    for name, (a, b, _) in metrics.items():
        aurocs[name] = float(roc_auc_score(y, np.concatenate([a, b])))

    print("=" * 70)
    print("指标可分离性（AUROC：0.5 = 完全不可分，1.0 = 完美分离）")
    print("=" * 70)
    for name, auc in sorted(aurocs.items(), key=lambda x: -x[1]):
        flag = "★ 可用" if auc >= 0.9 else ("~ 一般" if auc >= 0.75 else "✗ 不可用")
        print(f"  {name:<14} AUROC = {auc:.4f}   {flag}")

    # ---------- 4. 操作点扫描 ----------
    print()
    print("=" * 70)
    print("操作点扫描：不同误杀率下能拦下多少非叶片输入")
    print("=" * 70)
    scan = {}
    for name, (a, b, _) in metrics.items():
        scan[name] = {}
        print(f"\n  指标：{name}")
        for fpr in SCAN_FPRS:
            thr = float(np.quantile(a, 1 - fpr))
            block = float((b > thr).mean())
            scan[name][f"{fpr:.2f}"] = {"threshold": thr, "block_rate": block}
            bar = "█" * int(round(block * 30))
            print(f"    误杀 {fpr * 100:>4.0f}%  →  拦截 {block * 100:5.1f}%  {bar}")

    # ---------- 5. 推荐操作点 ----------
    best_name = max(aurocs, key=aurocs.get)
    recommendations = {}
    for label, fpr in (("conservative", 0.02), ("balanced", 0.05)):
        rec = scan[best_name][f"{fpr:.2f}"]
        predictions = []
        for k, d in per_group.items():
            src = d["energy"] if best_name == "energy" else (
                -d["maxlogit"] if best_name == "neg_maxlogit" else -d["msp"]
            )
            predictions.append((k, float((src > rec["threshold"]).mean())))
        recommendations[label] = {
            "metric": best_name,
            "max_fpr": fpr,
            "threshold": rec["threshold"],
            # 阈值直接取自分布内分数的 (1-fpr) 分位，故误杀率按定义即 fpr
            "false_reject_rate": fpr,
            "overall_block_rate": rec["block_rate"],
            "per_group": dict(predictions),
        }

    print()
    print("=" * 70)
    print(f"推荐操作点（最优指标：{best_name}，AUROC {aurocs[best_name]:.4f}）")
    print("=" * 70)
    for label, cn in (("conservative", "保守"), ("balanced", "平衡")):
        r = recommendations[label]
        print(f"\n  【{cn}】误杀率 ≈ {r['false_reject_rate'] * 100:.0f}%  "
              f"→  阈值 = {r['threshold']:+.3f}  |  总拦截率 = {r['overall_block_rate'] * 100:.1f}%")
        for k, rate in sorted(r["per_group"].items(), key=lambda x: x[1]):
            bar = "█" * int(round(rate * 24))
            print(f"      {k:<11} {rate * 100:6.2f}%  {bar}")

    # ---------- 6. 存盘 ----------
    report = {
        "n_in": int(len(in_energy)),
        "n_ood": int(len(ood_energy)),
        "temperature": float(T),
        "auroc": aurocs,
        "group_sizes": {k: v["n"] for k, v in per_group.items()},
        "scan": scan,
        "best_metric": best_name,
        "recommendations": recommendations,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写入：{OUT_JSON}")


if __name__ == "__main__":
    main()
