# -*- coding: utf-8 -*-
"""OOD 阈值重标定：用**真实场景照片**（而非实验室图）扫描 MSP 操作点。

背景
----
原阈值 `OOD_MSP_THRESHOLD = 0.98` 是在 PlantVillage（实验室黑底居中图）上标定的：
误杀 6.2%、拦截 82.9%。但实测真实田间照片（PlantDoc）误杀率高达 53.8% ——
因为真实照片背景复杂，模型置信度整体偏低，MSP 判据大量误伤真叶片。

本脚本把「分布内」正样本换成真实照片，重新扫描操作点，看能否找到一个
在真实场景下可用的阈值。

用法
----
    py -3.12 scripts/tune_ood_threshold.py                              # 用 PlantVillage 测试集（原口径）
    py -3.12 scripts/tune_ood_threshold.py --in-dir outputs/realtest    # 用真实田间照片
    py -3.12 scripts/tune_ood_threshold.py --in-dir outputs/realtest --max-in 100
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from src.config import OOD_CHROMA_THRESHOLD  # noqa: E402
from src.model.predict import Predictor  # noqa: E402

IMGS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OOD_ROOT = Path("data/ood_set")


def list_images(root, limit=None, per_dir=None):
    root = Path(root)
    files = []
    if not root.is_dir():
        return files
    # 若下有子目录则按目录均匀取样，保证类别均衡
    subs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    if subs:
        for d in subs:
            fs = sorted(p for p in d.rglob("*") if p.suffix.lower() in IMGS)
            if per_dir:
                fs = fs[:per_dir]
            files += fs
    else:
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMGS)
    if limit and len(files) > limit:
        rng = np.random.default_rng(42)
        files = list(rng.choice(files, limit, replace=False))
    return files


def score(p, files):
    rows = []
    for i, f in enumerate(files, 1):
        try:
            with Image.open(f) as im:
                img = im.convert("RGB")
                _, ood = p.predict_image_with_ood(img, top_k=1)
            rows.append((ood["leaf_prob"], ood["mean_chroma"]))
        except Exception:
            continue
        if i % 100 == 0:
            print(f"    {i}/{len(files)} ...", flush=True)
    return np.array(rows) if rows else np.zeros((0, 2))


ap = argparse.ArgumentParser()
ap.add_argument("--in-dir", default="data/split/test", help="分布内（真实叶片）目录")
ap.add_argument("--max-in", type=int, default=400, help="分布内抽样上限")
ap.add_argument("--per-dir", type=int, default=0, help="分布内每个子目录取样上限（0=不限制）")
args = ap.parse_args()

in_files = list_images(args.in_dir, limit=args.max_in or None,
                       per_dir=args.per_dir or None)
ood_files = list_images(OOD_ROOT)
print(f"分布内 {args.in_dir}: {len(in_files)} 张")
print(f"分布外 {OOD_ROOT}: {len(ood_files)} 张")
print()

p = Predictor.get_instance()
print("推理：分布内 ...")
IN = score(p, in_files)
print("推理：分布外 ...")
OOD = score(p, ood_files)
if len(IN) == 0 or len(OOD) == 0:
    print("样本不足，退出")
    sys.exit(1)

in_msp, in_ch = IN[:, 0], IN[:, 1]
ood_msp, ood_ch = OOD[:, 0], OOD[:, 1]

print()
print("=" * 74)
print(f"MSP 分布对比")
print("=" * 74)
for name, a in (("分布内(真实叶片)", in_msp), ("分布外(非叶片)", ood_msp)):
    q = np.percentile(a, [5, 25, 50, 75, 95])
    print(f"  {name:<16} P5={q[0]:.4f}  P25={q[1]:.4f}  中位={q[2]:.4f}  P75={q[3]:.4f}  P95={q[4]:.4f}")

print()
print("=" * 74)
print("MSP 阈值扫描（判据：MSP < T）")
print("=" * 74)
print(f"  {'阈值T':>7}  {'真实叶片误杀':>12}  {'非叶片拦截':>10}  {'平衡点':>8}")
best = None
for t in [0.50, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]:
    frr = float((in_msp < t).mean()) * 100
    blk = float((ood_msp < t).mean()) * 100
    bal = blk - frr          # 拦截率 - 误杀率，越大越好
    mark = ""
    if best is None or bal > best[1]:
        best = (t, bal, frr, blk)
        mark = " ★"
    print(f"  {t:>7.3f}  {frr:>11.1f}%  {blk:>9.1f}%  {bal:>7.1f}{mark}")

print()
print("=" * 74)
print(f"组合判据（MSP < T  或  色度 < {OOD_CHROMA_THRESHOLD}）")
print("=" * 74)
print(f"  {'阈值T':>7}  {'真实叶片误杀':>12}  {'非叶片拦截':>10}")
for t in [0.50, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98]:
    frr = float(((in_msp < t) | (in_ch < OOD_CHROMA_THRESHOLD)).mean()) * 100
    blk = float(((ood_msp < t) | (ood_ch < OOD_CHROMA_THRESHOLD)).mean()) * 100
    print(f"  {t:>7.3f}  {frr:>11.1f}%  {blk:>9.1f}%")

print()
print("=" * 74)
print("色度判据的独立贡献")
print("=" * 74)
print(f"  真实叶片被色度误伤: {int((in_ch < OOD_CHROMA_THRESHOLD).sum())}/{len(in_ch)}"
      f"  ({(in_ch < OOD_CHROMA_THRESHOLD).mean()*100:.1f}%)")
print(f"  非叶片被色度拦下  : {int((ood_ch < OOD_CHROMA_THRESHOLD).sum())}/{len(ood_ch)}"
      f"  ({(ood_ch < OOD_CHROMA_THRESHOLD).mean()*100:.1f}%)")
print()
print(f"注：当前线上阈值 = 0.98（★ 为本次样本下「拦截率−误杀率」最大的操作点）")
