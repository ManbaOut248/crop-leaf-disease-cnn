# -*- coding: utf-8 -*-
"""标定「拍法」指标：区分「单片叶片近景」与「整株远景」。

思路（不训练模型，纯图像统计 + 连通域）：
    · 单叶近景：绿色区域连成一整块、几乎铺满画面、四周边缘多为绿色；
    · 整株远景：绿色被打散成几十上百片小叶子，叶片之间夹杂土壤/竹架/天空，
      最大连通块占比低，边缘多为非绿背景。

四个候选指标：
    green_ratio      绿色像素占比
    largest_ratio    最大绿色连通块 / 全图像素
    border_green     图像四边框上绿色像素占比
    n_leaf_blobs     面积≥0.5% 的绿色连通块数量（整株会明显偏多）

用法：分别喂入近景集与远景集，看四个指标的分布能否被单阈值切开。
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]


def framing_metrics(pil_img, size=160):
    """计算拍法指标；统一缩到 size×size 以消除分辨率影响（连通域统计也随之稳定）。"""
    im = pil_img.convert("RGB").resize((size, size), Image.BILINEAR)
    a = np.asarray(im).astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]

    # 超绿指数 ExG：突出植被，抵消亮度差异（G - (R+B)/2 的归一化形式）
    exg = 2 * g - r - b
    # 绿色掩膜：既要 ExG 明显为正，又要绿通道本身占优，抑制灰白/黄褐背景
    mask = (exg > 0.06) & (g > 0.22) & (g > r * 1.02) & (g > b * 1.02)

    total = mask.size
    green_ratio = mask.sum() / total
    if mask.sum() == 0:
        return dict(green_ratio=0.0, largest_ratio=0.0, border_green=0.0, n_leaf_blobs=0)

    # 4 邻域连通域：叶片内部的细小纹理噪点会被并进大块，整株的独立叶片则各自成块
    lab, n = ndimage.label(mask)
    if n == 0:
        return dict(green_ratio=0.0, largest_ratio=0.0, border_green=0.0, n_leaf_blobs=0)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    largest_ratio = float(sizes.max()) / total
    n_leaf_blobs = int((sizes >= 0.005 * total).sum())  # 只数有实际面积的块

    border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    border_green = float(border.mean())

    return dict(green_ratio=round(float(green_ratio), 4),
                largest_ratio=round(largest_ratio, 4),
                border_green=round(border_green, 4),
                n_leaf_blobs=n_leaf_blobs)


def collect(dirpath: Path, limit=None):
    """取目录下的图片（含子目录），返回 [(路径, 指标)]。"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in sorted(dirpath.rglob("*")) if p.suffix.lower() in exts]
    if limit:
        files = files[:limit]
    out = []
    for p in files:
        try:
            im = Image.open(p)
            im.load()  # verify() 抓不住截断图
            out.append((p, framing_metrics(im)))
        except Exception as e:
            print(f"  [跳过] {p.name}: {e}")
    return out


def describe(tag, rows):
    """打印四个指标的分布（分位数），用于判断阈值落点。"""
    if not rows:
        print(f"{tag}: 空")
        return
    keys = ["green_ratio", "largest_ratio", "border_green", "n_leaf_blobs"]
    print(f"\n=== {tag}（n={len(rows)}）===")
    print(f"  {'指标':16s} {'min':>7s} {'p10':>7s} {'p25':>7s} {'中位':>7s} {'p75':>7s} {'p90':>7s} {'max':>7s}")
    for k in keys:
        v = np.array([r[1][k] for r in rows], dtype=float)
        q = np.percentile(v, [0, 10, 25, 50, 75, 90, 100])
        print(f"  {k:16s} " + " ".join(f"{x:7.3f}" for x in q))


def main():
    # 近景集：test_images 的叶片图（PlantDoc 真实照片，单片取景）
    near = collect(ROOT / "test_images/01_叶片识别") + collect(ROOT / "test_images/02_番茄专项")
    # 远景集：DigiGreen 番茄切片（印度农户整株手机实拍），按标注筛出番茄图
    import csv
    rows = list(csv.DictReader(open(ROOT / "data/_downloads/digigreen/annotations.csv", encoding="utf-8")))
    tom_names = {Path(r["image_file"]).name for r in rows if "tomato" in (r.get("crop") or "").lower()}
    far = []
    for name in sorted(tom_names):
        p = ROOT / "data/_downloads/digigreen/images" / name
        if not p.exists():
            continue
        try:
            im = Image.open(p)
            im.load()
            far.append((p, framing_metrics(im)))
        except Exception as e:
            print(f"  [跳过] {name}: {e}")

    describe(f"近景（test_images 叶片 {len(near)} 张）", near)
    describe(f"远景（DigiGreen 番茄 {len(far)} 张）", far)

    # 逐图对比前 8 条，确认指标方向符合预期
    print("\n近景样例：")
    for p, m in near[:8]:
        print(f"  {p.name:34s} green={m['green_ratio']:.2f} largest={m['largest_ratio']:.2f} "
              f"border={m['border_green']:.2f} blobs={m['n_leaf_blobs']}")
    print("\n远景样例：")
    for p, m in far[:8]:
        print(f"  {p.name:34s} green={m['green_ratio']:.2f} largest={m['largest_ratio']:.2f} "
              f"border={m['border_green']:.2f} blobs={m['n_leaf_blobs']}")


if __name__ == "__main__":
    main()
