# -*- coding: utf-8 -*-
"""标定「拍法」指标（第二轮）：改用尺度线索而非颜色占比。

第一轮教训：green_ratio / largest_ratio / border_green 在近景与远景上分布几乎完全重叠
（整株番茄的叶片互相重叠，绿色同样铺满画面且连成一片），**无法区分**。

第二轮改用两个与「被摄物体在画面中的物理尺度」相关、与绿色占比无关的指标：

    dt_rel    绿色区域距离变换的中位数 / 图像边长
              —— 衡量「绿色区域有多粗」。单片叶子填满画面时，内切半径很大；
                 整株远景里每片小叶都是一条细带，内切半径很小。
    edge_den  边缘密度（Sobel 梯度超过阈值的像素占比）
              —— 整株远景含大量叶片轮廓、叶脉、枝干、土壤，边缘稠密；
                 单片近景是一个大平面，边缘稀疏。

二者需在两侧分布上出现明显间隔才可用，否则本轮结论同样是「不可分」。
"""
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
SIZE = 160


def metrics2(pil_img):
    """计算尺度线索指标（统一缩放到 160×160，消除分辨率影响）。"""
    im = pil_img.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    a = np.asarray(im).astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    exg = 2 * g - r - b
    mask = (exg > 0.06) & (g > 0.22)

    # --- 尺度线索 1：绿色区域的内切半径（距离变换） ---
    if mask.sum() > 0:
        dt = ndimage.distance_transform_edt(mask)
        dt_rel = float(np.median(dt[mask])) / SIZE
        dt_p90 = float(np.percentile(dt[mask], 90)) / SIZE
    else:
        dt_rel = dt_p90 = 0.0

    # --- 尺度线索 2：边缘密度（灰度 Sobel 幅值） ---
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    edge_den = float((mag > 0.10).mean())

    return {"dt_rel": round(dt_rel, 4), "dt_p90": round(dt_p90, 4),
            "edge_den": round(edge_den, 4)}


def load_dir(paths):
    out = []
    for p in paths:
        try:
            im = Image.open(p)
            im.load()
            out.append((p, metrics2(im)))
        except Exception as e:
            print(f"  [跳过] {p.name}: {e}")
    return out


def describe(tag, rows):
    keys = ["dt_rel", "dt_p90", "edge_den"]
    print(f"\n=== {tag}（n={len(rows)}）===")
    print(f"  {'指标':10s} {'min':>7s} {'p10':>7s} {'p25':>7s} {'中位':>7s} {'p75':>7s} {'p90':>7s} {'max':>7s}")
    for k in keys:
        v = np.array([r[1][k] for r in rows], float)
        q = np.percentile(v, [0, 10, 25, 50, 75, 90, 100])
        print(f"  {k:10s} " + " ".join(f"{x:7.3f}" for x in q))


def band_overlap(near, far, key):
    """用「近景 p10」与「远景 p90」看两分布是否可被单阈值切开。"""
    n = np.array([r[1][key] for r in near], float)
    f = np.array([r[1][key] for r in far], float)
    print(f"  {key}: 近景中位 {np.median(n):.3f} vs 远景中位 {np.median(f):.3f} "
          f"| 近景p10 {np.percentile(n,10):.3f} / 远景p90 {np.percentile(f,90):.3f}")


def main():
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    near_paths = [p for d in ("test_images/01_叶片识别", "test_images/02_番茄专项")
                  for p in sorted((ROOT / d).rglob("*")) if p.suffix.lower() in exts]
    near = load_dir(near_paths)

    rows = list(csv.DictReader(open(ROOT / "data/_downloads/digigreen/annotations.csv", encoding="utf-8")))
    tom = sorted({Path(r["image_file"]).name for r in rows if "tomato" in (r.get("crop") or "").lower()})
    far = load_dir([ROOT / "data/_downloads/digigreen/images" / n
                    for n in tom if (ROOT / "data/_downloads/digigreen/images" / n).exists()])

    describe(f"近景（test_images 叶片 {len(near)} 张）", near)
    describe(f"远景（DigiGreen 番茄 {len(far)} 张）", far)
    print("\n可分性：")
    for k in ("dt_rel", "dt_p90", "edge_den"):
        band_overlap(near, far, k)


if __name__ == "__main__":
    main()
