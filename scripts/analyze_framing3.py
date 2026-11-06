# -*- coding: utf-8 -*-
"""标定「拍法」指标（第三轮）：用现成的叶片检测头对画面分块打分。

前两轮用纯图像统计（绿色占比、连通域、距离变换、边缘密度）全部失败 ——
近景与远景分布重叠，无法单阈值切开。原因是整株远景的叶片彼此重叠、绿色同样铺满画面。

本轮改用**已训练好的叶片检测头**（512 维特征上的是否叶片二分类）：
把画面切成 3×3 网格，逐块判断「这块像不像叶片」，统计像叶片的块占比。

    tile_leaf_frac  像叶片的块 / 总块数
    tile_leaf_min   最不像叶片的那块的分数（衡量「有没有明显非叶片区域」）

预期：单片近景 → 9 块几乎都像叶片（frac≈1.0）；
      整株远景 → 含土壤/竹架/天空/果实的块被压低（frac 明显 <1）。
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.predict import Predictor  # noqa: E402

GRID = 3          # 3×3 分块
TILE = 224        # 与模型输入一致


def tile_leaf_scores(pred, img):
    """把图切成 GRID×GRID 块，返回每块的「是叶片」分数列表。"""
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    tw, th = w // GRID, h // GRID
    feats = []
    for i in range(GRID):
        for j in range(GRID):
            box = (j * tw, i * th, (j + 1) * tw, (i + 1) * th)
            tile = img.crop(box).resize((TILE, TILE), Image.BILINEAR)
            x = pred.transform(tile).unsqueeze(0).to(pred.device)
            pooled = {}
            # 与 _infer 相同：从 avgpool 钩子取 512 维池化特征喂给检测头
            hook = pred.model.avgpool.register_forward_hook(
                lambda _m, _i, out: pooled.__setitem__("f", out.flatten(1)))
            try:
                with torch.no_grad():
                    pred.model(x)
            finally:
                hook.remove()
            feats.append(pred.leaf_head.score(pooled["f"]))
    return feats


def summarize(scores):
    s = np.array(scores, float)
    return {"frac": round(float((s >= 0.5).mean()), 3),
            "min": round(float(s.min()), 3),
            "mean": round(float(s.mean()), 3)}


def load_rows(paths, pred, tag):
    out = []
    for k, p in enumerate(paths, 1):
        try:
            im = Image.open(p)
            im.load()
            out.append((p, summarize(tile_leaf_scores(pred, im))))
        except Exception as e:
            print(f"  [跳过] {p.name}: {e}")
        if k % 20 == 0:
            print(f"  {tag} {k}/{len(paths)}", flush=True)
    return out


def describe(tag, rows):
    print(f"\n=== {tag}（n={len(rows)}）===")
    for k in ("frac", "min", "mean"):
        v = np.array([r[1][k] for r in rows], float)
        q = np.percentile(v, [0, 10, 25, 50, 75, 90, 100])
        print(f"  {k:6s} min={q[0]:.2f} p10={q[1]:.2f} p25={q[2]:.2f} 中位={q[3]:.2f} "
              f"p75={q[4]:.2f} p90={q[5]:.2f} max={q[6]:.2f}")


def main():
    pred = Predictor.get_instance()
    if pred.leaf_head is None or not pred.leaf_head.ready:
        print("叶片检测头不可用，无法分块打分")
        return

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    near_paths = [p for d in ("test_images/01_叶片识别", "test_images/02_番茄专项")
                  for p in sorted((ROOT / d).rglob("*")) if p.suffix.lower() in exts]
    near = load_rows(near_paths, pred, "近景")

    rows = list(csv.DictReader(open(ROOT / "data/_downloads/digigreen/annotations.csv", encoding="utf-8")))
    tom = sorted({Path(r["image_file"]).name for r in rows if "tomato" in (r.get("crop") or "").lower()})
    far_paths = [ROOT / "data/_downloads/digigreen/images" / n
                 for n in tom if (ROOT / "data/_downloads/digigreen/images" / n).exists()]
    far = load_rows(far_paths, pred, "远景")

    describe(f"近景（test_images 叶片 {len(near)} 张）", near)
    describe(f"远景（DigiGreen 番茄 {len(far)} 张）", far)

    print("\n逐图对比（前 12 条）：")
    for p, m in near[:6]:
        print(f"  近景 {p.name[:30]:30s} frac={m['frac']:.2f} min={m['min']:.2f} mean={m['mean']:.2f}")
    for p, m in far[:6]:
        print(f"  远景 {p.name[:30]:30s} frac={m['frac']:.2f} min={m['min']:.2f} mean={m['mean']:.2f}")

    # 阈值扫描：按「近景被误判为远景的比例」衡量代价
    print("\n阈值扫描（rule: frac < T 判为「叶片没占满画面」）：")
    for T in (0.60, 0.70, 0.78, 0.85, 0.90):
        n_warn = sum(1 for _, m in near if m["frac"] < T)
        f_warn = sum(1 for _, m in far if m["frac"] < T)
        print(f"  T={T:.2f}  近景误报 {n_warn}/{len(near)} ({n_warn/max(len(near),1)*100:.0f}%)"
              f"  远景命中 {f_warn}/{len(far)} ({f_warn/max(len(far),1)*100:.0f}%)")


if __name__ == "__main__":
    main()
