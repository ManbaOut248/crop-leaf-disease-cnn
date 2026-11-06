# -*- coding: utf-8 -*-
"""
实验：不同裁剪策略对真实域识别率的影响（不改产品代码，纯离线对比）。

背景
----
部署时的预处理是 `Resize(IMAGE_SIZE*1.14) + CenterCrop(IMAGE_SIZE)`（见 src/data/transforms.py）。
`Resize` 只给一个参数时按**短边**等比缩放，随后中心裁 224。对 4:3 的照片，
等效于只保留原图中心约 58% 的面积，且**不关心叶片在哪**。

真实照片里叶片常常偏移、只占画面一部分，机械中心裁剪可能：
  (a) 裁掉叶片主体 / 病斑（信息丢失）；
  (b) 把背景换成了另一种背景，域差异并未缩小。

本脚本对比 5 种策略，回答「先裁番茄真实图能否提升识别率」：
  1. baseline    —— 现状（Resize + CenterCrop），用于复现 43.9%
  2. pad_square  —— 整图 pad 成正方形后缩放，保留 100% 内容（不裁任何像素）
  3. crop60      —— 只取中心 60%，模拟「裁得更紧」
  4. fore_bbox   —— 梯度能量法估叶片外接框（模拟「用户手工把叶片裁满画面」）
  5. multicrop   —— 整图 + 中心两档 + 四角，共 7 视图平均 logits（多裁剪集成）

评估口径与 eval_realtest.py 完全一致：带作物先验（只在该作物候选类内 argmax），
故 baseline 应复现 29/66。准确率只取决于 logits 排序，与温度校准无关。

用法：
    X:\\Anconda3\\python.exe scripts\\exp_crop_realtest.py            # 番茄
    X:\\Anconda3\\python.exe scripts\\exp_crop_realtest.py --all      # 全部 225 张
"""
import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image, ImageFilter

from src.config import CLASS_DIRS, DISEASE_KNOWLEDGE_PATH
from src.model.predict import Predictor


# ===================== 裁剪策略 =====================
def pad_square(im, fill=(114, 114, 114)):
    """整图补齐成正方形（灰底），不丢任何像素。"""
    w, h = im.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), fill)
    canvas.paste(im, ((s - w) // 2, (s - h) // 2))
    return canvas


def center_crop_ratio(im, keep):
    """中心裁剪，保留原图 keep 比例。"""
    w, h = im.size
    nw, nh = max(8, int(w * keep)), max(8, int(h * keep))
    x, y = (w - nw) // 2, (h - nh) // 2
    return im.crop((x, y, x + nw, y + nh))


def fore_bbox(im, pad=0.12):
    """用梯度能量估叶片外接框。

    思路：叶片有叶脉、病斑、锯齿边缘 → 梯度强；常见背景（纸板 / 天空 / 白墙 /
    纯色桌面）梯度弱。故对梯度幅值取高分位作前景，再取外接框并外扩。
    不用颜色阈值：病害叶片会变黄褐，绿度阈值会漏检病斑区域。
    """
    gray = im.convert("L").filter(ImageFilter.GaussianBlur(1.2))
    arr = np.asarray(gray, dtype=np.float32)
    gy, gx = np.gradient(arr)
    energy = np.hypot(gx, gy)
    thr = np.percentile(energy, 80)          # 取梯度最强的 20% 像素作前景种子
    ys, xs = np.where(energy > thr)
    if xs.size < 20:
        return im
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0, y1 - y0
    x0 -= bw * pad
    x1 += bw * pad
    y0 -= bh * pad
    y1 += bh * pad
    # 扩成正方形，避免后续 CenterCrop 再切掉一边
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) / 2
    W, H = im.size
    box = (max(0, int(cx - half)), max(0, int(cy - half)),
           min(W, int(cx + half)), min(H, int(cy + half)))
    if box[2] - box[0] < 48 or box[3] - box[1] < 48:
        return im
    return im.crop(box)


def views_multicrop(im):
    """多裁剪集成视图：整图 pad + 中心两档 + 四角。"""
    out = [pad_square(im), center_crop_ratio(im, 0.70)]
    w, h = im.size
    for fx, fy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        vw, vh = int(w * 0.70), int(h * 0.70)
        x, y = int(fx * (w - vw)), int(fy * (h - vh))
        out.append(im.crop((x, y, x + vw, y + vh)))
    return out


# ===================== 推理 =====================
def masked_logits(pred, im, ids):
    """按部署口径推一批 logits，并施加作物先验掩码（非候选类置 -inf）。"""
    x = pred.transform(im).unsqueeze(0).to(pred.device)
    with torch.no_grad():
        lg = pred.model(x)
    if ids:
        m = torch.full_like(lg, float("-inf"))
        m[:, ids] = lg[:, ids]
        lg = m
    return lg


def run_strategy(pred, items, crop_of, name):
    """返回 (正确数, 总数, 逐类统计, 逐图对错标记)。"""
    per = defaultdict(lambda: [0, 0])
    flags = []
    for path, cid in items:
        im = Image.open(path).convert("RGB")
        ids = pred.resolve_crop(crop_of(cid))
        if name == "multicrop":
            lg = torch.stack([masked_logits(pred, v, ids)
                              for v in views_multicrop(im)]).mean(0)
        elif name == "baseline":
            lg = masked_logits(pred, im, ids)
        elif name == "pad_square":
            lg = masked_logits(pred, pad_square(im), ids)
        elif name == "crop60":
            lg = masked_logits(pred, center_crop_ratio(im, 0.60), ids)
        elif name == "fore_bbox":
            lg = masked_logits(pred, fore_bbox(im), ids)
        else:
            raise ValueError(name)
        ok = int(lg.argmax(1).item() == cid)
        per[cid][0] += ok
        per[cid][1] += 1
        flags.append(ok)
    c = sum(v[0] for v in per.values())
    t = sum(v[1] for v in per.values())
    return c, t, per, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="评估全部 225 张，不只番茄")
    ap.add_argument("--index", default="outputs/index/dataset_index_realtest.json")
    ap.add_argument("--detail", action="store_true", help="打印逐类对比")
    ap.add_argument("--only", default=None, help="只跑指定策略，逗号分隔（如 baseline,multicrop）")
    ap.add_argument("--dump", default=None, help="逐图对错写入 json，供配对显著性检验使用")
    args = ap.parse_args()

    pred = Predictor.get_instance()
    know = json.loads(Path(DISEASE_KNOWLEDGE_PATH).read_text(encoding="utf-8"))
    crop_of = lambda c: (know.get(CLASS_DIRS[c]) or {}).get("crop", "")

    idx = json.loads(Path(args.index).read_text(encoding="utf-8"))
    items = [(Path(p), int(cid)) for p, cid in idx["test"]]
    if not args.all:
        items = [(p, c) for p, c in items if crop_of(c) == "番茄"]

    from src.config import IMAGE_SIZE
    print(f"评估集：{'全部作物' if args.all else '番茄'} "
          f"{len(items)} 张｜IMAGE_SIZE={IMAGE_SIZE}")

    strategies = ["baseline", "pad_square", "crop60", "fore_bbox", "multicrop"]
    if args.only:
        strategies = [s.strip() for s in args.only.split(",") if s.strip()]
    label = {
        "baseline": "现状（Resize+CenterCrop）",
        "pad_square": "整图 pad 正方形（不裁）",
        "crop60": "中心再裁 60%（更紧）",
        "fore_bbox": "叶片外接框（模拟手工裁满）",
        "multicrop": "7 视图多裁剪集成",
    }
    results = {}
    for s in strategies:
        c, t, per, flags = run_strategy(pred, items, crop_of, s)
        results[s] = (c, t, per)
        if args.dump:
            Path(args.dump).write_text(json.dumps(
                {"strategy": s, "image_size": IMAGE_SIZE,
                 "items": [str(p) for p, _ in items], "flags": flags},
                ensure_ascii=False), encoding="utf-8")
            print(f"[dump] {s} -> {args.dump}")

    base_c = results[strategies[0]][0]
    print("\n" + "=" * 62)
    print(f"{'策略':<30}{'正确/总数':>12}{'准确率':>9}{'Δ':>8}")
    print("-" * 62)
    for s in strategies:
        c, t, _ = results[s]
        d = "" if s == "baseline" else f"{(c - base_c) / t * 100:+.1f}pp"
        print(f"{label[s]:<28}{f'{c}/{t}':>12}{c / t:>8.1%}{d:>10}")
    print("=" * 62)

    if args.detail:
        print("\n逐类对比（只在某一策略下判对的类别说明该策略有效）：")
        classes = sorted({cid for _, cid in items})
        print(f"{'类别':<44}{'base':>7}{'pad':>7}{'c60':>7}{'bbox':>7}{'multi':>7}")
        for cid in classes:
            row = [results[s][2][cid] for s in strategies]
            cells = "".join(f"{v[0]}/{v[1]:<5}" for v in row)
            print(f"{CLASS_DIRS[cid]:<44}{cells}")


if __name__ == "__main__":
    main()
