# -*- coding: utf-8 -*-
"""
实验二：番茄真实图按「样本典型度」分组，分别测裁剪/滑动窗口的效果。

动机
----
排序网格显示，番茄 66 张里混有非单叶特写样本：
  · 带文字的科普信息图 / 百科截图
  · 整株远景（叶片只占画面一小块）
  · 图库白底 / 黑底合成图、多叶平铺
这些样本「裁不裁」根本不是主要矛盾。若不分开看，平均值会把结论冲淡。

分组口径（不靠人工标注，用模型自身信号）：
  · leaf_prob —— OOD 叶片检测头输出，低值意味着「不像单叶特写」
  · 再叠加「最大绿色连通域占比」，区分远景（绿色分散）与信息图（大面积白/黑底）

对比两种用法：
  · baseline   —— 现状单次中心裁剪
  · slide9max  —— 3×3 重叠滑窗，取置信度最高的一块（等价于「自动裁出最像叶片的那块」）
  · slide9avg  —— 同上但概率平均
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from src.config import CLASS_DIRS, DISEASE_KNOWLEDGE_PATH
from src.model.predict import Predictor


def green_ratio(im):
    """最大绿色连通域占画面比例（粗略衡量「有没有大片作物冠层」）。"""
    a = np.asarray(im.resize((128, 128)).convert("RGB"), np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mask = (g > r * 1.05) & (g > b * 1.05) & (g > 0.18)
    return float(mask.mean())


def slide_views(im, n=3, overlap=0.5):
    """n×n 重叠滑窗视图。"""
    w, h = im.size
    step = 1.0 / (n - (n - 1) * overlap)
    size = step * (1 - 0)
    cell = 1.0 / (n - (n - 1) * overlap)
    out = []
    for i in range(n):
        for j in range(n):
            x0 = (j * (w * (1 - cell)) / max(1, n - 1)) if n > 1 else 0
            y0 = (i * (h * (1 - cell)) / max(1, n - 1)) if n > 1 else 0
            out.append(im.crop((int(x0), int(y0),
                                int(x0 + w * cell), int(y0 + h * cell))))
    return out


def masked_logits(pred, im, ids):
    x = pred.transform(im).unsqueeze(0).to(pred.device)
    with torch.no_grad():
        lg = pred.model(x)
    if ids:
        m = torch.full_like(lg, float("-inf"))
        m[:, ids] = lg[:, ids]
        lg = m
    return lg


def main():
    pred = Predictor.get_instance()
    know = json.loads(Path(DISEASE_KNOWLEDGE_PATH).read_text(encoding="utf-8"))
    crop_of = lambda c: (know.get(CLASS_DIRS[c]) or {}).get("crop", "")

    idx = json.loads(Path("outputs/index/dataset_index_realtest.json").read_text(encoding="utf-8"))
    items = [(Path(p), int(c)) for p, c in idx["test"] if crop_of(int(c)) == "番茄"]

    rows = []
    for path, cid in items:
        im = Image.open(path).convert("RGB")
        res, ood = pred.predict_image_with_ood(im, top_k=1, crop="番茄")
        ids = pred.resolve_crop("番茄")

        # baseline
        lg0 = masked_logits(pred, im, ids)
        ok0 = int(lg0.argmax(1).item() == cid)

        # 滑窗
        vs = slide_views(im, 3, 0.5)
        lgs = torch.stack([masked_logits(pred, v, ids) for v in vs])
        probs = torch.softmax(lgs, dim=1)
        # max 用法：取「最高置信度视图」的预测
        best_i = int(probs.max(dim=1).values.argmax())
        ok_max = int(lgs[best_i].argmax(1).item() == cid)
        # avg 用法
        ok_avg = int(lgs.mean(0).argmax(1).item() == cid)

        rows.append(dict(path=path, cid=cid, leaf=float(ood["leaf_prob"]),
                         green=green_ratio(im), ok0=ok0, ok_max=ok_max, ok_avg=ok_avg))

    # 分组：leaf_prob 高 = 像叶片特写；低 = 疑似非叶片/远景
    def bucket(r):
        if r["leaf"] >= 0.90:
            return "A 典型叶片(leaf≥0.90)"
        if r["leaf"] >= 0.50:
            return "B 边缘(0.50≤leaf<0.90)"
        return "C 疑似非叶片(leaf<0.50)"

    groups = {}
    for r in rows:
        groups.setdefault(bucket(r), []).append(r)

    print(f"番茄共 {len(rows)} 张\n")
    print(f"{'分组':<26}{'张数':>6}{'baseline':>12}{'滑窗max':>11}{'滑窗avg':>11}")
    print("-" * 68)
    for name in sorted(groups):
        g = groups[name]
        n = len(g)
        f = lambda k: f"{sum(r[k] for r in g)}/{n}"
        print(f"{name:<26}{n:>6}{f('ok0'):>12}{f('ok_max'):>11}{f('ok_avg'):>11}")
    n = len(rows)
    f = lambda k: f"{sum(r[k] for r in rows)}/{n}"
    print("-" * 68)
    print(f"{'合计':<26}{n:>6}{f('ok0'):>12}{f('ok_max'):>11}{f('ok_avg'):>11}")

    print("\nC 组（疑似非叶片 / 远景）清单：")
    for r in sorted(groups.get("C 疑似非叶片(leaf<0.50)", []), key=lambda x: x["leaf"]):
        print(f"  leaf={r['leaf']:.3f} green={r['green']:.2f} "
              f"{'base✓' if r['ok0'] else 'base✗'} "
              f"{'slide✓' if r['ok_max'] else 'slide✗'}  {r['path'].name}")

    print("\nB 组清单：")
    for r in sorted(groups.get("B 边缘(0.50≤leaf<0.90)", []), key=lambda x: x["leaf"]):
        print(f"  leaf={r['leaf']:.3f} green={r['green']:.2f} "
              f"{'base✓' if r['ok0'] else 'base✗'} "
              f"{'slide✓' if r['ok_max'] else 'slide✗'}  {r['path'].name}")


if __name__ == "__main__":
    main()
