# -*- coding: utf-8 -*-
"""诊断：各裁剪策略平均保留了多少原图面积 + 可视化样例。

目的：确认 `fore_bbox`（梯度能量法）真的裁掉了背景，而不是几乎没动原图。
否则「手工裁满无效」的结论就站不住。
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from scripts.exp_crop_realtest import fore_bbox, center_crop_ratio, pad_square


def main():
    idx = json.loads(Path("outputs/index/dataset_index_realtest.json").read_text(encoding="utf-8"))
    items = [(Path(p), int(c)) for p, c in idx["test"]]

    from src.config import CLASS_DIRS, DISEASE_KNOWLEDGE_PATH
    know = json.loads(Path(DISEASE_KNOWLEDGE_PATH).read_text(encoding="utf-8"))
    crop_of = lambda c: (know.get(CLASS_DIRS[c]) or {}).get("crop", "")
    tomato = [(p, c) for p, c in items if crop_of(c) == "番茄"]

    ratios = {"bbox": [], "crop60": []}
    for p, _ in tomato:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        b = fore_bbox(im)
        ratios["bbox"].append((b.size[0] * b.size[1]) / (w * h))
        ratios["crop60"].append(0.36)  # 0.6*0.6

    print("番茄 66 张，裁剪后保留的原图面积比例：")
    for k, v in ratios.items():
        arr = np.array(v)
        print(f"  {k:<8} 平均 {arr.mean():.1%}  中位 {np.median(arr):.1%}  "
              f"最小 {arr.min():.1%}  最大 {arr.max():.1%}  <80% 的占比 {(arr < 0.8).mean():.0%}")

    # 导出 6 张对比图：原图 | bbox 裁剪
    out = Path("outputs/crop_diag")
    out.mkdir(parents=True, exist_ok=True)
    for i, (p, _) in enumerate(tomato[:6]):
        im = Image.open(p).convert("RGB")
        b = fore_bbox(im)
        H = 260
        def fit(x):
            r = H / x.size[1]
            return x.resize((max(1, int(x.size[0] * r)), H))
        a, c = fit(im), fit(b)
        canvas = Image.new("RGB", (a.size[0] + c.size[0] + 12, H), (255, 255, 255))
        canvas.paste(a, (0, 0))
        canvas.paste(c, (a.size[0] + 12, 0))
        canvas.save(out / f"{i:02d}_{p.stem}.jpg", quality=88)
    print(f"\n对比图已导出: {out}（左=原图，右=bbox 裁剪）")


if __name__ == "__main__":
    main()
