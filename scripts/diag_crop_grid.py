# -*- coding: utf-8 -*-
"""把番茄真实图拼成缩略图网格，人工核对「背景到底占多少、叶片是否本来就在画面里」。

用途：判断「裁剪」这个前提是否成立。若多数图本身就是叶片占满画面，
那裁剪能带来的收益上限就很低，不必再优化裁剪算法。
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw

from src.config import CLASS_DIRS, DISEASE_KNOWLEDGE_PATH


def main():
    idx = json.loads(Path("outputs/index/dataset_index_realtest.json").read_text(encoding="utf-8"))
    know = json.loads(Path(DISEASE_KNOWLEDGE_PATH).read_text(encoding="utf-8"))
    crop_of = lambda c: (know.get(CLASS_DIRS[c]) or {}).get("crop", "")

    items = [(Path(p), int(c)) for p, c in idx["test"] if crop_of(int(c)) == "番茄"]
    # 每个类抽 2 张，凑 16 张（覆盖番茄 8 个类）
    by_cls = {}
    for p, c in items:
        by_cls.setdefault(c, []).append(p)
    picked = []
    for c, ps in sorted(by_cls.items()):
        picked.extend(ps[:2])
    picked = picked[:16]

    CELL = 168
    COLS, ROWS = 4, 4
    canvas = Image.new("RGB", (COLS * CELL, ROWS * (CELL + 18)), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    for i, p in enumerate(picked):
        im = Image.open(p).convert("RGB")
        im.thumbnail((CELL, CELL))
        r, c = divmod(i, COLS)
        x = c * CELL + (CELL - im.size[0]) // 2
        y = r * (CELL + 18) + (CELL - im.size[1]) // 2
        canvas.paste(im, (x, y))
        label = p.parent.name.replace("Tomato___", "")[:22]
        draw.text((c * CELL + 4, r * (CELL + 18) + CELL + 3), f"{i} {label}", fill=(30, 30, 30))

    out = Path("outputs/crop_diag")
    out.mkdir(parents=True, exist_ok=True)
    canvas.save(out / "tomato_grid.jpg", quality=90)
    print(f"已导出 {out / 'tomato_grid.jpg'}（{len(picked)} 张，每格左上角为序号+类别）")
    for i, p in enumerate(picked):
        print(f"  {i:2d}  {p.name}")


if __name__ == "__main__":
    main()
