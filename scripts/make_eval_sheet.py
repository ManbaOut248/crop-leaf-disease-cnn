# -*- coding: utf-8 -*-
"""把评估结果拼成一张对比图（正确 vs 错判），用于人工核验与答辩展示。

为什么需要它
---------------------------------------------------------------
数字（Top-1 17.5%）容易被质疑"是不是标签搞错了/图片坏了"。
把真实标签与模型预测并排画出来，一眼就能确认：
标签来自目录名、图片是真实网络照片、错判是模型的问题而不是评估脚本的问题。

用法
    py -3.12 scripts/make_eval_sheet.py
产物
    outputs/web_eval_sheet.png
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "outputs" / "web_eval_report.json"
OUT = ROOT / "outputs" / "web_eval_sheet.png"

CELL = 260          # 单格边长
PAD = 12
HEAD = 54
FONT = "C:/Windows/Fonts/msyh.ttc"

GREEN = (36, 129, 74)
RED = (214, 69, 62)
INK = (28, 38, 32)
GREY = (107, 122, 114)


def main():
    d = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = d["rows"]
    right = [r for r in rows if r["top1_ok"]][:6]
    wrong = [r for r in rows if not r["top1_ok"]][:6]

    cols = 4
    n_rows = 3
    W = cols * CELL + (cols + 1) * PAD
    H = HEAD + n_rows * (CELL + 46) + (n_rows + 1) * PAD
    canvas = Image.new("RGB", (W, H), (247, 250, 248))
    draw = ImageDraw.Draw(canvas)

    f_title = ImageFont.truetype(FONT, 22)
    f_sub = ImageFont.truetype(FONT, 13)
    f_cell = ImageFont.truetype(FONT, 13)

    draw.text((PAD + 4, 14), "真实网络照片外部验证 · PlantDoc test（未参与训练）", font=f_title, fill=INK)
    draw.text((PAD + 4, 42),
              f"Top-1 {d['top1']:.1%}（{d['n']} 张）　Top-3 {d['top3']:.1%}　"
              f"作物级 {d['crop_level']:.1%}　左 6 张为正确，右 6 张为错判",
              font=f_sub, fill=GREY)

    items = right + wrong
    for i, r in enumerate(items):
        c, rr = i % cols, i // cols
        x = PAD + c * (CELL + PAD)
        y = HEAD + PAD + rr * (CELL + 46 + PAD)
        img = Image.open(ROOT / r["path"]).convert("RGB")
        img.thumbnail((CELL, CELL), Image.LANCZOS)
        box = Image.new("RGB", (CELL, CELL), (255, 255, 255))
        box.paste(img, ((CELL - img.width) // 2, (CELL - img.height) // 2))
        canvas.paste(box, (x, y))

        ok = r["top1_ok"]
        color = GREEN if ok else RED
        # 边框颜色 = 对/错（不只靠文字，避免看图时漏判）
        draw.rectangle([x, y, x + CELL - 1, y + CELL - 1], outline=color, width=3)

        draw.text((x + 2, y + CELL + 4), ("[正确] " if ok else "[错判] ") + "真实：" + r["true_name"],
                  font=f_cell, fill=color)
        draw.text((x + 2, y + CELL + 22),
                  f"预测：{r['pred_name']}（{r['confidence']:.2f}）",
                  font=f_cell, fill=INK if ok else GREY)

    canvas.save(OUT)
    print(f"已生成 {OUT.relative_to(ROOT)}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
