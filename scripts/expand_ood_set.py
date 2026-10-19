# -*- coding: utf-8 -*-
"""
扩充「非叶片」负样本集（OOD 检测头的训练/评估素材）。

为什么需要扩充
---------------------------------------------------------------
叶片检测头的跨来源留一验证里，最悲观的一档 AUROC 只有 0.77。
瓶颈不在模型（513 个参数的逻辑回归已经够用），而在**负样本多样性**：
原有 363 张里 228 张来自同一批科学计算库照片、120 张是程序合成、
仅 15 张是真实界面截图 —— 面对「没见过的那一类非叶片」时泛化不足。

本脚本做两件事：

1) **几何 / 色彩增强**（localphoto、realshot）
   每张源图生成若干变体：水平翻转、旋转、随机裁剪、亮度/对比度/饱和度抖动、
   轻微模糊与噪声。真实用户手机拍的界面截图、随手裁的图，形态千变万化。

2) **新增两个「真实误传」类别**（此前完全没有覆盖）
   · charts   图表截图 —— 用户经常误传统计图、报表、仪表盘
   · docshot  文档 / 文字截图 —— 误传论文截图、表格、说明书

关键约束：**文件名带 `__aug<N>` 后缀，训练脚本按「源图分组」划分**
---------------------------------------------------------------
增强图与源图高度相似。若按文件随机划分，测试集里会混入训练图的近邻副本，
AUROC 会被显著虚高（这正是此前 MSP 判据踩过的坑）。
故本脚本统一命名为 `<源文件主干>__aug<N>.png`，
训练脚本用「`__aug` 之前的部分」作为分组键，保证同一源图的所有变体
只会落在训练侧或测试侧之一。

用法
    py -3.12 scripts/expand_ood_set.py                # 全量生成
    py -3.12 scripts/expand_ood_set.py --aug-per 3    # 每张源图增强 3 份
    py -3.12 scripts/expand_ood_set.py --clean        # 先清空上次生成的增强图

输出：data/ood_set/{localphoto,realshot,charts,docshot}/（已在 .gitignore 中）
"""

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

SEED = 42
SIZE = 224                      # 与模型输入同尺寸，避免后续缩放的二次损失
AUG_PER_SRC = 3                 # 每张源图生成多少变体
N_CHARTS = 80                   # 图表截图数量
N_DOCSHOTS = 60                 # 文档截图数量

OOD_ROOT = Path("data/ood_set")
AUG_SUFFIX = "__aug"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(SEED)
np.random.seed(SEED)


# ========================= 1) 几何 + 色彩增强 =========================
def augment(img, rng):
    """对单张图施加随机几何与色彩扰动，返回 224x224 RGB。

    参数刻意取得温和（旋转 ≤ 12°、裁剪保留 ≥ 78% 视野）：
    过度形变会把图变成"抽象纹理"，反而让检测头学到与真实误传无关的模式。
    """
    w, h = img.size
    # --- 几何：水平翻转 ---
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # --- 几何：小幅旋转（扩边后裁回，避免黑角）---
    if rng.random() < 0.6:
        ang = rng.uniform(-12, 12)
        img = img.rotate(ang, resample=Image.BICUBIC, expand=False)
    # --- 几何：随机裁剪（保留 78%~100% 视野），再缩放回 224 ---
    ratio = rng.uniform(0.78, 1.0)
    cw, ch = int(w * ratio), int(h * ratio)
    x0 = rng.randint(0, max(0, w - cw))
    y0 = rng.randint(0, max(0, h - ch))
    img = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((SIZE, SIZE), Image.BICUBIC)
    # --- 色彩：亮度 / 对比度 / 饱和度 ---
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.78, 1.22))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.80, 1.20))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.65, 1.35))
    # --- 质感：轻微模糊（模拟截图缩放 / 手抖）---
    if rng.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 1.1)))
    # --- 质感：轻噪声 ---
    if rng.random() < 0.35:
        a = np.asarray(img).astype(np.int16)
        a += np.random.normal(0, rng.uniform(1.5, 5.0), a.shape).astype(np.int16)
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return img


def group_of(p: Path) -> str:
    """分组键：去掉 `__augN` 后缀。同一源图的所有变体共享同一分组。"""
    return p.stem.split(AUG_SUFFIX)[0]


def expand_dir(d: Path, per_src: int, tag: str) -> int:
    """对目录内**原始**图（不含上次生成的增强图）生成 per_src 个变体。"""
    if not d.is_dir():
        return 0
    srcs = [p for p in sorted(d.iterdir())
            if p.suffix.lower() in IMG_EXT and AUG_SUFFIX not in p.stem]
    n = 0
    for p in srcs:
        try:
            base = Image.open(p).convert("RGB")
        except Exception:
            continue
        for i in range(per_src):
            out = d / f"{p.stem}{AUG_SUFFIX}{i}.png"
            if out.exists():          # 幂等：重复运行不重复生成
                continue
            augment(base, random).save(out)
            n += 1
    print(f"  {tag:<12} {len(srcs):>3} 张源图 → 新增 {n:>4} 张变体")
    return n


# ========================= 2) 图表截图 =========================
def build_charts(out_dir: Path, n: int) -> int:
    """用 matplotlib 生成各类统计图表，模拟「误传报表/仪表盘截图」。

    为什么值得单独做一类：实际误传的图里，图表占比不低（截图、报表、PPT），
    而它们的特征与照片、与合成噪声都不同 —— 是检验泛化能力的有效新来源。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    kinds = ["line", "bar", "barh", "scatter", "pie", "hist", "area", "heat"]
    made = 0
    for i in range(n):
        kind = kinds[i % len(kinds)]
        fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=90)
        rng = np.random.default_rng(SEED + i)
        k = rng.integers(3, 9)

        if kind == "line":
            x = np.arange(k * 3)
            for _ in range(rng.integers(1, 4)):
                ax.plot(x, np.cumsum(rng.normal(0, 1, len(x))))
            ax.set_title(f"Trend {i}")
        elif kind == "bar":
            ax.bar([f"Q{j+1}" for j in range(k)], rng.integers(5, 100, k))
            ax.set_title(f"Quarterly {i}")
        elif kind == "barh":
            ax.barh([f"Item{j}" for j in range(k)], rng.integers(5, 100, k))
            ax.set_title(f"Ranking {i}")
        elif kind == "scatter":
            ax.scatter(rng.normal(0, 1, 60), rng.normal(0, 1, 60),
                       c=rng.random(60), cmap="viridis")
            ax.set_title(f"Correlation {i}")
        elif kind == "pie":
            ax.pie(rng.integers(5, 40, k), labels=[f"S{j}" for j in range(k)],
                   autopct="%1.0f%%")
            ax.set_title(f"Share {i}")
        elif kind == "hist":
            ax.hist(rng.normal(0, 1, 400), bins=rng.integers(8, 25))
            ax.set_title(f"Distribution {i}")
        elif kind == "area":
            ax.stackplot(np.arange(k * 2), [rng.integers(1, 20, k * 2) for _ in range(3)])
            ax.set_title(f"Composition {i}")
        else:  # heat
            ax.imshow(rng.random((k, k)), cmap="magma")
            ax.set_title(f"Matrix {i}")
            fig.colorbar(ax.images[0], ax=ax)

        ax.grid(rng.random() < 0.6, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"chart_{kind}_{i:03d}.png", facecolor="white")
        plt.close(fig)
        made += 1
    print(f"  {'charts':<12} 生成 {made:>4} 张（{len(kinds)} 种图表类型）")
    return made


# ========================= 3) 文档 / 文字截图 =========================
def build_docshots(out_dir: Path, n: int) -> int:
    """用 PIL 渲染「文档 / 表格 / 说明书」类截图。

    覆盖另一类高频误传：论文截图、报表、说明书照片。它们的大面积白底 +
    密集黑色笔画，是模型容易与「叶脉纹理」混淆的区域，值得专门测试。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    font_path = "C:/Windows/Fonts/arial.ttf"
    made = 0
    for i in range(n):
        rng = random.Random(SEED + i)
        bg = rng.choice([(255, 255, 255), (248, 248, 244), (238, 240, 242), (30, 30, 32)])
        fg = (25, 25, 25) if sum(bg) > 380 else (228, 228, 228)
        img = Image.new("RGB", (SIZE * 2, SIZE * 2), bg)
        d = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, rng.randint(13, 20))
        title = ImageFont.truetype(font_path, rng.randint(24, 32))

        d.text((28, 22), rng.choice(["Report", "Manual", "Notes", "Specification",
                                     "Invoice", "Summary"]), font=title, fill=fg)
        y = 74
        style = i % 3
        if style == 0:                      # 段落文字
            while y < img.height - 40:
                width = rng.randint(0, 110)
                words = " ".join(rng.choice(
                    ["data", "model", "result", "value", "sample", "method",
                     "input", "output", "score", "analysis"]) for _ in range(width // 10 + 1))
                d.text((28, y), words[:width], font=font, fill=fg)
                y += rng.randint(22, 30)
        elif style == 1:                    # 表格
            rows, cols = rng.randint(5, 9), rng.randint(3, 5)
            cw = (img.width - 56) // cols
            for r in range(rows):
                for c in range(cols):
                    x0, y0 = 28 + c * cw, y + r * 26
                    d.rectangle([x0, y0, x0 + cw, y0 + 26], outline=fg)
                    if r > 0:
                        d.text((x0 + 6, y0 + 6), f"{rng.randint(0, 999)}",
                               font=font, fill=fg)
        else:                               # 列表 / 要点
            while y < img.height - 40:
                d.text((28, y), "- " + " ".join(rng.choice(
                    ["item", "step", "note", "check", "value"])
                    for _ in range(rng.randint(2, 6))), font=font, fill=fg)
                y += rng.randint(24, 34)

        # 轻微旋转 + 模糊，模拟拍照/截图缩放，避免与"程序生成"过于同质
        if rng.random() < 0.5:
            img = img.rotate(rng.uniform(-3, 3), resample=Image.BICUBIC,
                             fillcolor=bg, expand=False)
        img.save(out_dir / f"doc_{style}_{i:03d}.png")
        made += 1
    print(f"  {'docshot':<12} 生成 {made:>4} 张（段落 / 表格 / 列表）")
    return made


# ========================= 主流程 =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug-per", type=int, default=AUG_PER_SRC)
    ap.add_argument("--n-charts", type=int, default=N_CHARTS)
    ap.add_argument("--n-docs", type=int, default=N_DOCSHOTS)
    ap.add_argument("--clean", action="store_true", help="先删除已有的增强图与新增类别")
    args = ap.parse_args()

    print("=" * 68)
    print("扩充非叶片负样本集")
    print("=" * 68)

    if args.clean:
        for sub in ("localphoto", "realshot"):
            for p in (OOD_ROOT / sub).glob(f"*{AUG_SUFFIX}*.png"):
                p.unlink()
        for sub in ("charts", "docshot"):
            shutil.rmtree(OOD_ROOT / sub, ignore_errors=True)
        print("  已清理旧产物")

    total = 0
    total += expand_dir(OOD_ROOT / "localphoto", args.aug_per, "localphoto")
    total += expand_dir(OOD_ROOT / "realshot", args.aug_per, "realshot")
    total += build_charts(OOD_ROOT / "charts", args.n_charts)
    total += build_docshots(OOD_ROOT / "docshot", args.n_docs)

    print("\n" + "=" * 68)
    print("最终构成")
    print("=" * 68)
    grand = 0
    for d in sorted(OOD_ROOT.iterdir()):
        if not d.is_dir():
            continue
        fs = [p for p in d.iterdir() if p.suffix.lower() in IMG_EXT]
        groups = len({group_of(p) for p in fs})
        grand += len(fs)
        print(f"  {d.name:<12} {len(fs):>4} 张 / {groups:>3} 个源图分组")
    print(f"  {'合计':<12} {grand:>4} 张   （本次新增 {total} 张）")
    print("\n提示：训练脚本按「源图分组」划分训练/评估，增强图不会跨侧泄漏。")
