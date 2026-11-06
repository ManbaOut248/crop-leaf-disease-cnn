# -*- coding: utf-8 -*-
"""生成「病害参考图集」：为 38 个类别各挑 3 张代表性图片，输出到前端静态目录。

用途（两处共用同一套图，避免重复打包）：
    1. 病害识别结果页 —— Top-3 每个候选病害旁边给一张参考图，方便农户对照
    2. 病害知识库 —— 每张病害卡片显示参考图

为什么从 data/split/test 取而不是 train：
    test 集是**评估集**，从不参与训练，用它做展示不会有「拿训练图当范例」的
    疑虑；而且 test 每类 15~550 张，够挑。train 也可以，但没必要。

为什么按分位数挑，而不是随机：
    PV 里混有少量构图异常图（叶片偏边、过曝、极小）。按文件名排序后取
    20% / 50% / 80% 三个分位，等价于在「稳定中段」抽样，比 random.choice
    重复抽到烂图更可靠，且结果可复现（无随机种子依赖）。

输出：
    src/web/public/disease-ref/<class_id>-<1..3>.webp   400x400，约 15~25 KB/张
    src/web/public/disease-ref/manifest.json            前端可直接 fetch
"""
import json
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CLASS_DIRS  # noqa: E402

SPLIT_DIR = PROJECT_ROOT / "data" / "split" / "test"
OUT_DIR = PROJECT_ROOT / "src" / "web" / "public" / "disease-ref"

# 400px：知识库卡片展示宽约 200px，2x 屏需 400px；
# 再大体积翻倍但看不出差别（PV 原图本身只有 256px）。
SIZE = 400
QUALITY = 80
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
# 三分位：避开首尾的极端图
QUANTILES = (0.2, 0.5, 0.8)


def pick_samples(files, n=3):
    """按分位从已排序列表里挑 n 张，保证不重复。

    Args:
        files: 已排序的候选图片路径列表
        n: 需要的张数

    Returns:
        长度 <= n 的路径列表（某类图太少时可能不足 n 张）
    """
    if not files:
        return []
    picked, used = [], set()
    for q in QUANTILES[:n]:
        idx = min(int(len(files) * q), len(files) - 1)
        # 分位点撞车时向后顺延，避免同一张图被选两次
        while idx in used and idx + 1 < len(files):
            idx += 1
        if idx not in used:
            used.add(idx)
            picked.append(files[idx])
    return picked


def to_square(im, size=SIZE):
    """中心裁成正方形再缩放。

    为什么不直接 resize：PV 图本身是 256x256 方图，缩放无损；
    但少数异常图是非方形，直接 resize 会把病斑拉变形，中心裁更接近
    用户在识别页看到的构图。
    """
    w, h = im.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    im = im.crop((left, top, left + side, top + side))
    return im.resize((size, size), Image.LANCZOS)


def main():
    if not SPLIT_DIR.is_dir():
        raise SystemExit(f"找不到测试集目录：{SPLIT_DIR}（需先运行数据切分脚本）")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {}
    total_bytes = 0
    for cid, cls in enumerate(CLASS_DIRS):
        folder = SPLIT_DIR / cls
        if not folder.is_dir():
            print(f"[跳过] {cid:2d} {cls}：目录不存在")
            continue
        files = sorted(
            [f for f in folder.iterdir() if f.suffix.lower() in IMG_EXT],
            key=lambda p: p.name,
        )
        picked = pick_samples(files)
        rel = []
        for i, src in enumerate(picked, start=1):
            name = f"{cid}-{i}.webp"
            dst = OUT_DIR / name
            with Image.open(src) as im:
                # 用 load() 而非 verify()：verify 抓不住截断的 JPEG
                im = im.convert("RGB")
                to_square(im).save(dst, format="WEBP", quality=QUALITY, method=6)
            total_bytes += dst.stat().st_size
            rel.append(f"/disease-ref/{name}")
        manifest[str(cid)] = rel
        print(f"[ok] {cid:2d} {cls:52s} {len(rel)} 张 (候选 {len(files)})")

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\n完成：{len(manifest)} 类 / {sum(len(v) for v in manifest.values())} 张，"
          f"合计 {total_bytes / 1024 / 1024:.2f} MB → {OUT_DIR}")


if __name__ == "__main__":
    main()
