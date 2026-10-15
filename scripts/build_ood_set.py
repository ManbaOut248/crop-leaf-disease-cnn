# -*- coding: utf-8 -*-
"""构造 OOD（分布外）测试集，用于标定「非叶片输入」的拒识阈值。

为什么需要它：
    模型是 38 类闭集 softmax 分类器，没有"以上都不是"选项，对任何输入都会
    吐出一个类别（实测把动漫插画判成"番茄叶霉病 96.87%"）。要加拒识能力，
    就必须先有一批"非叶片"样本，才能标定阈值、评估误杀率。

来源（互为补充，默认全部免联网）：
    1. synthetic   程序合成 —— 纯色 / 渐变 / 噪声 / 棋盘 / 几何 / 文字。零依赖，
                   覆盖最极端的输入（用户可能误传截图、色块、图表）。
    2. localphoto  本地图库真实照片 —— skimage / sklearn 内置样例（人像、猫、咖啡、
                   砖墙、文档、文字、星空、眼底、细胞…），每张随机裁剪扩充。
                   **这是主力**：免下载，且正对"用户随手拍错/传错"的真实分布。
    3. realshot    真实场景 —— 项目内界面截图 + 用户实际误传过的图（动漫插画）。
    4. cifar10 / flowers102  公开数据集，可选。
                   **实测本机环境拉不动**：toronto.edu 与 ox.ac.uk 速度 0 B/s，
                   HF 镜像 429 限流。故默认跳过，需显式加 --try-download。

输出目录：data/ood_set/<来源>/   （已加入 .gitignore，不入库）

用法：
    py -3.12 scripts/build_ood_set.py                  # 推荐：合成 + 本地照片 + 真实截图
    py -3.12 scripts/build_ood_set.py --try-download   # 额外尝试公开数据集（大概率失败）
"""
import argparse
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUT_ROOT = PROJECT_ROOT / "data" / "ood_set"
DATA_ROOT = PROJECT_ROOT / "data"

SIZE = 224
SYNTH_PER_KIND = 15
N_CIFAR = 500
N_FLOWERS = 300
PER_PHOTO = 12  # 每张本地图库照片裁多少个 224×224 窗口


# ============================ 1) 程序合成 ============================
def build_synthetic(out_dir: Path) -> int:
    """生成各类"完全不像叶片"的图像。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    count = 0

    def save(arr_or_img, kind, idx):
        nonlocal count
        img = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
        img.convert("RGB").save(out_dir / f"{kind}_{idx:03d}.png")
        count += 1

    for i in range(SYNTH_PER_KIND):
        # --- 纯色（随机色相，含灰阶） ---
        c = tuple(int(v) for v in rng.integers(0, 256, 3))
        save(Image.new("RGB", (SIZE, SIZE), c), "solid", i)

        # --- 线性渐变 ---
        a = rng.integers(0, 256, 3).astype(np.float32)
        b = rng.integers(0, 256, 3).astype(np.float32)
        ramp = np.linspace(0, 1, SIZE, dtype=np.float32)[:, None, None]
        grad = (a[None, None, :] * (1 - ramp) + b[None, None, :] * ramp)
        grad = np.repeat(grad, SIZE, axis=1).astype(np.uint8)
        save(grad, "gradient", i)

        # --- 高斯噪声 / 均匀噪声 ---
        save(rng.normal(128, 45, (SIZE, SIZE, 3)).clip(0, 255).astype(np.uint8), "gauss_noise", i)
        save(rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8), "uniform_noise", i)

        # --- 棋盘格 ---
        blk = int(rng.integers(8, 40))
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        chess = (((yy // blk) + (xx // blk)) % 2 * 255).astype(np.uint8)
        save(np.stack([chess] * 3, axis=-1), "checker", i)

        # --- 条纹 ---
        period = int(rng.integers(6, 30))
        stripes = ((np.arange(SIZE) % period) < period / 2).astype(np.uint8) * 255
        save(np.stack([np.tile(stripes[:, None], (1, SIZE))] * 3, axis=-1), "stripes", i)

        # --- 几何图形（圆 / 矩形）---
        canvas = Image.new("RGB", (SIZE, SIZE), tuple(int(v) for v in rng.integers(200, 256, 3)))
        d = ImageDraw.Draw(canvas)
        for _ in range(int(rng.integers(3, 9))):
            x0, y0 = rng.integers(0, SIZE, 2)
            x1, y1 = rng.integers(0, SIZE, 2)
            fill = tuple(int(v) for v in rng.integers(0, 256, 3))
            if rng.random() < 0.5:
                d.ellipse([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=fill)
            else:
                d.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=fill)
        save(canvas, "shapes", i)

        # --- 文字块（模拟截图 / 文档） ---
        canvas = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
        d = ImageDraw.Draw(canvas)
        for row in range(6):
            y = 20 + row * 32
            w = int(rng.integers(SIZE // 3, SIZE - 30))
            d.rectangle([18, y, 18 + w, y + 14], fill=(30, 30, 30))
        save(canvas, "textblock", i)

    return count


# ============================ 2) CIFAR-10 ============================
def build_cifar(out_dir: Path) -> int:
    """自然图像（宽泛 OOD）。32×32 原图，评估时会被 resize 到 224。"""
    try:
        from torchvision.datasets import CIFAR10
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"torchvision 不可用，跳过 CIFAR-10：{e}")
        return 0

    try:
        ds = CIFAR10(root=str(DATA_ROOT / "_downloads"), train=False, download=True)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"CIFAR-10 下载失败，跳过：{e}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    # 每类均匀抽样，避免只抽到某几种
    labels = np.array(ds.targets)
    per = max(1, N_CIFAR // len(set(ds.targets)))
    idx = []
    for cls in sorted(set(ds.targets)):
        idx.extend(np.where(labels == cls)[0][:per].tolist())
    idx = idx[:N_CIFAR]

    for n, i in enumerate(idx):
        img, label = ds[i]
        img.resize((SIZE, SIZE), Image.BICUBIC).save(out_dir / f"cifar_{n:04d}_c{label}.png")
    return len(idx)


# ========================= 3) Flowers102（难例）=========================
def build_flowers(out_dir: Path) -> int:
    """花卉照片 —— 绿色茎叶 + 真实质感，与叶片分布最接近的 OOD 难例。"""
    try:
        from torchvision.datasets import Flowers102
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"torchvision 不可用，跳过 Flowers102：{e}")
        return 0

    try:
        ds = Flowers102(root=str(DATA_ROOT / "_downloads"), split="test", download=True)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"Flowers102 下载失败，跳过：{e}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, len(ds) // N_FLOWERS)
    picked = list(range(0, len(ds), step))[:N_FLOWERS]
    for n, i in enumerate(picked):
        img, _ = ds[i]
        img.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC).save(out_dir / f"flower_{n:04d}.png")
    return len(picked)


# ================== 4) 本地图库真实照片（免下载主力）==================
def build_localphoto(out_dir: Path) -> int:
    """科学计算库自带的真实照片 —— 免联网，覆盖"用户随手误传"的图像分布。

    为什么用它：toronto.edu / ox.ac.uk 实测速度 0 B/s、HF 镜像 429 限流，
    公开数据集实际拉不动。而 skimage / sklearn 内置的样例照片本身就是
    真实摄影或扫描图（人像、猫、咖啡、砖墙、文档、文字、星空、眼底、细胞…），
    与"用户拍错/传错"的场景高度一致，且零下载成本。

    每张图随机缩放 + 裁剪多个 224×224 窗口来扩充样本量。
    注意：同源裁剪彼此相关，统计上不算完全独立样本 —— 解读时按"图源"看待，
    不要当成 200 个独立个体。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    count = 0
    sources = []

    try:
        from skimage import data as skdata

        for n in (
            "astronaut", "brick", "camera", "chelsea", "clock", "coffee",
            "grass", "gravel", "horse", "hubble_deep_field", "moon", "page",
            "retina", "rocket", "text", "cat", "cell",
        ):
            # 注意：skimage.data 走 lazy_loader，属性不在 __dict__ 里，
            # 必须用 getattr 才能触发加载（用 __dict__[n] 会全部 KeyError 被吞掉）
            try:
                sources.append((n, getattr(skdata, n)()))
            except Exception:  # noqa: BLE001 —— 部分样例需联网下载，跳过
                continue
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"skimage 不可用：{e}")

    try:
        from sklearn.datasets import load_sample_images

        for i, im in enumerate(load_sample_images().images):
            sources.append((f"sklearn{i}", im))
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"sklearn 样例图不可用：{e}")

    for name, arr in sources:
        img = (arr if isinstance(arr, Image.Image) else Image.fromarray(arr)).convert("RGB")
        w, h = img.size
        for i in range(PER_PHOTO):
            # 先随机放大（保证能裁出 SIZE×SIZE），再随机取窗口
            scale = max(1.0, SIZE / min(w, h)) * rng.uniform(1.0, 1.6)
            big = img.resize((max(SIZE, int(w * scale)), max(SIZE, int(h * scale))), Image.BICUBIC)
            bw, bh = big.size
            x = int(rng.integers(0, max(1, bw - SIZE + 1)))
            y = int(rng.integers(0, max(1, bh - SIZE + 1)))
            big.crop((x, y, x + SIZE, y + SIZE)).save(out_dir / f"{name}_{i:02d}.png")
            count += 1
    return count


# ========================= 5) 真实场景 =========================
def build_realshot(out_dir: Path) -> int:
    """项目内的界面截图（用户可能截屏误传）+ 用户实际误传过的图。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    # 4.1 界面截图：theme_shots 与 ppt_shots
    for src_dir in (PROJECT_ROOT / "outputs" / "theme_shots", PROJECT_ROOT / "outputs" / "ppt_shots"):
        if not src_dir.is_dir():
            continue
        for p in sorted(src_dir.glob("*.png")):
            shutil.copy2(p, out_dir / f"ui_{src_dir.name}_{p.name}")
            count += 1

    # 4.2 用户实际误传的图（动漫插画）—— 真实误传样本，最有价值
    user_wrong = PROJECT_ROOT / "data" / "uploads" / "facbf8d64be345408c24c3ede2d2ff14.jpg"
    if user_wrong.is_file():
        with Image.open(user_wrong) as im:
            im.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC).save(out_dir / "user_anime_illustration.png")
        count += 1

    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--try-download",
        action="store_true",
        help="额外尝试下载 CIFAR-10 / Flowers102（本机网络实测拉不动，默认跳过）",
    )
    args = ap.parse_args()

    print(f"输出目录：{OUT_ROOT}\n")
    stats = {}
    stats["synthetic"] = build_synthetic(OUT_ROOT / "synthetic")
    print(f"  synthetic   合成图（色块/噪声）  : {stats['synthetic']:>4} 张")
    stats["localphoto"] = build_localphoto(OUT_ROOT / "localphoto")
    print(f"  localphoto  本地图库真实照片    : {stats['localphoto']:>4} 张")
    stats["realshot"] = build_realshot(OUT_ROOT / "realshot")
    print(f"  realshot    界面截图/真实误传   : {stats['realshot']:>4} 张")

    if not args.try_download:
        print("\n（默认跳过公开数据集：本机网络拉不动，详见脚本顶部说明）")
    else:
        stats["cifar10"] = build_cifar(OUT_ROOT / "cifar10")
        print(f"  cifar10     自然图像           : {stats['cifar10']:>4} 张")
        stats["flowers102"] = build_flowers(OUT_ROOT / "flowers102")
        print(f"  flowers102  花卉（难例）       : {stats['flowers102']:>4} 张")

    total = sum(stats.values())
    print(f"\n合计 OOD 样本：{total} 张")


if __name__ == "__main__":
    main()
