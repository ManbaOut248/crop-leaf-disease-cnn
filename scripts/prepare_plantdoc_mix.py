# -*- coding: utf-8 -*-
"""
把克隆的 PlantDoc 官方数据集（pratikkayal/PlantDoc-Dataset，CC BY 4.0）映射到本项目的
38 类体系，并生成「混合训练索引」与「真实域测试索引」，供 scripts/finetune_domain.py 使用。

为什么需要
---------------------------------------------------------------
之前的 outputs/realtest 只覆盖了 PlantDoc 22 类，且漏掉了 6 个关键病害类
（番茄细菌斑/晚疫、葡萄黑腐、玉米锈病等）——正是真实域「病害类 0%」的根因。
本脚本改用**官方完整仓库**（train/ + test/ 官方划分，天然不重叠），把 28 个 PlantDoc
类全部映射到 38 类，补齐缺失病害类的真实标注。

防泄漏铁律
---------------------------------------------------------------
- PlantDoc 官方 train/ 只进「混合训练索引」的 train 段；官方 test/ 只进「真实域测试索引」
  的 test 段；两者官方划分保证不重叠。
- 真实域测试索引**绝不**进入 finetune 的训练/验证，只用于训练后最终评估。
- PlantVillage 原有 train/val/test 切分（防泄漏分组生成，见 build_index）原样保留，
  真实域数据仅追加，不破坏其划分。

输出
---------------------------------------------------------------
- data/real_mix/<38类全名>/       : 复制自 PlantDoc train 的真实域训练图（重命名，去 URL 残留）
- data/real_mix_test/<38类全名>/   : 复制自 PlantDoc test 的真实域评估图（仅评估用）
- outputs/index/dataset_index_mixed.json    : train=PV+真实域 / val=PV / test=PV
- outputs/index/dataset_index_realtest.json : test=真实域（train/val 空，仅供评估脚本读）

用法
    python -m scripts.prepare_plantdoc_mix
"""

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CLASS_DIRS  # 与生成 dataset_index.json 时同序，label 一致

# 与 export_plantdoc.py 一致的文件名安全化（Windows 非法字符 -> '_'）
_SAFE = re.compile(r"[^A-Za-z0-9._\-/]")


def safe(p: str) -> str:
    return "/".join(_SAFE.sub("_", part) for part in p.split("/"))

# PlantDoc 官方类名 -> 本项目 38 类全名（与 CLASS_DIRS 元素一致）
PLANTDOC_TO_38 = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato leaf": "Tomato___healthy",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf black rot": "Grape___Black_rot",
    "grape leaf": "Grape___healthy",
}

RAW = ROOT / "data" / "plantdoc_raw" / "exported"  # 经 export_plantdoc.py 安全化导出
REAL_MIX_DIR = ROOT / "data" / "real_mix"
REAL_TEST_DIR = ROOT / "data" / "real_mix_test"
PV_INDEX = ROOT / "outputs" / "index" / "dataset_index.json"
MIXED_INDEX = ROOT / "outputs" / "index" / "dataset_index_mixed.json"
REALTEST_INDEX = ROOT / "outputs" / "index" / "dataset_index_realtest.json"

SAFE_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def copy_class(src_dir: Path, dst_dir: Path):
    """把 src_dir 下所有图片重命名为 <类名>_<计数>.<ext> 复制到 dst_dir，返回复制数。"""
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src_dir.iterdir()):
        if p.suffix.lower() not in SAFE_SUFFIX:
            continue
        try:
            # 轻量校验：能打开且非 0 字节
            if p.stat().st_size < 512:
                continue
            dst = dst_dir / f"{dst_dir.name}__{n:05d}{p.suffix.lower()}"
            shutil.copyfile(p, dst)
            n += 1
        except Exception:
            continue
    return n


def main():
    if not RAW.exists():
        raise SystemExit(f"未找到导出的 PlantDoc 数据：{RAW}\n请先运行：python -m scripts.export_plantdoc")

    print(f"PlantDoc 源: {RAW}")
    print(f"映射表: {len(PLANTDOC_TO_38)} 个 PlantDoc 类 -> 38 类\n")

    # label 查表（与生成 PV 索引时同一 CLASS_DIRS 顺序，保证一致）
    name_to_label = {name: i for i, name in enumerate(CLASS_DIRS)}

    real_train, real_test = [], []
    print(f"{'PlantDoc 类':<32}{'-> 38类':<42}{'train':>6}{'test':>6}")
    missing = []
    for gh_name, target38 in PLANTDOC_TO_38.items():
        src_train = RAW / "train" / safe(gh_name)
        src_test = RAW / "test" / safe(gh_name)
        if not src_train.exists():
            missing.append(gh_name)
            print(f"{gh_name:<32}{target38:<42}{'缺失':>6}")
            continue
        label = name_to_label[target38]
        # 真实域训练图（进混合索引 train）
        ntr = copy_class(src_train, REAL_MIX_DIR / target38)
        for p in sorted((REAL_MIX_DIR / target38).iterdir()):
            real_train.append([str(p.resolve()), label])
        # 真实域测试图（仅评估，不进训练）；test 中缺失的类直接跳过
        if src_test.exists():
            nte = copy_class(src_test, REAL_TEST_DIR / target38)
            for p in sorted((REAL_TEST_DIR / target38).iterdir()):
                real_test.append([str(p.resolve()), label])
        else:
            nte = 0
        print(f"{gh_name:<32}{target38:<42}{ntr:>6}{nte:>6}")

    if missing:
        print(f"\n⚠️ 以下 PlantDoc 类在仓库中缺失（跳过）：{missing}")

    # —— 评估集纯洁性：去重 + 防泄漏 ——
    # PlantDoc 源数据存在内部重复（同图多文件名），且官方 train/test 划分有少量重叠。
    # 1) real_mix（训练真实图）内部按 md5 去重；
    # 2) real_test（评估集）内部去重，并剔除与 real_mix 重叠的文件（防泄漏到训练）。
    def _md5(p):
        return hashlib.md5(Path(p).read_bytes()).hexdigest()

    real_mix_sigs = {_md5(p) for p, _ in real_train}
    seen, real_train_u = set(), []
    for p, lab in real_train:
        s = _md5(p)
        if s in seen:
            continue
        seen.add(s)
        real_train_u.append([p, lab])
    real_train = real_train_u

    seen, real_test_u, leaked = set(), [], 0
    for p, lab in real_test:
        s = _md5(p)
        if s in seen:
            continue
        seen.add(s)
        if s in real_mix_sigs:   # 该图也出现在训练真实集 -> 泄漏，评估集剔除
            leaked += 1
            continue
        real_test_u.append([p, lab])
    real_test = real_test_u
    print(f"\n去重后：训练真实图 {len(real_train)} | 评估真实图 {len(real_test)}"
          f"（剔除泄漏 {leaked} 张）")

    # 加载 PV 索引，追加真实域
    pv = json.load(open(PV_INDEX, encoding="utf-8"))
    mixed = {
        "train": pv["train"] + real_train,
        "val": pv["val"],
        "test": pv["test"],
    }
    realtest_idx = {
        "train": [],
        "val": [],
        "test": real_test,
    }
    MIXED_INDEX.parent.mkdir(parents=True, exist_ok=True)
    json.dump(mixed, open(MIXED_INDEX, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(realtest_idx, open(REALTEST_INDEX, "w", encoding="utf-8"), ensure_ascii=False)

    print(f"\n=== 索引生成完成 ===")
    print(f"混合索引 train: {len(mixed['train'])} (PV {len(pv['train'])} + 真实 {len(real_train)})")
    print(f"混合索引 val  : {len(mixed['val'])} (PV)")
    print(f"混合索引 test : {len(mixed['test'])} (PV)")
    print(f"真实域测试    : {len(real_test)} 张（仅评估，未进训练）")
    print(f"\n产物:")
    print(f"  {MIXED_INDEX}")
    print(f"  {REALTEST_INDEX}")
    print(f"  真实域图片: {REAL_MIX_DIR} / {REAL_TEST_DIR}")


if __name__ == "__main__":
    main()
