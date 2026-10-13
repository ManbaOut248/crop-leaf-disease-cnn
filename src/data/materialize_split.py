# -*- coding: utf-8 -*-
"""
C-3：把按 dataset_index.json 划分的图片，组织成自包含的独立文件夹。

输出目录结构（与源 PlantVillage 同级，便于对照与交付）：
    data/split/train/<英文类名>/xxx.JPG
    data/split/val/<英文类名>/xxx.JPG
    data/split/test/<英文类名>/xxx.JPG

特点：
- 只读 dataset_index.json，按其中 train/val/test 三套清单复制；
- 类名沿用源目录英文名（CLASS_DIRS），与模型输出索引、config 完全一致；
- 普通复制（shutil.copy2，保留元数据），不依赖硬链接，拷贝走即可独立运行；
- 可重建：索引改变后重跑本脚本即可重新组织（已存在文件直接覆盖）；
- 不删除原始 DATA_ROOT 数据集，训练仍可照常使用。

用法（项目根目录）：
    X:/Anconda3/python.exe -m src.data.materialize_split
"""
import json
import shutil
from pathlib import Path

from src.config import PROJECT_ROOT, INDEX_DIR, CLASS_DIRS

SPLIT_OUT = PROJECT_ROOT / "data" / "split"   # 独立数据集根目录


def main():
    index_path = INDEX_DIR / "dataset_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))

    SPLIT_OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for split in ("train", "val", "test"):
        entries = index[split]
        n = len(entries)
        print(f"== 组织 {split} 集（{n} 张） ==")
        for k, (path, label) in enumerate(entries, 1):
            src = Path(path)
            cls = CLASS_DIRS[label]                 # 英文类名，与源目录一致
            dst = SPLIT_OUT / split / cls / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)                 # 复制并保留元数据
            total += 1
            if k % 5000 == 0 or k == n:
                print(f"    {split}: {k}/{n} 已复制")
    print(f"完成：共 {total} 张图片已组织到 {SPLIT_OUT}")


if __name__ == "__main__":
    main()
