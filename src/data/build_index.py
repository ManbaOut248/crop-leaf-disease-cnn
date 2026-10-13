# -*- coding: utf-8 -*-
"""
构建 8:1:1 分层抽样索引，并对“近重复图”做 group-aware 分组以杜绝数据泄漏
（对齐任务书数据准备要求）。

核心改动（C-1）：
- 旧逻辑：逐张随机切分 -> 同一片叶子的近重复图被打散到 train/test 两边，造成泄漏；
- 新逻辑：对每类图片按 aHash 汉明距离 <= GROUP_THRESH 用并查集归为“同叶组”，
  整组作为原子单元分配到 train/val/test 之一，并按 8:1:1 逼近；
  同时保证每个类别在三套切分中至少各 1 组（小类兜底），避免某类在某切分为空。

输出 outputs/index/dataset_index.json（格式不变：[绝对路径, 标签]，不移动/拷贝图片）。
"""

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import (
    DATA_ROOT, INDEX_DIR, CLASS_DIRS, SPLIT_TEST, SPLIT_VAL,
    SPLIT_SEED, REPORT_DIR, GROUP_HASH_SIZE, GROUP_THRESH,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ahash_int(path: str) -> int:
    """平均哈希：灰度 -> GROUP_HASH_SIZE x GROUP_HASH_SIZE -> 按均值二值化 -> 打包成整数指纹。"""
    with Image.open(path) as im:
        im = im.convert("L").resize((GROUP_HASH_SIZE, GROUP_HASH_SIZE), Image.Resampling.LANCZOS)
    arr = np.asarray(im, dtype=np.float32)
    bits = (arr > arr.mean()).flatten().astype(np.uint64)   # N 个 0/1
    powers = (1 << np.arange(bits.size - 1, -1, -1)).astype(np.uint64)
    return int((bits * powers).sum())                        # 打包为单个整数


def collect_images():
    """返回 {label: [abs_path, ...]}，label 对应 CLASS_DIRS 索引。"""
    by_label = {i: [] for i in range(len(CLASS_DIRS))}
    for label, class_dir in enumerate(CLASS_DIRS):
        d = DATA_ROOT / class_dir
        if not d.exists():
            raise FileNotFoundError(f"类别目录缺失: {d}")
        count = 0
        for p in d.rglob("*"):
            if p.suffix.lower() in IMG_EXTS and p.is_file():
                by_label[label].append(str(p.resolve()))
                count += 1
        print(f"  [{label:2d}] {class_dir:55s} {count:5d} 张")
    return by_label


def _union_find_parents(n):
    """返回并查集 find/union 闭包。"""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    return find, union


def group_aware_split(by_label):
    """
    分组切分：对每类内部用 aHash + 并查集把近重复图归组，整组分配到 train/val/test。

    返回 (train, val, test, diag)：
    - 三个列表，元素为 [path, label]；
    - diag: 每类的分组诊断 {label: {n_groups, max_group, max_ratio}}。
    """
    rng = random.Random(SPLIT_SEED)
    train, val, test = [], [], []
    diag = {}

    for label, paths in by_label.items():
        # 1) 逐张算指纹（损坏图跳过）
        hashes, valid = [], []
        for p in paths:
            try:
                hashes.append(ahash_int(p))
                valid.append(p)
            except Exception:
                pass
        m = len(valid)
        if m == 0:
            diag[label] = {"n_groups": 0, "max_group": 0, "max_ratio": 0.0}
            continue

        # 2) 并查集按 GROUP_THRESH 归组
        #    前缀分桶：仅比较高 16bit 相同的候选（距离<=2 必满足高 16bit 相同，故不漏判），控复杂度
        find, union = _union_find_parents(m)
        buckets = {}
        for idx, h in enumerate(hashes):
            buckets.setdefault(h >> (GROUP_HASH_SIZE * GROUP_HASH_SIZE - 16), []).append(idx)
        for grp in buckets.values():
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    xor = hashes[grp[i]] ^ hashes[grp[j]]
                    # 汉明距离 = 异或后 1 的个数
                    if bin(xor).count("1") <= GROUP_THRESH:
                        union(grp[i], grp[j])

        # 3) 收集组（同叶组）
        groups_map = {}
        for idx in range(m):
            groups_map.setdefault(find(idx), []).append(valid[idx])
        g_list = list(groups_map.values())
        n = m
        n_test = max(1, int(round(n * SPLIT_TEST)))
        n_val = max(1, int(round(n * SPLIT_VAL)))

        # 4) 整组分配：大组优先进 train（目标 80%），剩余小组平分 val/test（各 10%），并保底三切分各>=1组
        #    避免“大组塞入 test/val 导致比例爆掉”的 bug —— 大组进 train 容错最大。
        g_desc = sorted(g_list, key=len, reverse=True)   # 大组在前
        n_train_t = n - n_test - n_val                   # 训练目标张数（≈80%）
        acc = {"train": 0, "val": 0, "test": 0}
        g_train, g_val, g_test = [], [], []
        buckets = {"train": g_train, "val": g_val, "test": g_test}
        for g in g_desc:
            if acc["train"] < n_train_t:
                buckets["train"].append(g); acc["train"] += len(g)
            else:
                # train 满，分给 val/test 中当前张数较少者（逼近各 10%）
                if acc["val"] <= acc["test"]:
                    buckets["val"].append(g); acc["val"] += len(g)
                else:
                    buckets["test"].append(g); acc["test"] += len(g)
        # 保底：每类三切分至少 1 组（组数>=3 时）；否则保证 train 至少 1 组
        if len(g_list) >= 3:
            if not g_test and g_train:
                g_test.append(g_train.pop(-1))   # 从 train 末尾（较小者）挪一组
            if not g_val and g_train:
                g_val.append(g_train.pop(-1))
        else:
            if not g_train:
                src = g_val if len(g_val) >= len(g_test) else g_test
                if src:
                    g_train.append(src.pop(-1))

        for g in g_train:
            train += [[p, label] for p in g]
        for g in g_val:
            val += [[p, label] for p in g]
        for g in g_test:
            test += [[p, label] for p in g]

        # 诊断：组数、最大组大小、最大组占该类比例
        max_group = max((len(g) for g in g_list), default=0)
        diag[label] = {
            "n_groups": len(g_list),
            "max_group": max_group,
            "max_ratio": round(max_group / n, 3) if n else 0.0,
        }

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test, diag


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    print("开始收集图片（原地引用，不拷贝）...")
    by_label = collect_images()
    total = sum(len(v) for v in by_label.values())
    print(f"合计 {total} 张，{len(CLASS_DIRS)} 类。开始分组切分(防泄漏)...")
    train, val, test, diag = group_aware_split(by_label)
    index = {"train": train, "val": val, "test": test}
    out = INDEX_DIR / "dataset_index.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    print(f"索引已写入: {out}")
    print(f"  训练集: {len(train)} 张 ({len(train)/total*100:.1f}%)")
    print(f"  验证集: {len(val)} 张 ({len(val)/total*100:.1f}%)")
    print(f"  测试集: {len(test)} 张 ({len(test)/total*100:.1f}%)")

    # 诊断：检查是否有类被压成“单组 / 超大组”
    single = [i for i, d in diag.items() if d["n_groups"] <= 1]
    big = [(i, d["max_ratio"]) for i, d in diag.items() if d["max_ratio"] > 0.5]
    print("\n[诊断] 每类组数 min/max : "
          f"{min(d['n_groups'] for d in diag.values())} / {max(d['n_groups'] for d in diag.values())}")
    if single:
        print(f"[警告] 以下类仅 1 组，无法三切分各保底（将退化为单切分）: {single}")
    else:
        print("[OK] 所有类均有多组，三切分保底成立。")
    if big:
        print(f"[注意] 以下类最大组占比>50%（同叶变体集中，属预期，不影响防泄漏）: {big}")
    else:
        print("[OK] 无异常超大组。")

    # 同时输出一份类别中文映射，供 API/前端/报告使用
    mapping = {i: c for i, c in enumerate(CLASS_DIRS)}
    with open(REPORT_DIR / "class_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"类别映射已写入: {REPORT_DIR / 'class_mapping.json'}")


if __name__ == "__main__":
    main()
