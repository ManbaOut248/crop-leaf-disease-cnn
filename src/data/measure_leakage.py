# -*- coding: utf-8 -*-
"""
数据泄漏测量脚本（只读，不改任何文件）。

目的：量化当前随机切分（build_index.py 生成的 dataset_index.json）下，
PlantVillage 同叶近重复图“跨 train/val/test 边界”的数量（即数据泄漏）。

原理：
1) 对每张图算平均哈希 aHash（灰度 8x8 + 均值二值化，打包成 64bit 指纹）；
2) 在“同一类别内”做全量两两比较（numpy 向量化，无分桶漏判），得到汉明距离；
3) 输出两类关键证据：
   (A) 距离梯度直方图：距离 0/1/2/3/4/5/6/7/8+(含) 各有多少对；
   (B) 测试图真孪生（金标准交叉验证）：有多少张“测试图”在“训练集”里存在
       距离 <=k 的孪生（k=0/1/2/3 四档）——距离 0~1 为近精确重复（强证据），
       距离 2~3 可能含同类相似误判。

判读（以距离<=1 的金标准为准）：
- 距离=0 的精确重复为 0，或 距离<=1 占比 <5%  → 80% 多为同类相似误判，C 可不做；
- 距离<=1 占比 >20%                          → 真有大量精确重复，C 必做；
- 介于之间                                    → 灰色地带，需抽样目视确认。

用法：
    X:/Anconda3/python.exe -m src.data.measure_leakage
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import INDEX_DIR, CLASS_DIRS

# ===================== 测量参数（脚本内常量，不碰 config.py） =====================
HASH_SIZE = 8        # aHash 尺寸，8x8 = 64bit 指纹
STRICT_THR = 3       # 真重复阈值：汉明距离 <=3 视为“同一片叶子的近精确重复”
LEAK_THR = 8         # 宽松阈值：仅用于统计“跨切分近重复对”做对照参考
HIST_BINS = 9        # 直方图桶数：下标 0..7 = 距离 0..7，下标 8 = 距离 >=8


def ahash_int(path: str) -> int:
    """平均哈希：转灰度 -> 8x8 -> 按均值二值化 -> 打包成 64bit 整数指纹。"""
    with Image.open(path) as im:
        im = im.convert("L").resize((HASH_SIZE, HASH_SIZE), Image.Resampling.LANCZOS)
    arr = np.asarray(im, dtype=np.float32)
    bits = (arr > arr.mean()).flatten().astype(np.uint64)   # 64 个 0/1
    powers = (1 << np.arange(63, -1, -1)).astype(np.uint64)  # 2^63 ... 2^0
    return int((bits * powers).sum())                        # 打包为单个 uint64


def count_class_leak(paths, split_of, strict_thr, leak_thr):
    """
    对单个类别的全部图片（跨 train/val/test 合并）做全量两两比较。

    返回 (hist, leak_pairs, test_twin_pairs, ttc)：
    - hist           : 长度 9 的数组，距离 0..7 与 >=8 的成对数量分布
    - leak_pairs     : 跨切分近重复对（距离<=leak_thr 且两端切分不同）
    - test_twin_pairs: 测试图与训练图之间的真孪生对总数（距离<=strict_thr）
    - ttc            : 长度 4 的数组，测试图在训练集存在“距离<=k 孪生”的张数 (k=0,1,2,3)
    """
    n = len(paths)
    if n < 2:
        return np.zeros(HIST_BINS, dtype=np.int64), 0, 0, np.zeros(4, dtype=np.int64)

    # 逐张算指纹（损坏图跳过）
    hashes, valid_paths = [], []
    for p in paths:
        try:
            hashes.append(ahash_int(p))
            valid_paths.append(p)
        except Exception:
            pass
    n = len(hashes)
    if n < 2:
        return np.zeros(HIST_BINS, dtype=np.int64), 0, 0, np.zeros(4, dtype=np.int64)

    H = np.array(hashes, dtype=np.uint64)                 # 指纹数组
    splits = np.array([split_of[p] for p in valid_paths])  # 每张图的切分
    is_test = (splits == "test")
    hist = np.zeros(HIST_BINS, dtype=np.int64)
    leak_pairs = 0
    test_twin_pairs = 0          # 测试-训练之间 距离<=strict_thr 的孪生对总数
    ttc = np.zeros(4, dtype=np.int64)  # 距离<=k 的测试孪生张数, k=0,1,2,3

    for i in range(n):
        # 与所有图的异或 -> 按字节展开成 64bit 求和 = 汉明距离
        xor = H ^ H[i]
        d = np.unpackbits(xor.view(np.uint8)).reshape(n, 64).sum(axis=1)

        # ① 直方图：只统计上三角(j>i)避免重复计数
        j_gt = np.arange(n) > i
        b = np.minimum(d[j_gt], HIST_BINS - 1).astype(np.int64)
        hist += np.bincount(b, minlength=HIST_BINS)

        # ② 跨切分近重复对（宽松阈值，上三角）
        mask_leak = (d <= leak_thr) & j_gt & (splits != splits[i])
        leak_pairs += int(mask_leak.sum())

        # ③ 测试图孪生（金标准交叉验证）：i 是测试图，与所有训练图(j!=i)比距离
        #    统计“在训练集存在距离<=k 孪生”的测试图张数，k=0/1/2/3（去重，每档计一次）
        if is_test[i]:
            m_train = (splits == "train") & (np.arange(n) != i)
            d_train = d[m_train]
            if d_train.size:
                for k in range(4):
                    if int((d_train <= k).sum()) > 0:
                        ttc[k] += 1
                test_twin_pairs += int((d_train <= strict_thr).sum())

    return hist, leak_pairs, test_twin_pairs, ttc


def main():
    index_path = INDEX_DIR / "dataset_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"索引不存在，请先运行 build_index：{index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))

    # 构建 {路径: 所属切分}
    split_of = {}
    for split in ("train", "val", "test"):
        for p, _ in index[split]:
            split_of[p] = split

    # 全部样本按类别归组（跨切分合并）
    by_label = {i: [] for i in range(len(CLASS_DIRS))}
    for split in ("train", "val", "test"):
        for p, label in index[split]:
            by_label[label].append(p)

    test_n = len(index["test"])
    print(f"类别数: {len(CLASS_DIRS)} | 总图片(索引): {len(split_of)} | "
          f"测试集: {test_n}\n"
          f"STRICT_THR(真重复)={STRICT_THR} | LEAK_THR(宽松)={LEAK_THR}\n")

    grand_hist = np.zeros(HIST_BINS, dtype=np.int64)
    grand_leak = 0
    grand_ttp = 0
    grand_ttc = np.zeros(4, dtype=np.int64)

    for label in range(len(CLASS_DIRS)):
        hist, leak, ttp, ttc = count_class_leak(
            by_label[label], split_of, STRICT_THR, LEAK_THR)
        grand_hist += hist
        grand_leak += leak
        grand_ttp += ttp
        grand_ttc += ttc

    # ---------------- 距离梯度直方图 ----------------
    print("==================== 距离梯度直方图（类内两两全比较） ====================")
    labels = [str(k) for k in range(HIST_BINS - 1)] + [">=8"]
    total_pairs = int(grand_hist.sum())
    for k, lab in enumerate(labels):
        cnt = int(grand_hist[k])
        pct = cnt / total_pairs * 100 if total_pairs else 0
        tag = "  <- 真重复关注区" if k <= STRICT_THR else ""
        print(f"  汉明距离 {lab:>3s} : {cnt:12,d} 对 ({pct:6.2f}%){tag}")
    print(f"  {'合计':>10s} : {total_pairs:12,d} 对")
    print("  说明：距离 0~1 为近精确重复；2~3 可能含同类相似；4~8 仅为同类普通相似。")
    print("==========================================================================")

    # ---------------- 跨切分泄漏（宽松，仅参考） ----------------
    print(f"\n[参考] 跨切分近重复对(距离<={LEAK_THR} 两端切分不同) : {grand_leak:,} 对")

    # ---------------- 测试图真孪生（金标准交叉验证） ----------------
    print("\n==================== 测试图真孪生（金标准交叉验证） ====================")
    print(f"测试集总张数                                       : {test_n:,}")
    print(f"测试图-训练图 真孪生对总数(距离<={STRICT_THR})       : {grand_ttp:,}")
    print("测试图中在训练集存在孪生的张数（按阈值梯度，去重）：")
    for k in range(4):
        c = int(grand_ttc[k])
        pct = c / test_n * 100 if test_n else 0
        note = "  <- 近精确重复(最强证据)" if k <= 1 else "  <- 可能含同类相似"
        print(f"   距离<={k} : {c:5,d} 张 ({pct:6.3f}%){note}")
    print("==========================================================================")

    # ---------------- 结论 ----------------
    print("\n==================== 结论 ====================")
    r0 = grand_ttc[0] / test_n if test_n else 0   # 距离=0（精确重复）
    r1 = grand_ttc[1] / test_n if test_n else 0   # 距离<=1（近精确重复）
    r3 = grand_ttc[3] / test_n if test_n else 0   # 距离<=3
    if grand_ttc[0] == 0:
        print(f"距离=0 的精确重复为 0（占比 {r0*100:.3f}%）→ 当前切分无精确泄漏，C 可不执行。")
    elif r1 > 0.20:
        print(f"近精确重复(距离<=1)占比 {r1*100:.3f}% (>20%)，成绩含明显水分 → 建议执行 C（分组切分重训）。")
    elif r1 < 0.05:
        print(f"近精确重复(距离<=1)占比 {r1*100:.3f}% (<5%)，80% 多为同类相似误判 → C 可不做。")
    else:
        print(f"近精确重复(距离<=1)占比 {r1*100:.3f}%，处于灰色地带 → 建议抽样目视确认后再定。")
    print(f"（对照：宽松阈值距离<=3 占比 {r3*100:.3f}%，多数应为同类相似，不作为金标准）")
    print("=============================================")


if __name__ == "__main__":
    main()
