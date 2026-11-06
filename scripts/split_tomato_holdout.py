# -*- coding: utf-8 -*-
"""
第 0 层：为番茄建立「更大、可复现」的真实域评估集。

背景
---------------------------------------------------------------
真实域评估集（PlantDoc 官方 test，去重后）只有 225 张，其中番茄仅 66 张。
样本太小的直接后果：之前三次实验（裁剪 / multicrop / 320 分辨率）的
McNemar 检验全部 p>0.05，**无法区分真实提升与随机波动**。

做法
---------------------------------------------------------------
从 `data/real_mix` 的 677 张番茄真实图（原本全部参与训练）里，按类分层
固定留出 120 张，只用于评估，不再进训练：

    ```mixed_v2.train``` = 原 mixed.train 去掉这 120 张
    ```realtest_v2.test``` = 原 realtest.test（225，番茄 66）+ 这 120（番茄合计 186）

为什么必须固定种子 + 落盘清单
---------------------------------------------------------------
对照组（真实域权重=1.0）与实验组（=4.0）必须评估在**同一批图**上，
否则 McNemar 配对检验无从谈起。清单一旦落盘，重跑默认复用，不会重新抽。

防泄漏校验
---------------------------------------------------------------
- 留出图来自 PlantDoc train，评估集原有 225 张来自 PlantDoc test，
  prepare_plantdoc_mix.py 已按 md5 剔除过与 train 重叠的 test 图；
- 本脚本再次用 md5 交叉核对，若发现重叠直接报错退出。

用法
    python -m scripts.split_tomato_holdout            # 首次生成（之后复用清单）
    python -m scripts.split_tomato_holdout --force    # 强制重新抽（会覆盖清单）
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CLASS_DIRS

INDEX_DIR = ROOT / "outputs" / "index"
MIXED_INDEX = INDEX_DIR / "dataset_index_mixed.json"
REALTEST_INDEX = INDEX_DIR / "dataset_index_realtest.json"

MANIFEST = INDEX_DIR / "tomato_holdout_manifest.json"
MIXED_V2 = INDEX_DIR / "dataset_index_mixed_v2.json"
REALTEST_V2 = INDEX_DIR / "dataset_index_realtest_v2.json"

TARGET = 120
SEED = 42
MIN_KEEP_PER_CLASS = 5      # 样本数 >= 该值的类至少留 1 张进评估集


def norm(p):
    return str(p).replace("\\", "/")


def tomato_ids():
    return {i for i, n in enumerate(CLASS_DIRS) if n.startswith("Tomato")}


def stratify(counts, target):
    """按类比例分配留出数，最大余数法补齐；小类保底 1 张。"""
    total = sum(counts.values())
    quota = {c: counts[c] * target / total for c in counts}
    alloc = {c: int(quota[c]) for c in counts}
    # 大样本类保底 1
    for c in counts:
        if counts[c] >= MIN_KEEP_PER_CLASS and alloc[c] == 0:
            alloc[c] = 1
    # 最大余数补齐到 target
    rest = target - sum(alloc.values())
    order = sorted(counts, key=lambda c: (quota[c] - int(quota[c])), reverse=True)
    i = 0
    while rest > 0 and order:
        c = order[i % len(order)]
        if alloc[c] < counts[c]:
            alloc[c] += 1
            rest -= 1
        i += 1
    while rest < 0:  # 保底把小类顶上去后可能超量，从最大的类回收
        c = max(alloc, key=lambda x: alloc[x])
        if alloc[c] > 0:
            alloc[c] -= 1
            rest += 1
        else:
            break
    return alloc


def md5_of(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="忽略已有清单，重新抽样")
    args = ap.parse_args()

    if MANIFEST.exists() and not args.force:
        man = json.loads(MANIFEST.read_text(encoding="utf-8"))
        hold = [(p, int(l)) for p, l in man["items"]]
        print(f"复用已有留出清单：{MANIFEST}（{len(hold)} 张，seed={man['seed']}）")
        print("如需重新抽样请加 --force")
    else:
        mixed = json.loads(MIXED_INDEX.read_text(encoding="utf-8"))
        tids = tomato_ids()
        pool = defaultdict(list)
        for p, l in mixed["train"]:
            if int(l) in tids and "real_mix" in norm(p):
                pool[int(l)].append(norm(p))
        for l in pool:
            pool[l].sort()          # 排序后再抽，保证同种子结果稳定
        counts = {l: len(v) for l, v in pool.items()}

        import random
        rng = random.Random(SEED)
        alloc = stratify(counts, TARGET)
        hold = []
        for l in sorted(pool):
            k = alloc.get(l, 0)
            chosen = rng.sample(pool[l], k) if k else []
            hold.extend([(p, l) for p in sorted(chosen)])

        per_class = {
            CLASS_DIRS[l]: {"total": counts[l], "holdout": alloc.get(l, 0)}
            for l in sorted(pool)
        }
        MANIFEST.write_text(json.dumps({
            "seed": SEED, "target": TARGET, "min_keep_per_class": MIN_KEEP_PER_CLASS,
            "per_class": per_class, "items": hold,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"已抽样并写入清单：{MANIFEST}")

    # ===================== 生成 v2 索引 =====================
    mixed = json.loads(MIXED_INDEX.read_text(encoding="utf-8"))
    rt = json.loads(REALTEST_INDEX.read_text(encoding="utf-8"))

    hold_paths = {p for p, _ in hold}
    assert len(hold_paths) == len(hold), "留出清单存在重复路径"

    train_v2 = [it for it in mixed["train"] if norm(it[0]) not in hold_paths]
    removed = len(mixed["train"]) - len(train_v2)

    # 防泄漏：留出图 vs 原评估集，按 md5 交叉核对
    rt_paths = {norm(p) for p, _ in rt["test"]}
    overlap_path = hold_paths & rt_paths
    if overlap_path:
        raise SystemExit(f"❌ 留出图与评估集路径重叠 {len(overlap_path)} 张：{list(overlap_path)[:3]}")
    rt_sigs = {md5_of(p) for p in rt_paths}
    dup = [p for p in hold_paths if md5_of(p) in rt_sigs]
    if dup:
        raise SystemExit(f"❌ 留出图与评估集内容重复（md5 相同）{len(dup)} 张：{dup[:3]}")

    test_v2 = rt["test"] + [[p, l] for p, l in hold]

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    json.dump({"train": train_v2, "val": mixed["val"], "test": mixed["test"]},
              MIXED_V2.open("w", encoding="utf-8"), ensure_ascii=False)
    json.dump({"train": [], "val": [], "test": test_v2},
              REALTEST_V2.open("w", encoding="utf-8"), ensure_ascii=False)

    # ===================== 报告 =====================
    tids = tomato_ids()
    n_tom_new = sum(1 for _, l in test_v2 if int(l) in tids)
    print("\n=== 结果 ===")
    print(f"混合训练 train: {len(mixed['train'])} → {len(train_v2)}（移出 {removed} 张）")
    print(f"真实域评估 test: {len(rt['test'])} → {len(test_v2)}"
          f"（番茄 {sum(1 for _, l in rt['test'] if int(l) in tids)} → {n_tom_new}）")
    print(f"防泄漏校验：路径重叠 0 / md5 重复 0（已核对 {len(hold_paths)} 张留出图）")
    print("\n逐类：")
    for c in sorted(hold and {l for _, l in hold}):
        tot = sum(1 for p, l in mixed["train"] if int(l) == c and "real_mix" in norm(p))
        h = sum(1 for _, l in hold if int(l) == c)
        print(f"  {CLASS_DIRS[c]:<55}{h:>4} / {tot:>4}  留出 {h / tot:.0%}")
    print(f"\n产物：\n  {MANIFEST.name}\n  {MIXED_V2.name}\n  {REALTEST_V2.name}")


if __name__ == "__main__":
    main()
