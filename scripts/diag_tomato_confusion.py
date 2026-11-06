# -*- coding: utf-8 -*-
"""
番茄内部混淆诊断：66 张真实图究竟错到哪个类去。

为什么必须先做这一步
--------------------
部署时用户会选「番茄」，作物先验会把 softmax 掩码到番茄 10 类，
也就是说**预测结果已经不可能落到其他作物**。因此番茄的 43.9% 本质上是
「番茄 10 类选 1」的准确率，错分方向全在番茄内部。

如果错分集中在少数几对上（如 细菌斑 ↔ Septoria ↔ 早疫），
说明是细粒度判别问题，可针对性设计（对比学习 / 专用高分辨率分支）；
如果错误分散，说明是数据量问题，只能补数据。

输出：混淆矩阵 + 每类的主要误判目标 Top-3。
"""
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image

from src.config import CLASS_DIRS, DISEASE_KNOWLEDGE_PATH
from src.model.predict import Predictor

SHORT = {
    "Bacterial_spot": "细菌斑",
    "Early_blight": "早疫",
    "Late_blight": "晚疫",
    "Leaf_Mold": "叶霉",
    "Septoria_leaf_spot": "斑枯",
    "Spider_mites Two-spotted_spider_mite": "红蜘蛛",
    "Target_Spot": "靶斑",
    "Tomato_Yellow_Leaf_Curl_Virus": "黄卷叶",
    "Tomato_mosaic_virus": "花叶病毒",
    "healthy": "健康",
}


def short_of(class_dir: str) -> str:
    tail = class_dir.replace("Tomato___", "")
    return SHORT.get(tail, tail[:6])


def main():
    pred = Predictor.get_instance()
    know = json.loads(Path(DISEASE_KNOWLEDGE_PATH).read_text(encoding="utf-8"))
    crop_of = lambda c: (know.get(CLASS_DIRS[c]) or {}).get("crop", "")

    idx = json.loads(Path("outputs/index/dataset_index_realtest.json").read_text(encoding="utf-8"))
    items = [(Path(p), int(c)) for p, c in idx["test"] if crop_of(int(c)) == "番茄"]

    tomato_ids = pred.resolve_crop("番茄")
    pair = Counter()          # (true, pred) -> n
    per_true = defaultdict(Counter)
    correct = 0

    for path, cid in items:
        im = Image.open(path).convert("RGB")
        res, _ = pred.predict_image_with_ood(im, top_k=1, crop="番茄")
        pc = res[0]["class_id"]
        pair[(cid, pc)] += 1
        per_true[cid][pc] += 1
        correct += int(pc == cid)

    print(f"番茄 {len(items)} 张，Top-1 {correct}/{len(items)} = {correct/len(items):.1%}")
    print(f"作物先验候选类 {len(tomato_ids)} 个，故错分必然落在番茄内部\n")

    # 只保留出现过的真实类，按准确率升序
    rows = sorted(per_true, key=lambda c: sum(v for k, v in per_true[c].items() if k != c) / max(1, sum(per_true[c].values())), reverse=True)
    cols = sorted(per_true, key=lambda c: CLASS_DIRS[c])

    print("混淆矩阵（行=真实，列=预测，只列实际出现过的类）")
    head = "真实 \\ 预测".ljust(12) + "".join(short_of(CLASS_DIRS[c]).rjust(7) for c in cols) + "   合计"
    print(head)
    print("-" * len(head))
    for t in rows:
        cnt = per_true[t]
        tot = sum(cnt.values())
        line = short_of(CLASS_DIRS[t]).ljust(12)
        for p in cols:
            n = cnt.get(p, 0)
            line += (str(n) if n else "·").rjust(7)
        line += f"{tot:>7}"
        print(line)

    print("\n各类主要误判方向（错分 ≥1 才列出）")
    for t in rows:
        cnt = per_true[t]
        tot = sum(cnt.values())
        right = cnt.get(t, 0)
        wrongs = [(p, n) for p, n in cnt.items() if p != t]
        wrongs.sort(key=lambda x: -x[1])
        if not wrongs:
            print(f"  {short_of(CLASS_DIRS[t]):<6}{right}/{tot}  全部正确")
            continue
        desc = "，".join(f"→{short_of(CLASS_DIRS[p])} {n}" for p, n in wrongs[:3])
        print(f"  {short_of(CLASS_DIRS[t]):<6}{right}/{tot}   错向: {desc}")

    # 双向最强混淆对
    print("\n最强的双向混淆对（互错次数合计）")
    seen = set()
    pairs = []
    for (a, b), n in pair.items():
        if a == b:
            continue
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        m = pair.get((b, a), 0)
        pairs.append((n + m, a, b, n, m))
    pairs.sort(reverse=True)
    for tot, a, b, n, m in pairs[:4]:
        print(f"  {short_of(CLASS_DIRS[a])} ↔ {short_of(CLASS_DIRS[b]):<6} 合计 {tot} 次"
              f"（{n} + {m}）")


if __name__ == "__main__":
    main()
