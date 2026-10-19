# -*- coding: utf-8 -*-
"""用 PlantDoc **test 划分**（网络采集的真实田间照片）评估模型准确性。

与实验室指标的区别
---------------------------------------------------------------
`docs/05` 的 99.27% 来自 PlantVillage：黑底、居中、单一叶片。
本脚本用的是 PlantDoc test：真实田间/网络照片，背景杂乱、光照多变、构图随意。
两者差距就是**域间隙**的定量体现，是任务书 6.3「模型适用范围与局限性」的实测依据。

输出
---------------------------------------------------------------
  outputs/web_eval_report.json   逐张结果 + 汇总
  控制台：整体 Top-1/Top-3、Wilson 95% 置信区间、逐类准确率、主要混淆对

用法
    py -3.12 scripts/eval_web_images.py
    py -3.12 scripts/eval_web_images.py --limit 50      # 快速抽检
"""

import argparse
import json
import math
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from PIL import Image, ImageFile

from src.config import CLASS_DIRS, class_name_zh, DISEASE_KNOWLEDGE_PATH
from src.model.predict import predict_image_with_ood

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "outputs" / "realtest_test" / "manifest.jsonl"
OUT = ROOT / "outputs" / "web_eval_report.json"


_KNOW = json.loads(Path(DISEASE_KNOWLEDGE_PATH).read_text(encoding="utf-8"))


def crop_of(cid):
    """取类别对应的作物名（用于「作物对、病害错」这类部分正确统计）。"""
    en = CLASS_DIRS[cid]
    return (_KNOW.get(en) or {}).get("crop") or en.split("___")[0]


def wilson(k, n, z=1.96):
    """Wilson 区间：小样本比例估计的置信区间，比正态近似更稳（尤其 p 接近 0/1）。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只测前 N 张（抽检用）")
    args = ap.parse_args()

    if not MANIFEST.exists():
        raise SystemExit("缺少清单，请先运行 py -3.12 scripts/fetch_plantdoc_test.py")

    items = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        items = items[: args.limit]
    print(f"待评估 {len(items)} 张（PlantDoc test 划分，未参与任何训练/调参）\n")

    rows = []
    for i, it in enumerate(items, 1):
        p = ROOT / it["path"]
        try:
            img = Image.open(p).convert("RGB")
            res, ood = predict_image_with_ood(img, top_k=3)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}] 跳过 {p.name}: {type(e).__name__}")
            continue
        pred = [r["class_id"] for r in res]
        rows.append({
            "path": it["path"],
            "gh_dir": it["gh_dir"],
            "true_id": it["class_id"],
            "true_name": class_name_zh(CLASS_DIRS[it["class_id"]]),
            "pred_id": pred[0],
            "pred_name": res[0]["name_zh"],
            "confidence": res[0]["confidence"],
            "top3": [{"id": c, "name": class_name_zh(CLASS_DIRS[c]), "prob": r["confidence"]}
                     for c, r in zip(pred, res)],
            "top1_ok": pred[0] == it["class_id"],
            "top3_ok": it["class_id"] in pred,
            # 作物认对、病害认错 —— 对用户仍有参考价值（至少知道是什么作物出了问题）
            "crop_ok": crop_of(pred[0]) == crop_of(it["class_id"]),
            "likely_not_leaf": ood["likely_not_leaf"],
            "leaf_prob": ood["leaf_prob"],
        })
        if i % 20 == 0:
            acc = sum(r["top1_ok"] for r in rows) / len(rows)
            print(f"  {i}/{len(items)}  当前 Top-1 = {acc:.1%}")

    n = len(rows)
    k1 = sum(r["top1_ok"] for r in rows)
    k3 = sum(r["top3_ok"] for r in rows)
    kc = sum(r["crop_ok"] for r in rows)
    kf = sum(r["likely_not_leaf"] for r in rows)
    lo1, hi1 = wilson(k1, n)
    lo3, hi3 = wilson(k3, n)
    loc, hic = wilson(kc, n)
    lof, hif = wilson(kf, n)

    # 逐类
    per = defaultdict(lambda: {"n": 0, "ok": 0})
    for r in rows:
        per[r["true_name"]]["n"] += 1
        per[r["true_name"]]["ok"] += int(r["top1_ok"])

    # 混淆对（真实 → 预测，仅错误项）
    conf = Counter((r["true_name"], r["pred_name"]) for r in rows if not r["top1_ok"])

    print("\n" + "=" * 78)
    print("PlantDoc test 划分 · 真实网络照片评估")
    print("=" * 78)
    print(f"  样本数            {n} 张（{len(per)} 个类别）")
    print(f"  Top-1 准确率      {k1/n:.1%}   ({k1}/{n})   95%CI [{lo1:.1%}, {hi1:.1%}]")
    print(f"  Top-3 准确率      {k3/n:.1%}   ({k3}/{n})   95%CI [{lo3:.1%}, {hi3:.1%}]")
    print(f"  作物级准确率      {kc/n:.1%}   ({kc}/{n})   95%CI [{loc:.1%}, {hic:.1%}]"
          f"   ← 作物认对即可，不要求病害对")
    print(f"  非叶片误判率      {kf/n:.1%}   ({kf}/{n})   95%CI [{lof:.1%}, {hif:.1%}]"
          f"   ← 真实叶片被提示「可能不是叶片」")
    print(f"  平均置信度        {sum(r['confidence'] for r in rows)/n:.3f}")
    # 错判时的置信度：区分「不确定地错」与「自信地错」
    wrong = [r["confidence"] for r in rows if not r["top1_ok"]]
    if wrong:
        print(f"  错判平均置信度    {sum(wrong)/len(wrong):.3f}   "
              f"（其中 {sum(1 for c in wrong if c > 0.9)/len(wrong):.0%} 高于 0.9 —— 自信地错最危险）")

    print("\n  逐类准确率（样本数 ≥3 的类别）")
    print(f"    {'类别':<22}{'n':>4}{'correct':>9}{'acc':>8}")
    for name, d in sorted(per.items(), key=lambda kv: (-kv[1]["n"], kv[0])):
        if d["n"] >= 3:
            print(f"    {name:<22}{d['n']:>4}{d['ok']:>9}{d['ok']/d['n']:>8.0%}")

    if conf:
        print("\n  主要混淆（真实 → 预测，出现 ≥2 次）")
        for (t, p), c in conf.most_common(8):
            if c >= 2:
                print(f"    {t} → {p}   ×{c}")

    OUT.write_text(json.dumps({
        "source": "PlantDoc test split (CC BY 4.0)",
        "n": n,
        "top1": k1 / n, "top1_ci": [lo1, hi1],
        "top3": k3 / n, "top3_ci": [lo3, hi3],
        "crop_level": kc / n, "crop_level_ci": [loc, hic],
        "false_flagged": kf / n, "false_flagged_ci": [lof, hif],
        "per_class": {k: v for k, v in per.items()},
        "confusions": [{"true": t, "pred": p, "count": c} for (t, p), c in conf.most_common()],
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  明细已写入 {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
