"""作物先验（crop prior）的端到端评估：不开 / 开 两种口径对比。

背景
    120 张 PlantDoc test 真实照片上 Top-1 仅 17.5%。错误主因是「只看斑点纹理、
    不看作物」——苹果雪松锈病的橙黄斑点被判成玉米灰斑病，置信度 0.96~1.00。
    若用户先指定作物，把 softmax 限制在该作物的候选类内，应能消除这类跨作物错误。

评估口径（三个都必须报，否则会虚增）
    1. baseline          不指定作物（全 38 类）
    2. crop_oracle       指定「真实作物」—— 这是**上界**，假设用户永远选对
    3. crop_predicted    指定「模型预测的作物」—— 现实下限，因为模型自动定作物很差

    另外把「该作物只有 1 个候选类」的样本单独剔除。这类约束等于强制答对
    （蓝莓/树莓/大豆/南瓜/柑橘在 38 类里各只有一个健康类），计入会虚增收益。

用法
    py -3.12 scripts/eval_crop_prior.py
    py -3.12 scripts/eval_crop_prior.py --limit 60     # 快速抽样
"""

import argparse
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageFile

from src.config import CLASS_DIRS, DISEASE_KNOWLEDGE_PATH
from src.model.predict import Predictor

ImageFile.LOAD_TRUNCATED_IMAGES = True

MANIFEST = Path("outputs/realtest_test/manifest.jsonl")
REPORT = Path("outputs/crop_prior_report.json")


def wilson(k, n, z=1.96):
    """Wilson 得分区间：小样本下比正态近似可靠（不会给出越界或过窄的区间）。"""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张（0 = 全部）")
    args = ap.parse_args()

    know = json.loads(Path(DISEASE_KNOWLEDGE_PATH).read_text(encoding="utf-8"))
    crop_of = lambda cid: (know.get(CLASS_DIRS[cid]) or {}).get("crop", "")

    by_crop = defaultdict(list)
    for i in range(len(CLASS_DIRS)):
        by_crop[crop_of(i)].append(i)

    items = [
        json.loads(x)
        for x in MANIFEST.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    if args.limit:
        items = items[: args.limit]

    pred = Predictor.get_instance()
    print(f"样本 {len(items)} 张 | 作物 {len(by_crop)} 种 | 作物先验已启用: "
          f"{bool(pred.crop_class_ids)}")

    rows = []
    for n, it in enumerate(items, 1):
        try:
            img = Image.open(it["path"]).convert("RGB")
        except Exception:
            continue
        tid = it["class_id"]
        t_crop = crop_of(tid)

        # 口径 1：不指定作物
        r_base = pred.predict_image(img, top_k=3)
        # 口径 2：指定真实作物（上界）
        r_oracle = pred.predict_image(img, top_k=3, crop=t_crop)
        # 口径 3：用模型预测的作物再约束（现实下限）
        pred_crop = r_base[0]["crop"]
        r_predcrop = pred.predict_image(img, top_k=3, crop=pred_crop)

        rows.append({
            "path": it["path"],
            "true_id": tid,
            "true_name": CLASS_DIRS[tid],
            "true_crop": t_crop,
            "n_candidates": len(by_crop[t_crop]),
            "base_id": r_base[0]["class_id"],
            "base_crop": pred_crop,
            "base_conf": r_base[0]["confidence"],
            "oracle_id": r_oracle[0]["class_id"],
            "oracle_conf": r_oracle[0]["confidence"],
            "predcrop_id": r_predcrop[0]["class_id"],
            "predcrop_conf": r_predcrop[0]["confidence"],
        })
        if n % 30 == 0:
            print(f"  ... {n}/{len(items)}")

    def acc(key, subset):
        if not subset:
            return 0.0, 0, 0
        k = sum(1 for r in subset if r[key] == r["true_id"])
        return k / len(subset), k, len(subset)

    multi = [r for r in rows if r["n_candidates"] > 1]
    single = [r for r in rows if r["n_candidates"] == 1]

    print("\n" + "=" * 74)
    print("作物先验效果（Top-1 准确率）")
    print("=" * 74)
    print(f"{'口径':<26}{'全部':>16}{'多候选作物':>18}{'单候选作物':>14}")
    for label, key in (
        ("① 不指定作物（线上默认）", "base_id"),
        ("② 指定真实作物（上界）", "oracle_id"),
        ("③ 指定模型预测作物（下限）", "predcrop_id"),
    ):
        a_all, k_all, n_all = acc(key, rows)
        a_m, k_m, n_m = acc(key, multi)
        a_s, k_s, n_s = acc(key, single)
        print(f"{label:<26}{a_all:>7.1%} ({k_all:>2}/{n_all})"
              f"{a_m:>10.1%} ({k_m:>2}/{n_m})"
              f"{a_s:>9.1%} ({k_s:>2}/{n_s})")

    a_m, k_m, n_m = acc("oracle_id", multi)
    a_b, _, _ = acc("base_id", multi)
    lo, hi = wilson(k_m, n_m)
    print(f"\n多候选作物上的真实收益：{a_b:.1%} → {a_m:.1%}"
          f"（{a_m - a_b:+.1%}），95%CI [{lo:.1%}, {hi:.1%}]")

    # 作物识别准确率（决定「自动定作物」这条路是否可行）
    crop_ok = sum(1 for r in rows if r["base_crop"] == r["true_crop"])
    print(f"模型自动识别作物准确率：{crop_ok / len(rows):.1%}（{crop_ok}/{len(rows)}）")

    # 作物内 Top-3：把候选限制到作物后，Top-3 是否覆盖正确答案
    print("\n各作物上「指定真实作物」的 Top-1（样本数 ≥3 才列出）：")
    for c, n_img in Counter(r["true_crop"] for r in rows).most_common():
        if n_img < 3:
            continue
        sub = [r for r in rows if r["true_crop"] == c]
        a_bc = sum(1 for r in sub if r["base_id"] == r["true_id"]) / len(sub)
        a_oc = sum(1 for r in sub if r["oracle_id"] == r["true_id"]) / len(sub)
        print(f"  {c:<6} n={n_img:<3} {len(by_crop[c])} 候选   "
              f"{a_bc:>6.1%} → {a_oc:>6.1%}")

    report = {
        "n": len(rows),
        "n_multi": len(multi),
        "n_single": len(single),
        "accuracy": {
            "baseline_all": acc("base_id", rows)[0],
            "baseline_multi": acc("base_id", multi)[0],
            "oracle_all": acc("oracle_id", rows)[0],
            "oracle_multi": acc("oracle_id", multi)[0],
            "oracle_multi_ci": wilson(*acc("oracle_id", multi)[1:]),
            "predcrop_all": acc("predcrop_id", rows)[0],
            "predcrop_multi": acc("predcrop_id", multi)[0],
            "crop_detect_acc": crop_ok / len(rows),
        },
        "rows": rows,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细已写入 {REPORT}")


if __name__ == "__main__":
    main()
