# -*- coding: utf-8 -*-
"""DigiGreen 真实农户手机照片 —— 外部域（跨域）评估。

数据特点：印度小农户用手机实拍后发给 Farmer.Chat，农学家逐张标注。
与训练/评估所用的 PlantDoc（网络抓取图）属于**不同来源域**，可用来回答
「模型换到农户手机实拍照上还能不能看」。

标签映射：只有落在 38 类体系内的诊断才计分；
叶螨/果实螟虫/粉虱/青枯病/缺素等不在体系内的诊断单列为「体系外」探针。
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.predict import Predictor  # noqa: E402
from src.config import class_name_zh  # noqa: E402

DG = ROOT / "data/_downloads/digigreen"

# 诊断名称 → 38 类目录名（只收录体系内可判分的）
MAP = {
    "Leaf Curl Virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Early Blight": "Tomato___Early_blight",
    "Late Blight": "Tomato___Late_blight",
    "Healthy": "Tomato___healthy",
}


def load_tomato_rows():
    """读标注，返回 {图片文件名: [(诊断, 详情), ...]}，只保留 crop 含 tomato 的。"""
    rows = list(csv.DictReader(open(DG / "annotations.csv", encoding="utf-8")))
    grouped = defaultdict(list)
    for r in rows:
        if "tomato" in (r.get("crop") or "").lower():
            grouped[r["image_file"]].append((r.get("diagnosis") or "").strip())
    return grouped


def resolve_labels(diagnoses):
    """把诊断列表解析为 (体系内类别集合, 是否含体系外诊断)。

    一条记录可能写多个诊断（如 "Leaf Curl Virus; Whiteflies"），
    分号拆分后逐个映射：能映射的进集合，映射不上的记体系外。
    """
    mapped, outside = set(), []
    for d in diagnoses:
        for tok in [t.strip() for t in (d or "").split(";") if t.strip()]:
            if tok in MAP:
                mapped.add(MAP[tok])
            else:
                outside.append(tok)
    return mapped, outside


def main():
    grouped = load_tomato_rows()
    pred = Predictor.get_instance()
    records = []

    for fname, diagnoses in sorted(grouped.items()):
        path = DG / "images" / Path(fname).name
        if not path.exists():
            continue
        try:
            img = Image.open(path).convert("RGB")
            img.load()  # verify() 抓不住截断图，必须 load
        except Exception as e:
            print(f"[跳过] {fname}: {e}")
            continue
        # 模拟用户上传时选择作物「番茄」
        results, ood = pred.predict_image_with_ood(img, top_k=3, crop="番茄")
        mapped, outside = resolve_labels(diagnoses)
        # 期望类别转中文名，才能与 results 里的 name_zh 比对
        expected_zh = [class_name_zh(c) for c in sorted(mapped)]
        top1 = (results[0].get("name_zh") if results else None)
        top3 = [r.get("name_zh") for r in results]
        records.append({
            "image": fname,
            "diagnoses": diagnoses,
            "expected": sorted(mapped),
            "expected_zh": expected_zh,
            "outside_taxonomy": outside,
            "scored": len(mapped) == 1,          # 单诊断才计分，多诊断只记录
            "top1": top1,
            "top3": top3,
            "hit": bool(expected_zh) and top1 in expected_zh,
            "hit3": any(e in top3 for e in expected_zh),
            "conf": round(float(results[0].get("confidence", 0)) if results else 0, 4),
            "conf_cal": round(float(results[0].get("confidence_calibrated", 0)) if results else 0, 4),
            "leaf_prob": round(float(ood.get("leaf_prob", 0)), 4),
            "rejected": bool(ood.get("likely_not_leaf")),
        })

    # ===== 汇总 =====
    scored = [r for r in records if r["scored"]]
    correct = [r for r in scored if r["hit"]]
    outside = [r for r in records if not r["expected"]]
    print(f"\n番茄图片总数: {len(records)}")
    print(f"  体系内可判分: {len(scored)}  Top-1 正确 {len(correct)} "
          f"({len(correct)/max(len(scored),1)*100:.1f}%)")
    print(f"  体系外诊断（螨/虫/缺素/青枯等）: {len(outside)}  "
          f"其中被拒识 {sum(1 for r in outside if r['rejected'])}")
    print("\n按类别：")
    per = Counter(r["expected"][0] if r["expected"] else "(体系外)" for r in scored)
    ok = Counter(r["expected"][0] for r in correct)
    for cls, n in per.most_common():
        print(f"  {cls:45s} n={n:3d}  正确={ok.get(cls,0):3d} "
              f"({ok.get(cls,0)/n*100:5.1f}%)")
    print("\n逐张明细：")
    for r in records:
        mark = "OK " if r["hit"] else ("-- " if not r["expected"] else "X  ")
        print(f"  {mark}{Path(r['image']).name:14s} 标注={','.join(r['diagnoses'])[:38]:38s} "
              f"Top1={str(r['top1'])[:16]:16s} conf={r['conf']:.3f} "
              f"leaf={r['leaf_prob']:.3f}{' [拒识]' if r['rejected'] else ''}")

    out = ROOT / "outputs/digigreen_tomato_eval.json"
    json.dump({"summary": {"total": len(records), "scored": len(scored),
                           "correct": len(correct),
                           "outside": len(outside),
                           "outside_rejected": sum(1 for r in outside if r["rejected"])},
               "records": records}, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n明细已写入 {out}")


if __name__ == "__main__":
    main()
