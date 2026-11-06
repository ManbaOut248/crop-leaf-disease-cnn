"""DigiGreen 全量 989 张「整株取景 / 未知作物」能力边界评估（对应 docs/25）。

为什么做这个评估
----------------
docs/23 只在 47 张番茄切片上跑了 17.1%，样本太小、只覆盖一个作物，无法支撑
「模型在真实田间照片上到底能干什么」的结论。DigiGreen 的 989 张图恰好提供了
一个更严格的能力边界探针：

    74 种作物 / 145 种诊断 / 印度农户手机实拍（非实验室、非摆拍）

其中只有 4 种作物（番茄、马铃薯、玉米、柑橘等 ≈227 张）落在本项目 38 类体系内，
其余 760+ 张是模型**从未见过**的作物（辣椒、茄子、芒果、水稻、香蕉……）。
而本项目的 OOD 头只回答「这是不是叶片」，不回答「这是不是我认识的作物」——
所以未知作物会被强行映射到 38 类里最像的那一个。本脚本就是要把这件事量化。

双模式设计
----------
    no_prior   ：不指定作物。对应用户不选作物时的默认路径，也是未知作物唯一可能的路径
    crop_prior ：指定**正确**作物（仅体系内作物可跑）。测作物先验在真实田间的上限收益

输出
----
    outputs/digigreen_full_eval.jsonl  逐图逐模式一条记录（可断点续跑）
    stdout                             汇总指标（含 Wilson 95% 置信区间）
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from src.model.predict import Predictor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/_downloads/digigreen"
ANN = DATA / "annotations.csv"
OUT = ROOT / "outputs/digigreen_full_eval.jsonl"

def norm(s: str) -> str:
    """诊断/类名归一化：小写、去空格、全角括号转半角。

    ⚠️ DX_MAP 的键和这里的查询**必须同处一个归一化空间**：早先 DX_MAP 写了
    "early blight"（带空格）而查询是 norm(诊断)="earlyblight"，结果所有带空格的
    诊断都匹配不上，48 张可计分样本里 100% 是 "healthy" —— 病害图被静默漏光，
    诊断准确率变成「健康图准确率」。所以 DX_MAP 在定义后统一用 norm() 重建键。
    """
    return (s or "").strip().lower().replace(" ", "").replace("（", "(").replace("）", ")")


# ---------------------------------------------------------------------------
# 1. 作物映射：DigiGreen 作物名 → 本项目 14 种作物中文名
#    只映射**确实存在于 38 类体系**的作物；其余一律视为「未知作物」。
# ---------------------------------------------------------------------------
CROP_MAP = {
    "tomato": "番茄",
    "potato": "马铃薯",
    "maize": "玉米",
    "citrus": "柑橘",
    "strawberry": "草莓",
    "soybean": "大豆",
    "pepper": "甜椒",
    "squash": "南瓜",
}

# ---------------------------------------------------------------------------
# 2. 诊断映射：只在「诊断确实属于我们体系里的那一类」时才计分
#    宁可少算（映射不上就当体系外），也不硬凑 —— 映射错了会把噪声当信号。
# ---------------------------------------------------------------------------
DX_MAP = {
    ("番茄", "healthy"): "番茄（健康）",
    ("番茄", "leaf curl virus"): "番茄黄化曲叶病毒病",
    ("番茄", "early blight"): "番茄早疫病",
    ("番茄", "late blight"): "番茄晚疫病",
    ("西红柿", "healthy"): "番茄（健康）",          # 容错：不同拼法
    ("马铃薯", "healthy"): "马铃薯（健康）",
    ("马铃薯", "early blight"): "马铃薯早疫病",
    ("马铃薯", "late blight"): "马铃薯晚疫病",
    ("玉米", "healthy"): "玉米（健康）",
    # Turcicum leaf blight（大斑病）= Exserohilum turcicum 引起 = 北方叶枯病
    ("玉米", "turcicum leaf blight"): "玉米北方叶枯病",
}
# 键统一过 norm()：否则带空格的诊断（early blight / leaf curl virus 等）永远查不到，
# 会让「可计分样本」退化成只剩 healthy 的假象（详见 norm() 的注释）。
DX_MAP = {(crop, norm(dx)): zh for (crop, dx), zh in DX_MAP.items()}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 置信区间（小样本比率比正态近似可靠，本评估大量 n<50 的格子需要它）。"""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    # 端点钳位：k=0 时下界会算成 -0.0，直接显示负数会让人以为算错了
    return (max(0.0, (c - r) / d * 100), min(100.0, (c + r) / d * 100))


def pct(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100.0 * k / n:5.1f}%  [95%CI {lo:.1f}~{hi:.1f}]" if n else "0/0"


def load_annotations() -> dict[str, list[dict]]:
    """按图片聚合标注（一张图可能有多条诊断，多诊断的图不计入诊断准确率）。"""
    by_img: dict[str, list[dict]] = {}
    with open(ANN, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img = (row.get("image_file") or "").strip()
            if img:
                by_img.setdefault(img, []).append(row)
    return by_img


def build_tasks(by_img: dict[str, list[dict]]) -> list[dict]:
    """把标注整理成待推理任务：每张图一条，含真实作物、真值类别、是否可计分。"""
    tasks = []
    for img, rows in sorted(by_img.items()):
        crop_raw = (rows[0].get("crop") or "").strip()
        crop_zh = CROP_MAP.get(norm(crop_raw))          # None = 未知作物
        dxs = sorted({(r.get("diagnosis") or "").strip() for r in rows if (r.get("diagnosis") or "").strip()})
        expected = None
        if len(dxs) == 1 and crop_zh:
            expected = DX_MAP.get((crop_zh, norm(dxs[0])))
        tasks.append({
            "image": img,
            "crop_raw": crop_raw,
            "crop_zh": crop_zh,
            "diagnoses": dxs,
            "expected": expected,
            "scorable": bool(expected),
        })
    return tasks


def run_eval(tasks: list[dict], out_path: Path) -> None:
    """逐图逐模式推理，结果 append 到 JSONL（重跑自动跳过已完成项）。"""
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add((rec["image"], rec["mode"]))
                except json.JSONDecodeError:
                    continue
        print(f"[续跑] 已完成 {len(done)} 条记录")

    pred = Predictor.get_instance()
    print(f"[设备] {pred.device} | 待处理图片 {len(tasks)} 张")

    t0 = time.time()
    processed = 0
    with open(out_path, "a", encoding="utf-8") as fout:
        for i, task in enumerate(tasks, 1):
            path = DATA / task["image"]
            if not path.exists():
                continue
            try:
                img = Image.open(path).convert("RGB")
            except Exception as e:  # 损坏图跳过但不静默
                print(f"  [跳过] {task['image']}: {e}")
                continue

            # 模式一：不指定作物（默认路径）
            modes = [("no_prior", None)]
            # 模式二：指定正确作物（仅体系内作物才有意义）
            if task["crop_zh"]:
                modes.append(("crop_prior", task["crop_zh"]))

            for mode, crop in modes:
                if (task["image"], mode) in done:
                    continue
                results, ood = pred.predict_image_with_ood(img, top_k=3, crop=crop)
                top1 = results[0] if results else {}
                rec = {
                    "image": task["image"],
                    "mode": mode,
                    "crop_raw": task["crop_raw"],
                    "crop_zh": task["crop_zh"],
                    "diagnoses": task["diagnoses"],
                    "expected": task["expected"],
                    "scorable": task["scorable"],
                    "pred_name_zh": top1.get("name_zh"),
                    "pred_crop": top1.get("crop"),
                    "pred_en": top1.get("name_en"),
                    "top3": [r.get("name_zh") for r in results],
                    "conf_raw": top1.get("confidence"),
                    "conf_cal": ood.get("confidence_calibrated"),
                    "reliability": ood.get("reliability"),
                    "leaf_prob": ood.get("leaf_prob"),
                    "rejected": bool(ood.get("likely_not_leaf")),
                    "grayscale": bool(ood.get("grayscale")),
                    "crop_applied": bool(ood.get("crop_applied")),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                processed += 1

            fout.flush()
            if i % 50 == 0 or i == len(tasks):
                el = time.time() - t0
                print(f"  {i}/{len(tasks)} 张 | 新增 {processed} 条 | 用时 {el:.0f}s", flush=True)

    print(f"[完成] 本次新增 {processed} 条，文件 {out_path}")


def analyze(out_path: Path, tasks: list[dict]) -> None:
    """汇总指标：已知作物诊断准确率 / 未知作物误接受 / 拒识率 / 可靠性分布。"""
    recs: list[dict] = []
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    by_mode: dict[str, list[dict]] = {}
    for r in recs:
        by_mode.setdefault(r["mode"], []).append(r)

    n_img = len({r["image"] for r in recs})
    print("=" * 78)
    print(f"DigiGreen 全量评估 | 图片 {n_img} 张 | 记录 {len(recs)} 条")
    print("=" * 78)

    for mode in ("no_prior", "crop_prior"):
        rows = by_mode.get(mode, [])
        if not rows:
            continue
        print(f"\n########## 模式：{mode}（n={len(rows)}）##########")

        # ---- 总体：拒识与可靠性分布 ----
        rej = sum(1 for r in rows if r["rejected"])
        print(f"  拒识（判为非叶片）：{pct(rej, len(rows))}")
        rel = Counter(r["reliability"] for r in rows)
        print("  可靠性分布：" + " | ".join(f"{k}:{v} ({100*v/len(rows):.1f}%)" for k, v in rel.most_common()))
        hi = sum(1 for r in rows if r["reliability"] == "high")
        print(f"  高置信输出（reliability=high）：{pct(hi, len(rows))}")

        # ---- A. 体系内作物 ----
        known = [r for r in rows if r["crop_zh"]]
        print(f"\n  -- A. 体系内作物（{len(known)} 张）--")
        if known:
            crop_hit = sum(1 for r in known if r["pred_crop"] == r["crop_zh"])
            print(f"  作物判对（top1 的作物 == 真实作物）：{pct(crop_hit, len(known))}")
            score = [r for r in known if r["scorable"]]
            if score:
                h1 = sum(1 for r in score if norm(r["pred_name_zh"]) == norm(r["expected"]))
                h3 = sum(1 for r in score if any(norm(t) == norm(r["expected"]) for t in r["top3"]))
                print(f"  诊断 Top-1（可映射诊断 n={len(score)}）：{pct(h1, len(score))}")
                print(f"  诊断 Top-3：{pct(h3, len(score))}")
                # ⚠️ Top-3 在作物先验下是构造性偏高的：候选类只有 N 个时 Top-3 覆盖率天然是
                # min(3,N)/N（马铃薯仅 3 类 → 必然 100%）。所以必须逐作物拆开，并给出随机基线。
                base = {"番茄": "随机 Top-3 基线 30%", "马铃薯": "随机 Top-3 基线 100%（仅 3 类）",
                        "玉米": "随机 Top-3 基线 75%"}
                for crop in sorted({r["crop_zh"] for r in score}):
                    sub = [r for r in score if r["crop_zh"] == crop]
                    k1 = sum(1 for r in sub if norm(r["pred_name_zh"]) == norm(r["expected"]))
                    k3 = sum(1 for r in sub if any(norm(t) == norm(r["expected"]) for t in r["top3"]))
                    print(f"      {crop}: Top-1 {pct(k1, len(sub))} | Top-3 {pct(k3, len(sub))}"
                          f"  ({base.get(crop, '')})")
                # 健康 vs 病害分开：健康误报方向更危险（会诱导施药）
                healthy = [r for r in score if "健康" in r["expected"]]
                diseased = [r for r in score if "健康" not in r["expected"]]
                for label, sub in (("真值=健康", healthy), ("真值=病害", diseased)):
                    if sub:
                        k = sum(1 for r in sub if norm(r["pred_name_zh"]) == norm(r["expected"]))
                        print(f"      {label}: Top-1 {pct(k, len(sub))}")

        # ---- B. 未知作物（模型从未见过） ----
        unknown = [r for r in rows if not r["crop_zh"]]
        print(f"\n  -- B. 未知作物（{len(unknown)} 张，模型从未见过）--")
        if unknown:
            urej = sum(1 for r in unknown if r["rejected"])
            # 「硬答」必须排除已被拒识的：被拒识的图前端会提示「可能不是叶片」，
            # 不构成静默误接受，混进来会把风险说大。
            hard = [r for r in unknown if not r["rejected"]]
            uhi = sum(1 for r in hard if r["reliability"] == "high")
            umed = sum(1 for r in hard if r["reliability"] in ("high", "medium"))
            print(f"  拒识：{pct(urej, len(unknown))}  ← 理想应接近 100%，实测说明 OOD 头不认作物")
            print(f"  未被拒识（= 给了确定性答案）：{len(hard)}/{len(unknown)} = {100*len(hard)/len(unknown):.1f}%")
            print(f"  其中高置信硬答（high）：{pct(uhi, len(hard))}")
            print(f"  其中中/高置信硬答（high+medium）：{pct(umed, len(hard))}")
            pc = Counter(r["pred_crop"] for r in hard)
            top = " | ".join(f"{k or '?'}:{v}" for k, v in pc.most_common(6))
            print(f"  被判成的作物分布：{top}")
            # 逐作物看：哪些作物最容易被高置信误判（喂给产品做「不支持作物」清单）
            print("  高风险作物（n>=15 的高置信硬答率 Top5）：")
            per = []
            for crop in {r["crop_raw"] for r in hard}:
                sub = [r for r in hard if r["crop_raw"] == crop]
                if len(sub) >= 15:  # 先按样本量过滤再排序，否则 top5 会被 n=1 的 100% 占满
                    per.append((sum(1 for r in sub if r["reliability"] == "high") / len(sub), len(sub), crop))
            for rate, n, crop in sorted(per, reverse=True)[:5]:
                print(f"      {crop:16s} n={n:3d} 高置信硬答率 {100*rate:.0f}%")
            # 抽 3 条最高置信的硬答，供文档举例（并标注置信度，便于核对）
            print("  最高置信硬答样例：")
            for r in sorted(hard, key=lambda a: -(a["conf_cal"] or 0))[:3]:
                print(f"      {r['crop_raw']:14s} 真诊断={r['diagnoses'][0][:24]:26s} -> "
                      f"{r['pred_name_zh']:12s} conf={r['conf_cal']:.3f}")

        # ---- C. 体系内作物但诊断在体系外（蚜虫/缺素/青枯等） ----
        outside = [r for r in rows if r["crop_zh"] and not r["scorable"]]
        if outside:
            hard_o = [r for r in outside if not r["rejected"]]
            ohi = sum(1 for r in hard_o if r["reliability"] == "high")
            print(f"\n  -- C. 体系内作物 / 诊断在体系外（{len(outside)} 张）--")
            print(f"  仍给出高置信答案（未被拒识）：{pct(ohi, len(hard_o))}"
                  f"  ← 这类本应回答「不支持该病害」")


def main() -> None:
    if not ANN.exists():
        raise SystemExit(f"未找到标注文件：{ANN}")
    by_img = load_annotations()
    tasks = build_tasks(by_img)
    scorable = sum(1 for t in tasks if t["scorable"])
    known = sum(1 for t in tasks if t["crop_zh"])
    print(f"[标注] 图片 {len(tasks)} 张 | 体系内作物 {known} 张 | 可计分诊断 {scorable} 张")

    if "--analyze" not in sys.argv:
        run_eval(tasks, OUT)
    analyze(OUT, tasks)


if __name__ == "__main__":
    main()
