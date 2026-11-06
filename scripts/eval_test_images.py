# -*- coding: utf-8 -*-
"""test_images/ 真实标签批量评估（进程内 Predictor，不走 API，避免限流）。

- 标签来源：文件名（NN_作物_标签.jpg / TNN_作物_标签.jpg）
- 作物先验：与文件名中的作物一致（模拟用户上传时选作物）
- 负样本（03_负样本拒识）：不传 crop，检查 likely_not_leaf
- 输出：逐张结果表 + 分组汇总，写入 outputs/test_images_eval.json
"""
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # 保证可以从项目根导入 src.*

from src.model.predict import Predictor  # noqa: E402


def norm(s: str) -> str:
    """标签归一化：去空格 + 去掉全部括号（全/半角），便于字符串比对。"""
    for ch in (" ", "（", "）", "(", ")"):
        s = s.replace(ch, "")
    return s or ""


# 作物中文名清单（与 GET /crops 一致；长名在前，避免「桃」截胡「马铃薯」等前缀匹配）
CROPS = ["马铃薯", "树莓", "蓝莓", "草莓", "甜椒", "大豆", "南瓜", "樱桃",
         "葡萄", "玉米", "番茄", "苹果", "桃"]


def parse_name(path: Path):
    """从文件名解析 (作物, 期望标签)。

    三段式 NN_作物_标签 → 直接取；两段式 NN_标签（00_快速测试）→
    用「标签前缀匹配作物清单」推断作物（模拟用户上传时选了作物）。
    """
    stem = path.stem  # 例：01_苹果黑星病 / T11_番茄_Septoria叶斑病
    parts = stem.split("_")
    label = "_".join(parts[2:]) if len(parts) >= 3 else parts[-1]
    crop = None
    if len(parts) >= 3:
        crop = parts[1]
    else:
        # 前缀匹配：标签以作物名开头（如 苹果黑星病 → 苹果）
        for c in CROPS:
            if label.startswith(c):
                crop = c
                break
    return crop, label


def hit(expected: str, results) -> bool:
    """Top-1 是否命中：病名主体（去掉作物前缀后的标签）是 Top-1 名_zh 或别名的子串。"""
    if not results or not expected:
        return False
    top = results[0]
    exp = norm(expected)
    return exp in {norm(top.get("name_zh", "")), norm(top.get("alias", ""))} or \
        any(exp in norm(top.get(k, "")) for k in ("name_zh", "alias"))


def in_top3(expected: str, results) -> bool:
    """期望标签是否出现在 Top-3（病名主体与名_zh / 别名做子串比对）。"""
    if not expected:
        return False
    exp = norm(expected)
    for r in results:
        if exp in {norm(r.get("name_zh", "")), norm(r.get("alias", ""))} or \
                any(exp in norm(r.get(k, "")) for k in ("name_zh", "alias")):
            return True
    return False


def main():
    predictor = Predictor.get_instance()
    base = ROOT / "test_images"
    rows = []
    groups = {"01_叶片识别": [], "02_番茄专项": [], "00_快速测试": []}

    for sub in ["00_快速测试", "01_叶片识别", "02_番茄专项"]:
        d = base / sub
        if not d.exists():
            continue
        # Windows 下 glob 大小写不敏感，jpg/JPG 会重复匹配 —— 用 set 去重
        for f in sorted(set(d.glob("*.jpg")) | set(d.glob("*.JPG"))):
            crop, expected = parse_name(f)
            img = Image.open(f)
            results, ood = predictor.predict_image_with_ood(img, top_k=3, crop=crop)
            row = {
                "group": sub,
                "file": f.name,
                "crop": crop,
                "expected": expected,
                "top1": results[0]["name_zh"] if results else "",
                "top1_conf_raw": results[0]["confidence"] if results else 0,
                "conf_cal": ood["confidence_calibrated"],
                "reliability": ood["reliability"],
                "leaf_prob": ood["leaf_prob"],
                "likely_not_leaf": ood["likely_not_leaf"],
                "crop_applied": ood["crop_applied"],
                "top3": [r["name_zh"] for r in results],
                "correct": hit(expected, results),
                "top3_hit": in_top3(expected, results),
            }
            rows.append(row)
            groups[sub].append(row)
            mark = "OK " if row["correct"] else ("t3 " if row["top3_hit"] else "MISS")
            print(f"[{mark}] {f.name:42s} 期望={expected or '-':20s} "
                  f"预测={row['top1']:18s} 校准置信={row['conf_cal']:.3f} "
                  f"({row['reliability']})")

    # 负样本：应被 OOD 拦截（likely_not_leaf=True）
    neg = base / "03_负样本拒识"
    neg_rows = []
    if neg.exists():
        for f in sorted(neg.glob("*.jpg")):
            img = Image.open(f)
            results, ood = predictor.predict_image_with_ood(img, top_k=3)
            row = {
                "file": f.name,
                "top1": results[0]["name_zh"] if results else "",
                "leaf_prob": ood["leaf_prob"],
                "leaf_prob_source": ood["leaf_prob_source"],
                "likely_not_leaf": ood["likely_not_leaf"],
                "mean_chroma": ood["mean_chroma"],
                "grayscale": ood["grayscale"],
                "blocked": bool(ood["likely_not_leaf"]),
            }
            neg_rows.append(row)
            mark = "拦截" if row["blocked"] else "漏放"
            print(f"[{mark}] {f.name:42s} leaf_prob={row['leaf_prob']:.4f} "
                  f"top1误给={row['top1']}")

    # ============ 汇总 ============
    summary = {}
    for sub, rs in groups.items():
        if not rs:
            continue
        n = len(rs)
        summary[sub] = {
            "张数": n,
            "Top1正确": sum(r["correct"] for r in rs),
            "Top3命中": sum(r["top3_hit"] for r in rs),
            "作物先验生效": sum(r["crop_applied"] for r in rs),
        }
    summary["03_负样本拒识"] = {
        "张数": len(neg_rows),
        "成功拦截": sum(r["blocked"] for r in neg_rows),
    }
    total = sum(len(r) for r in groups.values())
    correct = sum(r["correct"] for rs in groups.values() for r in rs)
    summary["总计"] = {"张数": total, "Top1正确": correct,
                       "Top1准确率": round(correct / total, 4) if total else 0}

    out = {"summary": summary, "rows": rows, "negatives": neg_rows}
    out_path = ROOT / "outputs" / "test_images_eval.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n==== 汇总 ====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"明细已写入 {out_path}")


if __name__ == "__main__":
    main()
