# -*- coding: utf-8 -*-
"""
按「农作物」聚合识别率，回答「哪个作物识别成功率最低」。

三张表，口径不同不可混读：
  表 1  实验室域（PlantVillage 测试集 4914 张）—— 来自 outputs/eval_report.json
  表 2  真实域·微调前基线（PlantDoc test 120 张，不指定作物）—— 来自 outputs/web_eval_report.json
  表 3  真实域·当前部署权重（PlantDoc 去重留出集 225 张，带作物先验）
        —— 来自 backups/_realtest_eval_225.txt，由
           `python -m scripts.eval_realtest --index outputs/index/dataset_index_realtest.json`
           的输出重定向保存而来（见文件末尾说明）

用法：X:\\Anconda3\\python.exe scripts/report_per_crop.py
"""

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

CROP_PREFIX = {
    "Apple": "苹果", "Blueberry": "蓝莓", "Cherry": "樱桃", "Corn": "玉米",
    "Grape": "葡萄", "Orange": "柑橘", "Peach": "桃", "Pepper": "甜椒",
    "Potato": "马铃薯", "Raspberry": "树莓", "Soybean": "大豆", "Squash": "南瓜",
    "Strawberry": "草莓", "Tomato": "番茄",
}
ORDER = ["番茄", "玉米", "马铃薯", "甜椒", "苹果", "葡萄", "草莓", "桃",
         "蓝莓", "樱桃", "树莓", "大豆", "南瓜", "柑橘"]

LINE = re.compile(r"^\s*(\S.*?)\s+([\d.]+)%\s+\((\d+)/(\d+)\)\s*$")


def crop_by_en(en: str) -> str:
    prefix = en.split("___")[0].split(",")[0]
    return next((v for k, v in CROP_PREFIX.items() if prefix.startswith(k)), prefix)


def table(title: str, agg: dict, note: str = "") -> None:
    print(f"\n{title}")
    if note:
        print(f"  {note}")
    print(f"{'作物':<7}{'正确/总数':>12}{'准确率':>10}{'错分':>6}")
    print("-" * 40)
    keys = [c for c in ORDER if c in agg] + [c for c in agg if c not in ORDER]
    for c in keys:
        ok, n = agg[c]
        if n == 0:
            continue
        print(f"{c:<7}{f'{ok}/{n}':>12}{ok / n * 100:>9.1f}%{n - ok:>6}")
    print("-" * 40)
    tok = sum(v[0] for v in agg.values())
    tn = sum(v[1] for v in agg.values())
    if tn:
        print(f"{'合计':<7}{f'{tok}/{tn}':>12}{tok / tn * 100:>9.1f}%{tn - tok:>6}")


def main() -> int:
    # ---------- 表 1：实验室域 ----------
    lab = json.load(open(ROOT / "outputs/eval_report.json", encoding="utf-8"))["per_class"]
    lab_agg = collections.defaultdict(lambda: [0, 0])
    for _, v in lab.items():
        c = crop_by_en(v["en"])
        lab_agg[c][0] += round(v["recall"] * v["support"])
        lab_agg[c][1] += v["support"]
    table("表 1  实验室域（PlantVillage 测试集 4914 张）", lab_agg,
          "每个作物内部所有类别合并统计；总体 99.27%")

    # ---------- 表 2：真实域微调前基线 ----------
    from src.config import CLASS_NAMES_ZH
    zh2crop = {zh: crop_by_en(en) for en, zh in CLASS_NAMES_ZH.items()}
    real = json.load(open(ROOT / "outputs/web_eval_report.json", encoding="utf-8"))
    real_agg = collections.defaultdict(lambda: [0, 0])
    for zh, v in real["per_class"].items():
        c = zh2crop.get(zh, "未知")
        real_agg[c][0] += v["ok"]
        real_agg[c][1] += v["n"]
    table("表 2  真实域·微调前基线（PlantDoc test 120 张，不指定作物）", real_agg,
          "总体 17.5%（21/120）；未做作物先验，故普遍偏低")

    # ---------- 表 3：真实域当前权重 ----------
    txt = ROOT / "backups/_realtest_eval_225.txt"
    if not txt.exists():
        print(f"\n[跳过] 表 3 需要 {txt.name}，未找到。生成方式见本文件末尾。")
        return 0
    cur = collections.defaultdict(lambda: [0, 0])
    for ln in txt.read_text(encoding="utf-8").splitlines():
        m = LINE.match(ln)
        if not m:
            continue
        c = crop_by_en(m.group(1))
        cur[c][0] += int(m.group(3))
        cur[c][1] += int(m.group(4))
    table("表 3  真实域·当前部署权重（去重 225 张，带作物先验）", cur,
          "总体 76.9%（173/225）")

    # ---------- 番茄为什么最差 ----------
    print("\n番茄逐类明细（表 3 口径，准确率升序）：")
    detail = []
    for ln in txt.read_text(encoding="utf-8").splitlines():
        m = LINE.match(ln)
        if m and m.group(1).startswith("Tomato"):
            ok, n = int(m.group(3)), int(m.group(4))
            detail.append((m.group(1).split("___")[-1], ok, n))
    for name, ok, n in sorted(detail, key=lambda x: x[1] / x[2]):
        print(f"  {name:<34}{ok:>3}/{n:<3}{ok / n * 100:>6.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# ── 表 3 数据如何生成 ─────────────────────────────────────────────
#   X:\Anconda3\python.exe -m scripts.eval_realtest ^
#       --index outputs/index/dataset_index_realtest.json > backups/_realtest_eval_225.txt
#   然后手工去掉前几行日志即可（本脚本用正则只匹配「名称 百分比 (对/总)」行）。
