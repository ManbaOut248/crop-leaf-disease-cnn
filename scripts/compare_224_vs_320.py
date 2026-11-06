# -*- coding: utf-8 -*-
"""
224 vs 320 双分辨率逐类/逐作物对照（数据来自同一次真实域评估的两次运行）

用途：回答「320 比 224 提升多少」，并定位提升/退化分别发生在哪些类和作物。
数据源（均由 scripts/eval_realtest.py 生成，同一评估集、同一作物先验）：
  backups/_realtest_eval_224_full.txt   -> 173/225 = 76.9%
  backups/_realtest_eval_320.txt        -> 180/225 = 80.0%
"""

import re
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

# 作物前缀 -> 中文名（PlantDoc 的目录命名里玉米带括号、甜椒带逗号）
CROP_MAP = {
    "Apple": "苹果",
    "Blueberry": "蓝莓",
    "Cherry": "樱桃",
    "Corn": "玉米",
    "Grape": "葡萄",
    "Orange": "柑橘",
    "Peach": "桃",
    "Pepper": "甜椒",
    "Potato": "马铃薯",
    "Raspberry": "树莓",
    "Soybean": "大豆",
    "Squash": "南瓜",
    "Strawberry": "草莓",
    "Tomato": "番茄",
}

# 逐类行格式： "  Tomato___Bacterial_spot    12.5%  (1/8)"
LINE_RE = re.compile(r"^\s{2}(\S.*?)\s+([\d.]+)%\s+\((\d+)/(\d+)\)\s*$")


def parse(path):
    """把评估输出解析成 {类名: (正确数, 总数)}"""
    result = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LINE_RE.match(line.rstrip("\n"))
            if not m:
                continue
            result[m.group(1)] = (int(m.group(3)), int(m.group(4)))
    return result


def to_crop(class_name):
    """从类名提取作物中文名"""
    prefix = class_name.split("___")[0]
    for key, zh in CROP_MAP.items():
        if prefix.startswith(key):
            return zh
    return prefix


def crop_agg(per_class):
    """按作物聚合：(正确, 总数)"""
    agg = defaultdict(lambda: [0, 0])
    for name, (ok, n) in per_class.items():
        crop = to_crop(name)
        agg[crop][0] += ok
        agg[crop][1] += n
    return {k: tuple(v) for k, v in agg.items()}


def main():
    a224 = parse("backups/_realtest_eval_224_full.txt")
    a320 = parse("backups/_realtest_eval_320.txt")

    # ---------- 表 1：逐作物 ----------
    c224, c320 = crop_agg(a224), crop_agg(a320)
    rows = []
    for crop in set(c224) | set(c320):
        o1, n1 = c224.get(crop, (0, 0))
        o2, n2 = c320.get(crop, (0, 0))
        if n1 == 0:
            continue
        r1 = o1 / n1 * 100
        r2 = o2 / n2 * 100
        rows.append((crop, o1, n1, r1, o2, n2, r2, r2 - r1))
    rows.sort(key=lambda r: (-(r[7]), r[3]))

    print("【表 1】逐作物准确率（真实域 225 张，带作物先验）")
    print("-" * 74)
    print(f"{'作物':<8}{'224':>12}{'320':>12}{'变化':>12}{'净增对':>10}")
    print("-" * 74)
    for crop, o1, n1, r1, o2, n2, r2, d in rows:
        delta = f"{d:+.1f}pp" if abs(d) > 1e-9 else "—"
        net = f"{o2 - o1:+d}" if o2 != o1 else "0"
        print(f"{crop:<8}{f'{o1}/{n1} {r1:.1f}%':>14}{f'{o2}/{n2} {r2:.1f}%':>14}{delta:>12}{net:>10}")
    t1o, t1n = sum(v[0] for v in c224.values()), sum(v[1] for v in c224.values())
    t2o, t2n = sum(v[0] for v in c320.values()), sum(v[1] for v in c320.values())
    print("-" * 74)
    print(f"{'合计':<8}{f'{t1o}/{t1n} {t1o/t1n*100:.1f}%':>14}"
          f"{f'{t2o}/{t2n} {t2o/t2n*100:.1f}%':>14}"
          f"{f'{(t2o/t2n-t1o/t1n)*100:+.1f}pp':>12}{f'{t2o-t1o:+d}':>10}")

    # ---------- 表 2：变化最大的类 ----------
    cls_rows = []
    for name in set(a224) & set(a320):
        o1, n1 = a224[name]
        o2, n2 = a320[name]
        if n1 == 0:
            continue
        d = (o2 - o1) / n1 * 100
        cls_rows.append((name, o1, n1, o2, n2, d))
    cls_rows.sort(key=lambda r: -r[5])

    print("\n【表 2】提升最大的 8 个类")
    print("-" * 74)
    print(f"{'类别':<44}{'224':>8}{'320':>8}{'变化':>12}")
    print("-" * 74)
    for name, o1, n1, o2, n2, d in cls_rows[:8]:
        print(f"{name:<44}{f'{o1}/{n1}':>9}{f'{o2}/{n2}':>9}{f'{d:+.1f}pp':>12}")

    worse = [r for r in cls_rows if r[5] < 0]
    print(f"\n【表 3】反而退化的类（共 {len(worse)} 个）")
    print("-" * 74)
    if worse:
        print(f"{'类别':<44}{'224':>8}{'320':>8}{'变化':>12}")
        print("-" * 74)
        for name, o1, n1, o2, n2, d in sorted(worse, key=lambda r: r[5]):
            print(f"{name:<44}{f'{o1}/{n1}':>9}{f'{o2}/{n2}':>9}{f'{d:+.1f}pp':>12}")
    else:
        print("无（所有类均持平或提升）")

    # ---------- 提升来源归因 ----------
    print("\n【归因】净增的 7 张来自哪里")
    print("-" * 74)
    gains = defaultdict(int)
    losses = defaultdict(int)
    for name, o1, n1, o2, n2, d in cls_rows:
        crop = to_crop(name)
        if o2 > o1:
            gains[crop] += o2 - o1
        elif o2 < o1:
            losses[crop] += o1 - o2
    for crop, g in sorted(gains.items(), key=lambda kv: -kv[1]):
        l = losses.get(crop, 0)
        print(f"  {crop:<6} 多对 {g:2d} 张，少对 {l:2d} 张，净 {g - l:+d}")
    for crop, l in losses.items():
        if crop not in gains:
            print(f"  {crop:<6} 多对  0 张，少对 {l:2d} 张，净 {-l:+d}")


if __name__ == "__main__":
    main()
