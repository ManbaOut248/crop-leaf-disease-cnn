# -*- coding: utf-8 -*-
"""
对比两份 eval_v2_ab.py 的逐图结果：准确率差异 + McNemar 配对检验 + 逐作物明细。

为什么必须做配对检验
---------------------------------------------------------------
同一批图上两个模型的差异是**配对**的，用两个独立比例的区间比较会低估功效。
McNemar 只看「不一致对」（A 对 B 错 / A 错 B 对），直接检验两者出错模式是否对称。
样本量小的时候（如番茄 186 张）优先用精确二项检验，卡方近似会偏乐观。

用法
    python -m scripts.compare_v2_ab backups/v2_w4.json backups/v2_ctrl.json
    # 参数顺序：实验组在前，对照组在后（净增为正 = 实验组更好）
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")


def mcnemar(a, b):
    """a=实验组 flags, b=对照组 flags（同序）。返回 (实验组独对, 对照组独对, p)。"""
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if not x and y)
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    try:
        from scipy.stats import binomtest
        p = binomtest(min(only_a, only_b), n, 0.5).pvalue     # 双侧精确检验
    except Exception:
        chi2 = (abs(only_a - only_b) - 1) ** 2 / n            # 连续性校正卡方
        p = math.erfc(math.sqrt(max(chi2, 0) / 2))
    return only_a, only_b, p


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def load(p):
    d = json.load(open(p, encoding="utf-8"))
    return d, {r["path"]: r for r in d["rows"]}


def main():
    if len(sys.argv) < 3:
        raise SystemExit("用法: python -m scripts.compare_v2_ab <实验组.json> <对照组.json>")
    exp_p, ctl_p = sys.argv[1], sys.argv[2]
    exp, e_rows = load(exp_p)
    ctl, c_rows = load(ctl_p)

    paths = [r["path"] for r in ctl["rows"]]
    assert set(paths) == set(e_rows), "两份额结果不是同一批图，无法配对"

    from src.config import CLASS_DIRS

    is_tom = lambda c: CLASS_DIRS[c].startswith("Tomato")
    crop_of = lambda c: CLASS_DIRS[c].split("___")[0].split(",")[0]

    def block(tag, sel):
        sel_paths = [p for p in paths if sel(ctl_paths_label[p])]
        if not sel_paths:
            return
        fa = [e_rows[p]["ok"] for p in sel_paths]
        fb = [c_rows[p]["ok"] for p in sel_paths]
        ka, kb, n = sum(fa), sum(fb), len(fa)
        oa, ob, p_val = mcnemar(fa, fb)
        la, ha = wilson(ka, n)
        lb, hb = wilson(kb, n)
        print(f"\n── {tag}（{n} 张）")
        print(f"   实验组: {ka}/{n} = {ka / n:.1%}  95%CI [{la:.1%}, {ha:.1%}]")
        print(f"   对照组: {kb}/{n} = {kb / n:.1%}  95%CI [{lb:.1%}, {hb:.1%}]")
        print(f"   差异  : {(ka - kb) / n * 100:+.1f}pp（净增 {ka - kb} 张）")
        print(f"   McNemar: 实验组独对 {oa} / 对照组独对 {ob} → 精确 p = {p_val:.4f} "
              f"{'✅ 显著' if p_val < 0.05 else '⚠️ 不显著'}")

    ctl_paths_label = {r["path"]: r["label"] for r in ctl["rows"]}
    print("=" * 76)
    print(f"实验组: {exp.get('weights')}")
    print(f"对照组: {ctl.get('weights')}")
    print("=" * 76)
    block("整体", lambda c: True)
    block("番茄", is_tom)
    block("非番茄", lambda c: not is_tom(c))

    # 逐类
    print("\n── 逐类变化（仅列有变化的类）")
    by_cls_exp, by_cls_ctl = {}, {}
    for r in exp["rows"]:
        g = by_cls_exp.setdefault(r["label"], [0, 0])
        g[0] += r["ok"]; g[1] += 1
    for r in ctl["rows"]:
        g = by_cls_ctl.setdefault(r["label"], [0, 0])
        g[0] += r["ok"]; g[1] += 1
    changed = []
    for c in by_cls_ctl:
        ka, n = by_cls_exp[c]
        kb, _ = by_cls_ctl[c]
        if ka != kb:
            changed.append((ka - kb, c, ka, kb, n))
    if not changed:
        print("   （无变化）")
    for d, c, ka, kb, n in sorted(changed, reverse=True):
        print(f"   {CLASS_DIRS[c]:<54}{kb:>3} → {ka:>3}  ({d:+d})  n={n}")

    # 逐作物
    print("\n── 逐作物准确率")
    def crop_table(rows):
        agg = {}
        for r in rows:
            g = agg.setdefault(crop_of(r["label"]), [0, 0])
            g[0] += r["ok"]; g[1] += 1
        return agg
    ea, ca = crop_table(exp["rows"]), crop_table(ctl["rows"])
    print(f"   {'作物':<12}{'实验组':>12}{'对照组':>12}{'差异':>9}")
    for cr in sorted(ca, key=lambda x: ca[x][0] / ca[x][1]):
        ka, n = ea[cr]
        kb, _ = ca[cr]
        print(f"   {cr:<12}{f'{ka}/{n}':>12}{f'{kb}/{n}':>12}{(ka - kb) / n * 100:>+8.1f}pp")


if __name__ == "__main__":
    main()
