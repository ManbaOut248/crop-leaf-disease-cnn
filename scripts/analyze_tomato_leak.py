# -*- coding: utf-8 -*-
"""
番茄 186 张评估集拆分：已泄漏 120 张 vs 从未泄漏 66 张。

为什么必须拆
---------------------------------------------------------------
当前部署权重 models/best_resnet18_38.pth 是在**旧**混合索引上训的，那份索引里
677 张番茄真实图全部参与训练；后来从这 677 张里留出的 120 张，对部署权重来说
其实是「训练集里的图」。所以它在 186 张上的 48.4% 里，有 120 张是记忆加成，
不是泛化能力。

只有另外 66 张（本来就在 realtest 独立评估集里、从未进过任何训练集）才是公平的
泛化指标。新训的 ctrl / w4 起点是纯 PV 基座 + 干净切分，对 186 张全部未见。

结论用法：若 w4 在「66 张未泄漏」上不输部署权重，说明换掉部署权重是净收益 ——
总分数看起来低，只是因为它不再享受那 120 张的记忆红利。

拆分判据用目录前缀，不要用 basename
---------------------------------------------------------------
评估集 186 张 = data/real_mix_test/ 下 66 张（原独立评估集，从未进训练）
             + data/real_mix/ 下 120 张（新留出，路径没搬，只是从训练索引剔除）
踩过的坑：一开始用 basename 匹配，120 张被算成 132 张 —— 因为两个目录里有 12 个
文件名撞车（如 Tomato___Early_blight__00000.jpg 两边都有）。

用法
    python backups/_tomato_split.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # 让 `import src.config` 生效（脚本在 backups/ 下）

from src.config import CLASS_DIRS  # noqa: E402


def is_leaked(path):
    """real_mix/ 下 = 留出图（部署权重训练时见过）；real_mix_test/ 下 = 从未泄漏。"""
    return "real_mix_test" not in str(path).replace("\\", "/")


GROUPS = [
    ("纯PV基座", ROOT / "backups" / "v2_base_pv.json"),
    ("部署权重(含泄漏)", ROOT / "backups" / "v2_deployed.json"),
    ("ctrl(w=1.0)", ROOT / "backups" / "v2_ctrl.json"),
    ("w4(w=4.0)", ROOT / "backups" / "v2_w4.json"),
]


def load_tomato_ok(path):
    """返回 {图路径: 是否预测正确}，只保留番茄类。"""
    rows = json.load(open(path, encoding="utf-8"))["rows"]
    return {r["path"]: r["ok"] for r in rows if CLASS_DIRS[r["label"]].startswith("Tomato")}


def mcnemar(flags_a, flags_b):
    """配对精确二项检验（双侧）。a=实验组，b=对照组。"""
    from scipy.stats import binomtest
    only_a = sum(1 for x, y in zip(flags_a, flags_b) if x and not y)
    only_b = sum(1 for x, y in zip(flags_a, flags_b) if not x and y)
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    return only_a, only_b, binomtest(min(only_a, only_b), n, 0.5).pvalue


ok = {name: load_tomato_ok(p) for name, p in GROUPS}
ref = ok["w4(w=4.0)"]
leak_paths = [p for p in ref if is_leaked(p)]
clean_paths = [p for p in ref if not is_leaked(p)]
assert len(leak_paths) == 120, f"留出图应为 120 张，实际 {len(leak_paths)}"
assert len(clean_paths) == 66, f"未泄漏应为 66 张，实际 {len(clean_paths)}"

print(f"评估集番茄 {len(ref)} 张 = 已泄漏 {len(leak_paths)} 张 + 未泄漏 {len(clean_paths)} 张")
print()
print(f"{'权重':<18}{'番茄合计':>13}{'已泄漏120':>14}{'未泄漏66':>13}")
print("-" * 60)
for name, _ in GROUPS:
    d = ok[name]
    hl = sum(d[p] for p in leak_paths)
    hu = sum(d[p] for p in clean_paths)
    print(f"{name:<18}{hl + hu:>5}/{len(ref):<8}{hl:>6}/{len(leak_paths):<7}{hu:>5}/{len(clean_paths):<7}")

print("\n未泄漏 66 张上的配对检验（这才是真实泛化能力）")
print("-" * 60)
for a, b in [("w4(w=4.0)", "部署权重(含泄漏)"), ("w4(w=4.0)", "ctrl(w=1.0)"),
             ("w4(w=4.0)", "纯PV基座"), ("ctrl(w=1.0)", "部署权重(含泄漏)")]:
    fa = [ok[a][p] for p in clean_paths]
    fb = [ok[b][p] for p in clean_paths]
    oa, ob, p_val = mcnemar(fa, fb)
    mark = "✅ 显著" if p_val < 0.05 else "⚠️ 不显著"
    print(f"  {a:<14} vs {b:<14} {sum(fa):>2}/{len(fa)} vs {sum(fb):>2}/{len(fb)}  "
          f"净增 {sum(fa) - sum(fb):+d} 张 | 独对 {oa}/{ob} → p={p_val:.4f} {mark}")
