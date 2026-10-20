"""置信度校准实验：真实域温度缩放（Temperature Scaling）。

问题
    当前温度 T=1.0 是在 PlantVillage（黑底居中单叶）上拟合的，ECE 仅 0.0028。
    但换到真实田间照片，实测「错判时平均置信度 0.718，其中 32% 高于 0.9」——
    用户看到「玉米灰斑病 96%」完全不会怀疑，而整体只有 17.5% 的样本是对的。
    这是**校准失效**（confidence 与 accuracy 脱节），与「识别错误」是两件独立的事。

做法
    · 拟合集：PlantDoc **train**（648 张，有标签）
    · 评估集：PlantDoc **test**（120 张，从未参与拟合）—— 两者 MD5 零重叠
    · 温度缩放：p = softmax(logits / T)，网格搜 T 使拟合集 NLL 最小
    · 两种模式各拟合一套 T：
        ① global  全 38 类（用户不指定作物）
        ② crop    作物约束（用户指定了正确作物）—— 掩码后 softmax 更尖锐，需要更大 T

指标
    ECE          期望校准误差（15 等宽分箱，按样本数加权）
    MCE          最大校准误差（最差那个箱子）
    高置信错判率  conf > 0.9 且判错的比例 —— 最伤信任的那一类
    准确率       用于确认「校准没有偷偷改排序」

⚠️ 温度缩放是**单调变换**，不改变样本间排序，**Top-1 准确率完全不变**。
   它只让数字变诚实，不解决「判错」。凡是宣称「校准提升准确率」的说法都是错的。

用法
    py -3.12 scripts/calibrate_domain.py
    py -3.12 scripts/calibrate_domain.py --max-train 100    # 快速冒烟
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
from PIL import Image, ImageFile

from src.config import CLASS_DIRS, DISEASE_KNOWLEDGE_PATH
from src.model.predict import Predictor

ImageFile.LOAD_TRUNCATED_IMAGES = True

TRAIN_DIR = ROOT / "outputs" / "realtest"
TEST_MANIFEST = ROOT / "outputs" / "realtest_test" / "manifest.jsonl"
OUT_JSON = ROOT / "outputs" / "calibration_domain.json"          # 实验明细（含分箱）
RUNTIME_JSON = ROOT / "models" / "calibration_domain.json"       # 运行时依赖（小，入库）

# 可靠性分档阈值（由实测定，理由见 docs/14）
#   crop 模式下：high 档判对率 89.7%、medium 42.0%、low 22.6% —— 分档确实有区分度
#   global 模式下校准后置信度集中在 0.06~0.21，93% 落 low（符合 17.5% 的真实水平）
THR_HIGH = 0.55
THR_MEDIUM = 0.25

T_GRID = np.round(np.arange(0.5, 12.001, 0.05), 3)
N_CLS = len(CLASS_DIRS)


# ========================= 数据 =========================
def load_train():
    """PlantDoc train：目录 -> class_id 的映射复用 fetch_plantdoc.JOBS（单一来源）。"""
    import fetch_plantdoc as fp

    out = []
    for _gh_dir, sub, cid, _files in fp.JOBS:
        d = TRAIN_DIR / sub
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.stat().st_size > 5000:
                out.append({"path": str(p), "class_id": cid})
    return out


def load_test():
    return [
        json.loads(x)
        for x in TEST_MANIFEST.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


# ========================= 概率 / 拟合 =========================
def probs(logits, T, cand=None):
    """cand 为 (N, 38) 的 bool 矩阵（逐样本候选集）；None 表示不约束。"""
    z = logits / float(T)
    if cand is not None:
        z = z.masked_fill(~cand, float("-inf"))
    return torch.softmax(z, dim=1)


def nll(logits, labels, T, cand=None):
    p = probs(logits, T, cand)
    idx = torch.arange(len(labels))
    return -torch.log(p[idx, labels].clamp_min(1e-12)).mean().item()


def fit_T(logits, labels, cand=None):
    best_nll, best_T = float("inf"), 1.0
    for T in T_GRID:
        v = nll(logits, labels, float(T), cand)
        if v < best_nll:
            best_nll, best_T = v, float(T)
    return best_T, best_nll


def metrics(logits, labels, T, cand=None):
    p = probs(logits, T, cand)
    conf, pred = p.max(dim=1)
    return conf.numpy(), (pred.numpy() == np.asarray(labels)).astype(int)


# ========================= 校准指标 =========================
def ece_mce(conf, correct, n_bins=15):
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce, bins = 0.0, 0.0, []
    n = len(conf)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not m.any():
            bins.append({"range": f"{lo:.2f}-{hi:.2f}", "n": 0})
            continue
        a, c = float(correct[m].mean()), float(conf[m].mean())
        gap = abs(a - c)
        ece += m.sum() / n * gap
        mce = max(mce, gap)
        bins.append({"range": f"{lo:.2f}-{hi:.2f}", "n": int(m.sum()),
                     "acc": round(a, 4), "conf": round(c, 4), "gap": round(gap, 4)})
    return float(ece), float(mce), bins


def summarize(conf, correct):
    ece, mce, bins = ece_mce(conf, correct)
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    hi = conf > 0.9
    # 三档可靠性各自的判对率 —— 用来验证分档是否真的能区分「可信」与「不可信」
    tiers = {}
    for name, lo, up in (("high", THR_HIGH, 9.9),
                         ("medium", THR_MEDIUM, THR_HIGH),
                         ("low", 0.0, THR_MEDIUM)):
        m = (conf >= lo) & (conf < up)
        tiers[name] = {"n": int(m.sum()),
                       "share": round(float(m.mean()), 4),
                       "acc": round(float(correct[m].mean()), 4) if m.any() else None}
    return {
        "n": int(len(conf)),
        "accuracy": round(float(correct.mean()), 4),
        "mean_confidence": round(float(conf.mean()), 4),
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "overconfidence": round(float(conf.mean() - correct.mean()), 4),
        "high_conf_n": int(hi.sum()),
        "high_conf_wrong": int((hi & (correct == 0)).sum()),
        "high_conf_wrong_rate": round(float((hi & (correct == 0)).sum() / max(1, hi.sum())), 4),
        "tiers": tiers,
        "bins": bins,
    }


# ========================= 主流程 =========================
@torch.no_grad()
def extract_logits(pred, items, batch=16):
    model, dev, tf = pred.model, pred.device, pred.transform
    buf, chunks = [], []

    def flush():
        if buf:
            chunks.append(model(torch.stack(buf).to(dev)).cpu())
            buf.clear()

    for it in items:
        try:
            im = Image.open(it["path"]).convert("RGB")
            buf.append(tf(im))
        except Exception:
            continue
        if len(buf) == batch:
            flush()
    flush()
    return torch.cat(chunks) if chunks else torch.empty(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=0)
    args = ap.parse_args()

    know = json.loads(Path(DISEASE_KNOWLEDGE_PATH).read_text(encoding="utf-8"))
    crop_of = lambda cid: (know.get(CLASS_DIRS[cid]) or {}).get("crop", "")
    crop_ids = {}
    for i, en in enumerate(CLASS_DIRS):
        c = crop_of(i)
        if c:
            crop_ids.setdefault(c, []).append(i)

    pred = Predictor.get_instance()
    train, test = load_train(), load_test()
    if args.max_train:
        train = train[: args.max_train]
    print(f"拟合集 {len(train)} 张（PlantDoc train） | 评估集 {len(test)} 张（PlantDoc test）")

    t0 = time.time()
    lg_tr = extract_logits(pred, train)
    lg_te = extract_logits(pred, test)
    # 坏图会让 logits 数量少于 items —— 对齐截断，避免标签错位
    train, test = train[: len(lg_tr)], test[: len(lg_te)]
    y_tr = torch.as_tensor([it["class_id"] for it in train], dtype=torch.long)
    y_te = torch.as_tensor([it["class_id"] for it in test], dtype=torch.long)
    print(f"特征提取 {time.time()-t0:.0f}s | logits {tuple(lg_te.shape)}")

    def cand_matrix(items):
        """逐样本候选集（作物约束）；返回 None 表示不约束。"""
        m = torch.zeros((len(items), N_CLS), dtype=torch.bool)
        for i, it in enumerate(items):
            ids = crop_ids.get(crop_of(it["class_id"]), [])
            if ids:
                m[i, ids] = True
        return m if bool(m.any()) else None

    results = {}
    for mode in ("global", "crop"):
        c_tr = cand_matrix(train) if mode == "crop" else None
        c_te = cand_matrix(test) if mode == "crop" else None
        T, nll_val = fit_T(lg_tr, y_tr, c_tr)
        before = summarize(*metrics(lg_te, y_te, 1.0, c_te))
        after = summarize(*metrics(lg_te, y_te, T, c_te))
        results[mode] = {"T": T, "nll_train": round(nll_val, 4),
                         "before": before, "after": after}
        print(f"  [{mode}] T={T:.2f}  ECE {before['ece']:.4f} → {after['ece']:.4f}"
              f"  过度自信 {before['overconfidence']:+.3f} → {after['overconfidence']:+.3f}")

    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 运行时依赖：只存推理真正需要的几个数，不含分箱明细
    RUNTIME_JSON.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_JSON.write_text(json.dumps({
        "T_global": results["global"]["T"],
        "T_crop": results["crop"]["T"],
        "reliability_high": THR_HIGH,
        "reliability_medium": THR_MEDIUM,
        "fitted_on": "PlantDoc train (648 张)",
        "evaluated_on": "PlantDoc test (120 张, 未参与拟合)",
        "ece_global_before": results["global"]["before"]["ece"],
        "ece_global_after": results["global"]["after"]["ece"],
        "ece_crop_before": results["crop"]["before"]["ece"],
        "ece_crop_after": results["crop"]["after"]["ece"],
        "note": (
            "温度缩放只改置信度数值，不改变 Top-1（单调变换）。"
            "T 在真实田间数据上拟合，故对 PlantVillage 这类实验室风格图会偏保守（欠自信）—— "
            "这是刻意选择：宁可说「仅供参考」，也不能把 17% 准确率的场景显示成 93% 可靠。"
        ),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("置信度校准结果（评估集 = PlantDoc test，从未参与拟合）")
    print("=" * 80)
    print(f"{'模式':<14}{'T':>6}{'准确率':>9}{'平均置信':>10}{'ECE':>9}{'MCE':>9}"
          f"{'高置信错判':>11}")
    for mode in ("global", "crop"):
        r = results[mode]
        for key, label in (("before", "1.00"), ("after", f"{r['T']:.2f}")):
            s = r[key]
            print(f"{mode + '/' + key:<14}{label:>6}{s['accuracy']:>9.1%}"
                  f"{s['mean_confidence']:>10.3f}{s['ece']:>9.4f}{s['mce']:>9.4f}"
                  f"{str(s['high_conf_wrong']) + '/' + str(s['high_conf_n']):>11}")
        print("-" * 80)
    print("\n可靠性分档验证（校准后置信度 → 该档判对率；分档若无区分度则形同虚设）")
    for mode in ("global", "crop"):
        t = results[mode]["after"]["tiers"]
        parts = [f"{k}: {v['acc']:.1%}(n={v['n']})" if v["n"] else f"{k}: —"
                 for k, v in t.items()]
        print(f"  {mode:<8}" + "  |  ".join(parts))
    print(f"\n明细：{OUT_JSON}\n运行时文件：{RUNTIME_JSON}")


if __name__ == "__main__":
    main()
