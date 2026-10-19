# -*- coding: utf-8 -*-
"""
训练「叶片 / 非叶片」二分类检测头（方案 C2）。

为什么是独立检测头，而不是改 38 类模型
---------------------------------------------------------------
38 类模型的 softmax 置信度（MSP）在真实田间场景下**无法**用于判断
「这是不是叶片」：实测真实叶与非叶片的 MSP 分布几乎重合
（AUROC 0.553，固定误杀 10% 只能拦下 16.8%）。

而直接微调 38 类模型来改善 OOD 分离会陷入跷跷板：
想改善真实域（AUROC 0.733），PlantVillage 精度就从 99.34% 掉到 86.89%；
想保住精度（95.36%），OOD 就几乎没改善（0.556）。

本方案把两件事解耦 —— 38 类模型**完全冻结**，另训一个 512 维特征上的
逻辑回归检测头。好处：
  · 原模型精度分毫不动（99.34%）
  · 二分类远比 38 类简单，少量负样本即可
  · 训练秒级（特征是提一次缓存后复用）

数据
    正样本（叶片）：PlantVillage 抽样 + 真实田间叶片（PlantDoc）
    负样本（非叶）：OOD 集（程序合成 / 真实照片 / 界面截图）

用法
    py -3.12 scripts/train_ood_head.py
    产物：models/ood_head.npz（纯数值，加载时无需 pickle，避免反序列化风险）
"""

import json
import random
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.config import BEST_MODEL_PATH, MODEL_DIR, CLASS_DIRS
from src.data.transforms import get_val_transform
from src.model.predict import _load_checkpoint
from src.model.resnet import build_resnet18

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ========================= 配置 =========================
SEED = 42
PV_PER_CLASS_TRAIN = 30      # PlantVillage 每类抽多少张作正样本
PV_PER_CLASS_TEST = 20       # 评估时每类抽多少张
REAL_TRAIN_RATIO = 0.8       # 真实田间叶片：8 成训练 / 2 成留出评估
OOD_TRAIN_RATIO = 0.8        # 非叶片样本同理
FEAT_BATCH = 32
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# PlantDoc 本地目录名 → 项目 class_id（仅用于采样，不影响检测头训练）
LEAF_DIRS = [
    "tomato_late_blight", "corn_rust", "grape_black_rot", "apple_scab",
    "apple_cedar_rust", "apple_healthy", "blueberry_healthy", "cherry_healthy",
    "corn_gray_leaf_spot", "corn_northern_leaf_blight", "grape_healthy",
    "peach_healthy", "pepper_bacterial_spot", "pepper_healthy",
    "potato_early_blight", "raspberry_healthy", "soybean_healthy",
    "squash_powdery_mildew", "strawberry_healthy", "tomato_early_blight",
    "tomato_leaf_mold", "tomato_septoria_leaf_spot",
    "tomato_yellow_leaf_curl_virus", "tomato_mosaic_virus", "tomato_healthy",
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TF = get_val_transform()


def list_imgs(d, verify=True):
    """列出图片。verify=True 会真正解码一次以剔除截断文件（PlantDoc 经代理下载偶有残图）。"""
    d = Path(d)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.rglob("*")):
        if p.suffix.lower() not in IMG_EXT or p.stat().st_size == 0:
            continue
        if verify:
            try:
                with Image.open(p) as im:
                    im.load()
            except Exception:
                continue
        out.append(p)
    return out


# ========================= 1) 收集数据 =========================
print("=" * 70)
print("收集数据")
print("=" * 70)

pv_train, pv_test = [], []
for cname in CLASS_DIRS:
    fs = list_imgs(f"data/split/train/{cname}", verify=False)
    random.shuffle(fs)
    pv_train += fs[:PV_PER_CLASS_TRAIN]
    fs2 = list_imgs(f"data/split/test/{cname}", verify=False)
    random.shuffle(fs2)
    pv_test += fs2[:PV_PER_CLASS_TEST]

real_all = []
for sub in LEAF_DIRS:
    real_all += list_imgs(f"outputs/realtest/{sub}")
random.shuffle(real_all)
n_tr = int(len(real_all) * REAL_TRAIN_RATIO)
real_train, real_test = real_all[:n_tr], real_all[n_tr:]

ood_all = []
for sub in ("synthetic", "localphoto", "realshot"):
    ood_all += list_imgs(f"data/ood_set/{sub}")
random.shuffle(ood_all)
n_otr = int(len(ood_all) * OOD_TRAIN_RATIO)
ood_train, ood_test = ood_all[:n_otr], ood_all[n_otr:]

print(f"正样本  PlantVillage {len(pv_train)} 张（训练） / {len(pv_test)} 张（评估）")
print(f"        真实田间   {len(real_train)} 张（训练） / {len(real_test)} 张（评估）")
print(f"负样本  非叶片     {len(ood_train)} 张（训练） / {len(ood_test)} 张（评估）")

if not real_all or not ood_all:
    raise SystemExit("数据不足：请先运行 scripts/fetch_plantdoc.py 与 scripts/build_ood_set.py")

# ========================= 2) 提取特征 =========================
print("\n" + "=" * 70)
print("提取 512 维特征（38 类模型冻结，取全局池化层输出）")
print("=" * 70)

model = build_resnet18(pretrained=False)
_st = _load_checkpoint(BEST_MODEL_PATH, DEVICE)
model.load_state_dict(_st["model"] if "model" in _st else _st)
model.to(DEVICE).eval()
body = nn.Sequential(*list(model.children())[:-1])   # 去掉 Dropout 与 fc


@torch.no_grad()
def extract(paths, tag):
    out = []
    t0 = time.time()
    for i in range(0, len(paths), FEAT_BATCH):
        xs = []
        for p in paths[i:i + FEAT_BATCH]:
            try:
                xs.append(TF(Image.open(p).convert("RGB")))
            except Exception:
                xs.append(torch.zeros(3, 224, 224))   # 坏图用零图占位，不影响其余
        out.append(body(torch.stack(xs).to(DEVICE)).flatten(1).cpu().numpy())
    print(f"  {tag:<20} {len(paths):>4} 张   {time.time()-t0:>5.1f}s")
    return np.concatenate(out)


F_pv_tr = extract(pv_train, "PV 训练")
F_real_tr = extract(real_train, "真实田间 训练")
F_pv_te = extract(pv_test, "PV 评估")
F_real_te = extract(real_test, "真实田间 评估")
F_ood_tr = extract(ood_train, "非叶片 训练")
F_ood_te = extract(ood_test, "非叶片 评估")


# ========================= 3) 训练 + 评估 =========================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit(pos_list, neg, C=1.0):
    X = np.vstack(pos_list + [neg])
    y = np.concatenate([np.ones(sum(len(f) for f in pos_list)), np.zeros(len(neg))])
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=C)
    clf.fit(sc.transform(X), y)
    return sc, clf


def score(sc, clf, F):
    return clf.predict_proba(sc.transform(F))[:, 1]


def report(sc, clf, tag):
    s_pv = score(sc, clf, F_pv_te)
    s_real = score(sc, clf, F_real_te)
    s_ood = score(sc, clf, F_ood_te)

    def auc(p, n):
        return float(roc_auc_score(np.concatenate([np.ones(len(p)), np.zeros(len(n))]),
                                   np.concatenate([p, n])))

    print(f"\n[{tag}]")
    print(f"  真实叶 vs 非叶   AUROC = {auc(s_real, s_ood):.4f}")
    print(f"  PV 叶  vs 非叶   AUROC = {auc(s_pv, s_ood):.4f}")
    print(f"  得分中位：PV 叶 {np.median(s_pv):.3f} / 真实叶 {np.median(s_real):.3f} "
          f"/ 非叶 {np.median(s_ood):.3f}")
    for f in (0.05, 0.10, 0.20):
        thr = float(np.quantile(s_real, f))
        print(f"  真实叶误杀 {f:.0%}（阈值 {thr:.4f}）→ 非叶拦截 "
              f"{float((s_ood < thr).mean()):.1%}")
    return {"real_auroc": auc(s_real, s_ood), "pv_auroc": auc(s_pv, s_ood)}


print("\n" + "=" * 70)
print("训练逻辑回归（正样本：PV + 真实田间）")
print("=" * 70)
sc_eval, clf_eval = fit([F_pv_tr, F_real_tr], F_ood_tr)
eval_metrics = report(sc_eval, clf_eval, "留出集评估（真实田间 20% / 非叶片 20%）")

print("\n" + "=" * 70)
print("阈值标定：用全部真实田间叶片得分，取 5% 分位作为保守阈值")
print("=" * 70)
all_real = np.vstack([F_real_tr, F_real_te])
s_all_real = score(sc_eval, clf_eval, all_real)
thr_p5 = float(np.quantile(s_all_real, 0.05))
thr_p10 = float(np.quantile(s_all_real, 0.10))
print(f"  真实叶得分 P5={thr_p5:.4f}  P10={thr_p10:.4f}  中位={np.median(s_all_real):.4f}")
print(f"  建议阈值：{min(thr_p5, 0.5):.2f}（取 P5 与 0.5 中较小者，宁可宽松不误杀）")

# ========================= 4) 全量重训并导出 =========================
print("\n" + "=" * 70)
print("用全量数据重训并导出")
print("=" * 70)
sc_full, clf_full = fit([F_pv_tr, F_real_tr, F_pv_te, F_real_te],
                        np.vstack([F_ood_tr, F_ood_te]))

threshold = float(min(np.quantile(score(sc_full, clf_full, all_real), 0.05), 0.5))
out_path = MODEL_DIR / "ood_head.npz"
np.savez(
    out_path,
    mean=sc_full.mean_.astype(np.float64),
    scale=sc_full.scale_.astype(np.float64),
    coef=clf_full.coef_[0].astype(np.float64),
    intercept=np.array([clf_full.intercept_[0]], dtype=np.float64),
    threshold=np.array([threshold], dtype=np.float64),
    # 注意：meta 用普通字符串数组（不是 dtype=object），
    # 这样 np.load 不需要 allow_pickle=True，避免反序列化风险
    meta=np.array([json.dumps({
        "feature": "resnet18.avgpool 512d",
        "base_model": BEST_MODEL_PATH.name,
        "n_pos": int(len(F_pv_tr) + len(F_real_tr) + len(F_pv_te) + len(F_real_te)),
        "n_neg": int(len(F_ood_tr) + len(F_ood_te)),
        "train_ratio": {"real": REAL_TRAIN_RATIO, "ood": OOD_TRAIN_RATIO},
        "eval": eval_metrics,
        "threshold": threshold,
        "seed": SEED,
    })]),
)
print(f"已导出 {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")
print(f"  真实叶得分 P5 阈值 = {threshold:.4f}")
print(f"\n评估摘要: 真实叶 AUROC={eval_metrics['real_auroc']:.4f}  "
      f"PV叶 AUROC={eval_metrics['pv_auroc']:.4f}")
print("\n注意：上述 AUROC 是**同分布留出**结果，会高估泛化能力。")
print("跨来源检验（负样本留一来源）实测 AUROC 在 0.77~0.99 之间，"
      "详见 docs/11_域适应与OOD检测实验.md。")
