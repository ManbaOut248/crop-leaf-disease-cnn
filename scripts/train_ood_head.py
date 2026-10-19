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

灰度增强（2026-10-19 审计后加入）
---------------------------------------------------------------
起因：只用彩色图训练时，**灰阶真实叶片（黑白拍照 / 标本扫描）100% 被拒**
（检测头 1/40 判对，色度判据再补刀）。根因是训练正样本全是彩图，
检测头没见过灰度叶片的特征分布。

做法：把一半正样本灰度化后一起训练。实测：
    灰度叶通过率   5%  →  93%
    彩色 AUROC  0.9941 → 0.9918（几乎不变，误杀反而从 5.4% 降到 4.6%）
成本几乎为零，故默认开启（可用 --no-gray-aug 关闭）。

阈值约定
---------------------------------------------------------------
逻辑回归输出本身就是「是叶片」的概率，**0.5 是有明确含义的自然判别边界**，
故阈值固定为 0.5。此前用 `min(P5, 0.5)` 做分位数标定，实测在小样本下
P5 在不同划分间从 0.4543 漂到 0.8638，导致阈值行为不可预期 —— 已废弃。
P5 / P10 仅作为「该阈值对应的误杀率」参考写入 meta，不参与判定。

用法
    py -3.12 scripts/train_ood_head.py
    py -3.12 scripts/train_ood_head.py --no-gray-aug     # 关闭灰度增强（对照用）
    产物：models/ood_head.npz（纯数值，加载时无需 pickle，避免反序列化风险）
"""

import argparse
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
GRAY_AUG_RATIO = 0.5         # 正样本中灰度化的比例（0 = 关闭）
THRESHOLD = 0.5              # 决策阈值（固定，见文档字符串说明）
FEAT_BATCH = 32
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# PlantDoc 本地目录名（仅用于采样，不影响检测头训练）
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

_ap = argparse.ArgumentParser()
_ap.add_argument("--no-gray-aug", action="store_true", help="关闭灰度增强（对照实验用）")
_ap.add_argument("--tag", default="", help="产物后缀，便于多组实验并存")
ARGS, _ = _ap.parse_known_args()
if ARGS.no_gray_aug:
    GRAY_AUG_RATIO = 0.0
TAG = ARGS.tag

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
print("=" * 72)
print("收集数据")
print("=" * 72)

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
print(f"灰度增强: {GRAY_AUG_RATIO:.0%}（{'开启' if GRAY_AUG_RATIO > 0 else '关闭'}）")

if not real_all or not ood_all:
    raise SystemExit("数据不足：请先运行 scripts/fetch_plantdoc.py 与 scripts/build_ood_set.py")

# ========================= 2) 提取特征 =========================
print("\n" + "=" * 72)
print("提取 512 维特征（38 类模型冻结，取全局池化层输出）")
print("=" * 72)

model = build_resnet18(pretrained=False)
_st = _load_checkpoint(BEST_MODEL_PATH, DEVICE)
model.load_state_dict(_st["model"] if "model" in _st else _st)
model.to(DEVICE).eval()
body = nn.Sequential(*list(model.children())[:-1])   # 去掉 Dropout 与 fc


@torch.no_grad()
def extract(paths, tag, gray=False):
    """批量提特征。gray=True 时先把图转灰度再转回 RGB（保持 3 通道输入）。"""
    out = []
    t0 = time.time()
    for i in range(0, len(paths), FEAT_BATCH):
        xs = []
        for p in paths[i:i + FEAT_BATCH]:
            try:
                im = Image.open(p).convert("RGB")
                if gray:
                    im = im.convert("L").convert("RGB")
                xs.append(TF(im))
            except Exception:
                xs.append(torch.zeros(3, 224, 224))   # 坏图用零图占位，不影响其余
        out.append(body(torch.stack(xs).to(DEVICE)).flatten(1).cpu().numpy())
    print(f"  {tag:<22} {len(paths):>4} 张   {time.time()-t0:>5.1f}s")
    return np.concatenate(out)


F_pv_tr = extract(pv_train, "PV 训练")
F_real_tr = extract(real_train, "真实田间 训练")
F_pv_te = extract(pv_test, "PV 评估")
F_real_te = extract(real_test, "真实田间 评估")
F_ood_tr = extract(ood_train, "非叶片 训练")
F_ood_te = extract(ood_test, "非叶片 评估")

# 灰度增强特征（只对训练集，评估集单独提灰度版本用于验收）
F_pv_gray = F_real_gray = None
if GRAY_AUG_RATIO > 0:
    n_pv_g = int(len(pv_train) * GRAY_AUG_RATIO)
    n_real_g = int(len(real_train) * GRAY_AUG_RATIO)
    F_pv_gray = extract(pv_train[:n_pv_g], "PV 灰度增强", gray=True)
    F_real_gray = extract(real_train[:n_real_g], "真实叶 灰度增强", gray=True)
F_real_te_gray = extract(real_test, "真实叶评估 灰度", gray=True)


# ========================= 3) 训练 + 评估 =========================
def fit(pos_list, neg):
    X = np.vstack(pos_list + [neg])
    y = np.concatenate([np.ones(sum(len(f) for f in pos_list)), np.zeros(len(neg))])
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    clf.fit(sc.transform(X), y)
    return sc, clf


def score(sc, clf, F):
    return clf.predict_proba(sc.transform(F))[:, 1]


def report(sc, clf, tag):
    s_pv = score(sc, clf, F_pv_te)
    s_real = score(sc, clf, F_real_te)
    s_ood = score(sc, clf, F_ood_te)
    s_gray = score(sc, clf, F_real_te_gray)

    def auc(p, n):
        return float(roc_auc_score(np.concatenate([np.ones(len(p)), np.zeros(len(n))]),
                                   np.concatenate([p, n])))

    print(f"\n[{tag}]  阈值固定 {THRESHOLD}")
    print(f"  真实叶 vs 非叶   AUROC = {auc(s_real, s_ood):.4f}")
    print(f"  PV 叶  vs 非叶   AUROC = {auc(s_pv, s_ood):.4f}")
    print(f"  得分中位：PV 叶 {np.median(s_pv):.3f} / 真实叶 {np.median(s_real):.3f} "
          f"/ 非叶 {np.median(s_ood):.3f}")
    print(f"  阈值 {THRESHOLD} 处：真实叶误杀 {float((s_real < THRESHOLD).mean()):.1%}"
          f"  非叶片拦截 {float((s_ood < THRESHOLD).mean()):.1%}")
    print(f"  灰度化真实叶通过率（判为叶片）：{float((s_gray >= THRESHOLD).mean()):.0%}")
    return {
        "real_auroc": auc(s_real, s_ood),
        "pv_auroc": auc(s_pv, s_ood),
        "real_block_rate": float((s_ood < THRESHOLD).mean()),
        "real_false_reject": float((s_real < THRESHOLD).mean()),
        "gray_pass_rate": float((s_gray >= THRESHOLD).mean()),
    }


print("\n" + "=" * 72)
print("训练逻辑回归（正样本：PV + 真实田间" +
      ("，含灰度增强" if GRAY_AUG_RATIO > 0 else "") + "）")
print("=" * 72)
pos_eval = [F_pv_tr, F_real_tr] + ([F_pv_gray, F_real_gray] if GRAY_AUG_RATIO > 0 else [])
sc_eval, clf_eval = fit(pos_eval, F_ood_tr)
eval_metrics = report(sc_eval, clf_eval, "留出集评估（真实田间 20% / 非叶片 20%）")

# 参考信息：该阈值在留出集上对应的误杀率分位
print("\n" + "=" * 72)
print("阈值参考（仅记录，不参与判定）")
print("=" * 72)
s_all_real = score(sc_eval, clf_eval, np.vstack([F_real_tr, F_real_te]))
print(f"  真实叶得分：P5={np.quantile(s_all_real, 0.05):.4f}  "
      f"P10={np.quantile(s_all_real, 0.10):.4f}  中位={np.median(s_all_real):.4f}")

# ========================= 4) 全量重训并导出 =========================
print("\n" + "=" * 72)
print("用全量数据重训并导出")
print("=" * 72)
pos_full = [F_pv_tr, F_real_tr, F_pv_te, F_real_te]
if GRAY_AUG_RATIO > 0:
    pos_full += [F_pv_gray, F_real_gray,
                 extract(pv_test[:int(len(pv_test) * GRAY_AUG_RATIO)], "PV评估 灰度", gray=True),
                 extract(real_test[:int(len(real_test) * GRAY_AUG_RATIO)], "真实评估 灰度", gray=True)]
sc_full, clf_full = fit(pos_full, np.vstack([F_ood_tr, F_ood_te]))

out_path = MODEL_DIR / f"ood_head{TAG}.npz"
np.savez(
    out_path,
    mean=sc_full.mean_.astype(np.float64),
    scale=sc_full.scale_.astype(np.float64),
    coef=clf_full.coef_[0].astype(np.float64),
    intercept=np.array([clf_full.intercept_[0]], dtype=np.float64),
    threshold=np.array([THRESHOLD], dtype=np.float64),
    # 注意：meta 用普通字符串数组（不是 dtype=object），
    # 这样 np.load 不需要 allow_pickle=True，避免反序列化风险
    meta=np.array([json.dumps({
        "feature": "resnet18.avgpool 512d",
        "base_model": BEST_MODEL_PATH.name,
        "gray_aug_ratio": GRAY_AUG_RATIO,
        "threshold": THRESHOLD,
        "n_pos": int(sum(len(f) for f in pos_full)),
        "n_neg": int(len(F_ood_tr) + len(F_ood_te)),
        "train_ratio": {"real": REAL_TRAIN_RATIO, "ood": OOD_TRAIN_RATIO},
        "eval": eval_metrics,
        "seed": SEED,
    })]),
)
print(f"已导出 {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")
print(f"  灰度增强比例 = {GRAY_AUG_RATIO:.0%}   决策阈值 = {THRESHOLD}")
print(f"\n评估摘要: 真实叶 AUROC={eval_metrics['real_auroc']:.4f}  "
      f"误杀={eval_metrics['real_false_reject']:.1%}  "
      f"拦截={eval_metrics['real_block_rate']:.1%}  "
      f"灰度叶通过={eval_metrics['gray_pass_rate']:.0%}")
print("\n注意：上述 AUROC 是**同分布留出**结果，会高估泛化能力。")
print("跨来源检验（负样本留一来源）实测 AUROC 在 0.77~0.99 之间，"
      "详见 docs/11_域适应与OOD检测实验.md。")
