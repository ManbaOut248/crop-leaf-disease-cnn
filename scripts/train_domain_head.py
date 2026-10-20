"""域判别实验：能否可靠区分「实验室风格图」（PlantVillage）与「真实田间图」（PlantDoc）。

为什么需要
    实测同一个温度不可能同时适配两个域：
        PlantVillage：准确率 99.1%，T=1.0 → 置信 0.987（已很准）；T=6.05 → 0.325（严重欠自信）
        PlantDoc    ：准确率 17.5%，T=1.0 → 置信 0.727（严重过自信）；T=6.05 → 0.119（刚对上）
    因此必须按图选温度，前提是能判出「这张图像哪个域」。

做法
    复用叶片检测头已验证过的那套：冻结模型的 512 维 avgpool 特征 → 逻辑回归二分类。
    训练集：PV train 抽样 + PlantDoc train
    评估集：PV test 抽样 + PlantDoc test（与温度拟合集完全分开）

判据
    AUC ≥ 0.95 才值得落地；低于这个水平说明两域在特征空间重叠，按域校准不可靠。

用法
    py -3.12 scripts/train_domain_head.py
"""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from src.config import CLASS_DIRS
from src.model.predict import Predictor

ImageFile.LOAD_TRUNCATED_IMAGES = True

PV_ROOT = ROOT / "data" / "split" / "train"
PV_TEST_ROOT = ROOT / "data" / "split" / "test"
FIELD_ROOT = ROOT / "outputs" / "realtest"
FIELD_TEST = ROOT / "outputs" / "realtest_test" / "manifest.jsonl"
OUT = ROOT / "outputs" / "domain_head_eval.json"

PV_PER_CLASS = 25
SEED = 42


def list_pv(root, per_class=PV_PER_CLASS):
    rng = np.random.default_rng(SEED)
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        files = [p for p in d.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not files:
            continue
        if len(files) > per_class:
            idx = rng.choice(len(files), per_class, replace=False)
            files = [files[i] for i in idx]
        out += files
    return out


def list_field(root):
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir():
            out += [p for p in d.rglob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.stat().st_size > 5000]
    return out


@torch.no_grad()
def features(pred, paths, batch=24):
    """取 avgpool 后的 512 维特征。

    与 train_ood_head.py 同一写法：切掉最后两层（Dropout + fc），
    剩余部分输出 (N, 512, 1, 1)，flatten 后即 512 维。
    """
    model, dev, tf = pred.model, pred.device, pred.transform
    body = torch.nn.Sequential(*list(model.children())[:-1])
    buf, chunks = [], []

    def flush():
        if buf:
            chunks.append(body(torch.stack(buf).to(dev)).flatten(1).cpu())
            buf.clear()

    for p in paths:
        try:
            buf.append(tf(Image.open(p).convert("RGB")))
        except Exception:
            continue
        if len(buf) == batch:
            flush()
    flush()
    return torch.cat(chunks).numpy() if chunks else np.empty((0, 512))


def main():
    pred = Predictor.get_instance()
    pv_tr = list_pv(PV_ROOT)
    pv_te = list_pv(PV_TEST_ROOT, per_class=8)
    fd_tr = list_field(FIELD_ROOT)
    fd_te = list_field(ROOT / "outputs" / "realtest_test")
    print(f"PV  {len(pv_tr)} 训练 / {len(pv_te)} 评估")
    print(f"田间 {len(fd_tr)} 训练 / {len(fd_te)} 评估")

    t0 = time.time()
    X = {}
    for name, paths in (("pv_tr", pv_tr), ("pv_te", pv_te), ("fd_tr", fd_tr), ("fd_te", fd_te)):
        X[name] = features(pred, paths)
        print(f"  特征 {name:<6} {X[name].shape}  {time.time()-t0:.0f}s")

    Xtr = np.vstack([X["pv_tr"], X["fd_tr"]])
    ytr = np.r_[np.zeros(len(X["pv_tr"])), np.ones(len(X["fd_tr"]))]
    Xte = np.vstack([X["pv_te"], X["fd_te"]])
    yte = np.r_[np.zeros(len(X["pv_te"])), np.ones(len(X["fd_te"]))]

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(Xtr, ytr)

    sc_te = clf.predict_proba(Xte)[:, 1]
    auc_te = roc_auc_score(yte, sc_te)
    acc_te = ((sc_te > 0.5).astype(int) == yte).mean()
    cv = cross_val_score(clf, Xtr, ytr, cv=5, scoring="roc_auc")

    print(f"\n{'='*60}")
    print("域判别器（512 维特征 + 逻辑回归）")
    print("=" * 60)
    print(f"  训练集 AUC (5-fold CV)  {cv.mean():.4f} ± {cv.std():.4f}")
    print(f"  评估集 AUC              {auc_te:.4f}")
    print(f"  评估集准确率            {acc_te:.1%}")

    # 分域看错误方向 —— 判错哪一边代价更高？
    pv_wrong = (sc_te[:len(X['pv_te'])] > 0.5).sum()
    fd_wrong = (sc_te[len(X['pv_te']):] < 0.5).sum()
    print(f"  PV→误判为田间  {pv_wrong}/{len(X['pv_te'])}")
    print(f"  田间→误判为 PV {fd_wrong}/{len(X['fd_te'])}")

    OUT.write_text(json.dumps({
        "cv_auc_mean": round(float(cv.mean()), 4),
        "cv_auc_std": round(float(cv.std()), 4),
        "test_auc": round(float(auc_te), 4),
        "test_acc": round(float(acc_te), 4),
        "n_pv_train": len(X["pv_tr"]), "n_field_train": len(X["fd_tr"]),
        "n_pv_test": len(X["pv_te"]), "n_field_test": len(X["fd_te"]),
        "pv_to_field_errors": int(pv_wrong),
        "field_to_pv_errors": int(fd_wrong),
        "usable": bool(auc_te >= 0.95),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  结论：{'✅ 可用于按域校准' if auc_te >= 0.95 else '❌ AUC < 0.95，按域校准不可靠'}")
    print(f"  明细：{OUT}")


if __name__ == "__main__":
    main()
