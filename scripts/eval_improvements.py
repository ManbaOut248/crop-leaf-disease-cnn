# -*- coding: utf-8 -*-
"""量化两个可直接落地的改进手段对真实照片准确率的影响。

背景
---------------------------------------------------------------
PlantDoc test 上 Top-1 仅 17.5%，错误分析显示大量错判是**跨作物**的
（苹果锈病 → 玉米灰斑病）。这说明模型不是"完全不会看"，而是把
"叶片上的斑点纹理"当成了主要判据，忽略了作物形态。

三个假设，一次实验全部验证
---------------------------------------------------------------
H1 作物约束：若把候选限制在**真实作物**的类别子集内，准确率能提升多少？
   → 这是"两阶段：先定作物，再判病"方案的**上界**（oracle）。
H2 自动定作物：能否从 38 类概率**按作物求和**反推作物？
   → 决定两阶段能否不依赖用户手动选择。给出上界（H1）与此处实测的差距。
H3 TTA：水平翻转平均，能否稳定提点？
   → 零训练成本，纯推理期技巧。

用法
    py -3.12 scripts/eval_improvements.py
"""

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import torch
from PIL import Image, ImageFile

from src.config import CLASS_DIRS, DISEASE_KNOWLEDGE_PATH
from src.data.transforms import get_val_transform
from src.model.predict import Predictor

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "outputs" / "realtest_test" / "manifest.jsonl"
OUT = ROOT / "outputs" / "improvement_probe.json"

_KNOW = json.loads(Path(DISEASE_KNOWLEDGE_PATH).read_text(encoding="utf-8"))


def crop_of(cid):
    en = CLASS_DIRS[cid]
    return (_KNOW.get(en) or {}).get("crop") or en.split("___")[0]


# 作物 → 该类目下的类别索引（用于掩码）
CROP2IDS = defaultdict(list)
for i in range(len(CLASS_DIRS)):
    CROP2IDS[crop_of(i)].append(i)


def main():
    items = [json.loads(x) for x in MANIFEST.read_text(encoding="utf-8").splitlines() if x.strip()]
    print(f"样本 {len(items)} 张\n")

    pred = Predictor.get_instance()
    model, tf, device = pred.model, pred.transform, pred.device
    temp = getattr(pred, "temperature", 1.0)

    @torch.no_grad()
    def probs(img, flip=False):
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        x = tf(img).unsqueeze(0).to(device)
        logits = model(x) / temp
        return torch.softmax(logits, dim=1)[0].cpu()

    rows = []
    for it in items:
        p = ROOT / it["path"]
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        tid = it["class_id"]
        t_crop = crop_of(tid)

        p0 = probs(img)
        ph = probs(img, flip=True)
        p_tta = (p0 + ph) / 2          # 概率平均，比 logits 平均更稳（避免某一侧过自信）

        def top1(pr, mask=None):
            v = pr.clone()
            if mask is not None:
                keep = torch.zeros_like(v)
                keep[mask] = 1.0
                v = v * keep
            return int(torch.argmax(v))

        # 作物识别：把概率按作物求和，取最大的作物
        crop_sum = {c: float(sum(p0[i] for i in ids)) for c, ids in CROP2IDS.items()}
        pred_crop = max(crop_sum, key=crop_sum.get)

        rows.append({
            "tid": tid, "t_crop": t_crop,
            "base": top1(p0),
            "mask_oracle": top1(p0, CROP2IDS[t_crop]),      # 用真实作物做掩码（上界）
            "tta": top1(p_tta),
            "tta_mask_pred": top1(p_tta, CROP2IDS[pred_crop]),  # 用**预测**作物做掩码
            "crop_ok": pred_crop == t_crop,
            "pred_crop": pred_crop,
        })

    n = len(rows)
    acc = lambda key: sum(r[key] == r["tid"] for r in rows) / n
    print("=" * 74)
    print("真实网络照片（PlantDoc test）· 改进手段量化")
    print("=" * 74)
    print(f"  {'方案':<34}{'Top-1':>9}{'提升':>9}")
    print("-" * 74)
    base = acc("base")
    print(f"  {'① 基线（当前线上）':<32}{base:>9.1%}{'—':>9}")
    print(f"  {'② TTA（水平翻转平均）':<32}{acc('tta'):>9.1%}{acc('tta')-base:>+9.1%}")
    print(f"  {'③ 作物约束（真实作物，上界）':<32}{acc('mask_oracle'):>9.1%}"
          f"{acc('mask_oracle')-base:>+9.1%}")
    print(f"  {'④ 作物约束（预测作物）+ TTA':<32}{acc('tta_mask_pred'):>9.1%}"
          f"{acc('tta_mask_pred')-base:>+9.1%}")

    crop_acc = sum(r["crop_ok"] for r in rows) / n
    print("-" * 74)
    print(f"  作物识别（按概率求和取最大）准确率：{crop_acc:.1%}")
    print(f"    说明：若作物判错，约束会把正确答案排除 → ④ 的实际收益受此限制")
    print(f"    上界与实测的差距 = {acc('mask_oracle') - acc('tta_mask_pred'):.0%}，"
          f"这部分正是「作物判错」造成的损失")

    # 按作物分组的作物识别准确率（看哪些作物最难定）
    per_crop = defaultdict(lambda: [0, 0])
    for r in rows:
        per_crop[r["t_crop"]][1] += 1
        per_crop[r["t_crop"]][0] += int(r["crop_ok"])
    print("\n  各作物识别准确率（n≥5）")
    for c, (ok, tot) in sorted(per_crop.items(), key=lambda kv: -kv[1][1]):
        if tot >= 5:
            print(f"    {c:<8} {ok:>2}/{tot:<3} {ok/tot:>6.0%}")

    OUT.write_text(json.dumps({
        "n": n,
        "base": base,
        "tta": acc("tta"),
        "crop_constrained_oracle": acc("mask_oracle"),
        "crop_constrained_pred": acc("tta_mask_pred"),
        "crop_id_accuracy": crop_acc,
        "per_crop_id": {c: {"ok": v[0], "n": v[1]} for c, v in per_crop.items()},
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  明细已写入 {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
