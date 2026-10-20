# -*- coding: utf-8 -*-
"""
真实域（PlantDoc）带作物先验的 Top-1 准确率评估。

用途：量化「混合微调」对真实田间照片的效果。与现状基准（52.5% 带作物先验）对比。

- 评估集 data/real_mix_test（225 张，28 类，未进训练，防泄漏去重）
- 逐图预测时传入该图真实作物的「作物先验」（与部署时用户选作物的行为一致），
  即只在该作物的候选 38 类内归一化 softmax，再取 argmax
- 支持 --weights 指定任意权重（如 finetuned_resnet18_38.pth）与部署 best 对比

注意：准确率只取决于 logits 排序，与温度缩放无关；故本脚本不加载校准 T，
直接复用 Predictor 的 predict_image_with_ood（内部已含作物先验掩码）。

用法：
    python -m scripts.eval_realtest                      # 评估当前部署 best
    python -m scripts.eval_realtest --weights models/finetuned_resnet18_38.pth
"""
import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image
import torch
from src.config import CLASS_DIRS, DISEASE_KNOWLEDGE_PATH
from src.model.predict import Predictor

IMG_SUF = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None,
                    help="权重路径；默认用 Predictor 当前加载的（部署 best）")
    ap.add_argument("--split", default="data/real_mix_test",
                    help="评估集根目录（子目录名即 38 类全名）")
    args = ap.parse_args()

    pred = Predictor.get_instance()
    if args.weights:
        ck = torch.load(args.weights, map_location=pred.device)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        pred.model.load_state_dict(sd)
        print(f"已加载对比权重: {args.weights}")

    know = json.loads(Path(DISEASE_KNOWLEDGE_PATH).read_text(encoding="utf-8"))
    crop_of = lambda c: (know.get(CLASS_DIRS[c]) or {}).get("crop", "")

    root = Path(args.split)
    per = defaultdict(lambda: [0, 0])   # cid -> [correct, total]
    total_c = total_t = 0
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        try:
            cid = CLASS_DIRS.index(d.name)
        except ValueError:
            continue
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() not in IMG_SUF:
                continue
            try:
                im = Image.open(p).convert("RGB")
                res, _ = pred.predict_image_with_ood(im, top_k=1, crop=crop_of(cid))
                ok = res[0]["class_id"] == cid
            except Exception:
                ok = False
            per[cid][1] += 1
            per[cid][0] += ok
            total_c += ok
            total_t += 1

    print("=" * 70)
    print(f"真实域评估（带作物先验，PlantDoc）：{total_c}/{total_t} = "
          f"{total_c / max(1, total_t):.1%}")
    print("=" * 70)
    rows = sorted(per.items(), key=lambda kv: (kv[1][0] / kv[1][1] if kv[1][1] else 1))
    for cid, (c, t) in rows:
        if t:
            print(f"  {CLASS_DIRS[cid]:<46}{c / t:>7.1%}  ({c}/{t})")


if __name__ == "__main__":
    main()
