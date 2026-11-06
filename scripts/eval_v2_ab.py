# -*- coding: utf-8 -*-
"""
在「番茄扩展评估集 v2」（345 张 = 原 225 + 新增留出 120）上评估指定权重，
并落盘逐图结果，供 compare_v2_ab.py 做 McNemar 配对检验。

为什么单独一个脚本
---------------------------------------------------------------
eval_realtest.py 只打印逐类百分比，没有逐图对错，无法做配对检验。
A/B 结论（真实域加权到底有没有用）必须以「同一批图上的配对差异」为依据。

评估口径与 eval_realtest.py 完全一致：
- 带作物先验（只在该图真实作物的候选类内归一化 softmax），模拟用户选作物后的行为；
- 准确率只取决于 logits 排序，与温度校准无关，故不加载校准温度。

用法
    python -m scripts.eval_v2_ab --weights models/finetuned_v2_ctrl.pth --out backups/v2_ctrl.json
    python -m scripts.eval_v2_ab --weights models/finetuned_v2_w4.pth   --out backups/v2_w4.json
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

import torch
from PIL import Image

from src.config import CLASS_DIRS, DISEASE_KNOWLEDGE_PATH
from src.model.predict import Predictor

DEFAULT_INDEX = ROOT / "outputs" / "index" / "dataset_index_realtest_v2.json"


def crop_map():
    """class_id -> 作物中文名（与部署时用户选择的作物一致）。"""
    know = json.loads(Path(DISEASE_KNOWLEDGE_PATH).read_text(encoding="utf-8"))
    return [(know.get(n) or {}).get("crop", "") for n in CLASS_DIRS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="权重路径；不给则用 Predictor 默认加载的")
    ap.add_argument("--index", default=str(DEFAULT_INDEX))
    ap.add_argument("--out", required=True, help="逐图结果 JSON 输出路径")
    args = ap.parse_args()

    pred = Predictor.get_instance()
    if args.weights:
        ck = torch.load(args.weights, map_location=pred.device, weights_only=False)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        pred.model.load_state_dict(sd)
        print(f"权重: {args.weights}")
        if isinstance(ck, dict) and "epoch" in ck:
            print(f"  训练信息: epoch={ck.get('epoch')} PV_val_f1={ck.get('val_f1')} "
                  f"REAL_test={ck.get('real_test_acc')}")

    crops = crop_map()
    items = json.loads(Path(args.index).read_text(encoding="utf-8"))["test"]
    print(f"评估集: {args.index}（{len(items)} 张）")

    rows = []
    for p, cid in items:
        cid = int(cid)
        try:
            im = Image.open(p).convert("RGB")
            res, _ = pred.predict_image_with_ood(im, top_k=1, crop=crops[cid])
            pd = int(res[0]["class_id"])
        except Exception as e:
            pd = -1
            print(f"  [跳过] {Path(p).name}: {type(e).__name__}")
        rows.append({"path": str(p), "label": cid, "pred": pd, "ok": int(pd == cid)})

    def is_tomato(c):
        return CLASS_DIRS[c].startswith("Tomato")

    def crop_of(c):
        return CLASS_DIRS[c].split("___")[0].split(",")[0]

    # ===================== 汇总 =====================
    tom = [r for r in rows if is_tomato(r["label"])]
    non = [r for r in rows if not is_tomato(r["label"])]
    tot_ok = sum(r["ok"] for r in rows)

    print("=" * 72)
    print(f"整体   : {tot_ok}/{len(rows)} = {tot_ok / len(rows):.1%}")
    if tom:
        k = sum(r["ok"] for r in tom)
        print(f"番茄   : {k}/{len(tom)} = {k / len(tom):.1%}")
    if non:
        k = sum(r["ok"] for r in non)
        print(f"非番茄 : {k}/{len(non)} = {k / len(non):.1%}")
    print("=" * 72)

    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        g = agg[crop_of(r["label"])]
        g[0] += r["ok"]
        g[1] += 1
    print(f"{'作物':<14}{'正确/总数':>12}{'准确率':>9}")
    for c, (k, n) in sorted(agg.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        print(f"{c:<14}{f'{k}/{n}':>12}{k / n:>8.1%}")

    out = {
        "weights": args.weights,
        "index": args.index,
        "n": len(rows),
        "n_tomato": len(tom),
        "ok": tot_ok,
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"\n逐图结果已写入: {args.out}")


if __name__ == "__main__":
    main()
