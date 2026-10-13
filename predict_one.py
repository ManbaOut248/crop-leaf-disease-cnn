# -*- coding: utf-8 -*-
"""
单图识别快捷入口（命令行工具）：
    X:/Anconda3/python.exe predict_one.py <图片路径>

复用已训练的 ResNet18 38 分类模型（best_resnet18_38.pth）+ 温度缩放校准，
对单张图片输出 Top-3 预测（中文名/英文名/置信度）与 Top-1 的防治建议。

注意：与后端 /predict 走同一套推理代码（src.model.predict），结果一致。
"""
import sys
from pathlib import Path

from PIL import Image

# 保证在项目根目录运行时能 import src.*
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.model.predict import predict_image  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print("用法: X:/Anconda3/python.exe predict_one.py <图片路径>")
        return
    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"[错误] 图片不存在: {img_path}")
        return

    img = Image.open(img_path)
    results = predict_image(img, top_k=3)

    print("=" * 64)
    print(f"图片: {img_path.name}")
    print("=" * 64)
    # Top-3 列表
    for r in results:
        tag = "★ Top-1" if r["rank"] == 1 else f"  Top-{r['rank']}"
        print(f"{tag}  {r['name_zh']}  (置信度 {r['confidence'] * 100:.2f}%)")
        print(f"         英文名: {r['name_en']}")
    # Top-1 的知识详情
    top = results[0]
    print("-" * 64)
    print(f"作物   : {top['crop']}")
    print(f"症状   : {top['symptom']}")
    print(f"病因   : {top['cause']}")
    print(f"防治   : {top['prevention']}")
    print(f"治疗   : {top['treatment']}")
    print("=" * 64)


if __name__ == "__main__":
    main()
