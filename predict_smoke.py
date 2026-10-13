# -*- coding: utf-8 -*-
"""D-3 端到端冒烟测试：验证三种后端经 predict.py 均可加载并单图预测。"""
from PIL import Image
import numpy as np
import src.model.predict as P

dummy = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8"), "RGB")

for backend in ["fp32", "quantized", "pruned"]:
    P.INFER_BACKEND = backend          # 覆盖 predict 模块内的后端开关
    P.Predictor._instance = None       # 重置单例，强制按新后端重新加载
    res = P.predict_image(dummy, top_k=1)
    top = res[0]
    print(f"[{backend:<10}] top1={top['name_zh']}({top['name_en']}) conf={top['confidence']}")
print("D-3 冒烟测试通过：三种后端均经 predict_image 正常出结果")
