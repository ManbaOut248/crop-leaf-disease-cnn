# -*- coding: utf-8 -*-
"""
推理封装：加载最佳 ResNet18 38 分类模型，对单张 PIL 图像做预测。
对外提供 predict_image()：返回 Top-K 类别（含英文/中文名、置信度），
并合并 disease_knowledge.json 中的防治建议。供 FastAPI 调用。

改善（B 方案审查结论）：
- 推理统一 fp32，不再使用 AMP（与 evaluate.py 评估一致，避免 fp16/fp32 精度偏差）
- 应用 Temperature Scaling 校准（由 evaluate.py 拟合的 T）缩放 logits，
  使 softmax 置信度更可靠，缓解"恒为 1.0"的过自信问题
"""

import json
from pathlib import Path

import torch
from PIL import Image

from src.config import (
    MODEL_DIR, CLASS_DIRS, class_name_zh, DISEASE_KNOWLEDGE_PATH,
    CALIBRATION_PATH, INFER_BACKEND, PRUNED_MODEL_PATH, QUANTIZED_MODEL_PATH,
    BEST_MODEL_PATH, LAST_MODEL_PATH,
)
from src.data.transforms import get_val_transform
from src.model.resnet import build_resnet18
from src.model.quantize import load_quantized_model
from src.model.prune import build_pruned_model, PRUNE_RATIO


class Predictor:
    """单例式推理器，懒加载模型。"""

    _instance = None

    def __init__(self, top_k=3, device=None):
        self.top_k = top_k
        # 量化模型仅 CPU 生效（qnnpack 后端）；其余后端尊重 device 参数或自动选 CUDA
        if INFER_BACKEND == "quantized":
            self.device = torch.device("cpu")
        else:
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = get_val_transform()
        self.model = self._load_model()
        self.model.eval()
        self.knowledge = self._load_knowledge()
        self.temperature = self._load_calibration()  # 温度缩放校准参数

    @classmethod
    def get_instance(cls, **kwargs):
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    def _load_model(self):
        backend = INFER_BACKEND
        # ---- 后端1：INT8 量化（CPU 专用） ----
        if backend == "quantized":
            if not QUANTIZED_MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"未找到量化模型权重（{QUANTIZED_MODEL_PATH}）。请先运行：python -m src.model.quantize"
                )
            model = load_quantized_model()          # 重建 CPU INT8 模型并载入权重
            print(f"[Predictor] 已加载量化模型(INT8/CPU): {QUANTIZED_MODEL_PATH}")
            return model.cpu().eval()
        # ---- 后端2：结构化通道剪枝（fp32，通道数已减） ----
        if backend == "pruned":
            if not PRUNED_MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"未找到剪枝模型权重（{PRUNED_MODEL_PATH}）。请先运行：python -m src.model.prune"
                )
            # 必须用「剪枝后架构」的模型（build_pruned_model 按同一比例复现通道数），
            # 再载入微调后的剪枝权重；用 build_resnet18 会因通道数对不上而崩溃。
            # 比例优先取 checkpoint 内元数据，缺失时回退常量（兼容旧模型，见 #6）。
            ckpt = torch.load(PRUNED_MODEL_PATH, map_location=self.device)
            pr = ckpt.get("prune_ratio", PRUNE_RATIO)
            model = build_pruned_model(pr).to(self.device)
            model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
            print(f"[Predictor] 已加载剪枝模型: {PRUNED_MODEL_PATH} (prune_ratio={pr}, device={self.device})")
            return model
        # ---- 默认后端：fp32 防泄漏 best ----
        model = build_resnet18(pretrained=False).to(self.device)
        best_path = BEST_MODEL_PATH
        if not best_path.exists():
            # 退而求其次：用 last 断点（便于未训练完时也能演示）
            best_path = LAST_MODEL_PATH
        if not best_path.exists():
            raise FileNotFoundError(
                f"未找到模型权重（{best_path}）。请先运行训练：python -m src.model.train"
            )
        state = torch.load(best_path, map_location=self.device)
        sd = state["model"] if "model" in state else state
        model.load_state_dict(sd)
        print(f"[Predictor] 已加载模型: {best_path} (device={self.device})")
        return model

    def _load_knowledge(self):
        if DISEASE_KNOWLEDGE_PATH.exists():
            with open(DISEASE_KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _load_calibration(self):
        """加载 Temperature Scaling 校准温度 T（由 evaluate.py 在验证集拟合产出）。"""
        try:
            with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            T = float(cfg.get("temperature", 1.0))
            print(f"[Predictor] 加载置信度校准温度 T={T}")
            return T
        except Exception:
            print("[Predictor] 未找到校准参数文件，使用 T=1.0（不校准）")
            return 1.0

    @torch.no_grad()
    def predict_image(self, pil_image):
        """输入 PIL.Image，返回 Top-K 结果列表。统一 fp32 推理 + 温度缩放校准。"""
        img = pil_image.convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        logits = self.model(x)                  # 统一 fp32，不使用 AMP（与评估一致）
        logits = logits / self.temperature      # 温度缩放：T>1 抑制过自信，使置信度更可靠
        probs = torch.softmax(logits, dim=1)[0]
        topk = torch.topk(probs, self.top_k)
        results = []
        for rank, (idx, p) in enumerate(zip(topk.indices.tolist(), topk.values.tolist()), start=1):
            en = CLASS_DIRS[idx]
            know = self.knowledge.get(en, {})
            results.append({
                "rank": rank,
                "class_id": idx,
                "name_en": en,
                "name_zh": class_name_zh(en),
                "confidence": round(float(p), 4),
                "crop": know.get("crop", ""),
                "symptom": know.get("symptom", ""),
                "cause": know.get("cause", ""),
                "prevention": know.get("prevention", ""),
                "treatment": know.get("treatment", ""),
            })
        return results


def predict_image(pil_image, top_k=3):
    """便捷函数：懒加载全局 Predictor 并预测。"""
    predictor = Predictor.get_instance(top_k=top_k)
    return predictor.predict_image(pil_image)
