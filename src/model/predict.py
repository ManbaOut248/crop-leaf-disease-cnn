# -*- coding: utf-8 -*-
"""
推理封装：加载最佳 ResNet18 38 分类模型，对单张 PIL 图像做预测。
对外提供 predict_image()：返回 Top-K 类别（含英文/中文名、置信度），
并合并 disease_knowledge.json 中的防治建议。供 FastAPI 调用。

改善（B 方案审查结论）：
- 推理统一 fp32，不再使用 AMP（与 evaluate.py 评估一致，避免 fp16/fp32 精度偏差）
- 应用 Temperature Scaling 校准（由 evaluate.py 拟合的 T）缩放 logits，
  使 softmax 置信度更可靠，缓解"恒为 1.0"的过自信问题

改善（低优先级审查结论）：
- top_k 改为「逐次调用参数」而非单例缓存：此前 Predictor 单例在首次创建时固定 top_k，
  之后 predict_image(img, top_k=5) 会被静默忽略、仍返回 3 条（参数名存在却失效，具误导性）。
  现 get_instance() 只缓存模型（加载代价高），top_k 每次调用都生效。
- 推理前加入「EXIF 方向校正 + 预缩放」：手机照片常带旋转元数据（PIL 默认不旋转，会把正立叶片喂歪）；
  超大图（如 6000x4000）先缩到最长边 <= PRESCALE_SIZE，降低内存占用，对 224x224 输入精度无实质影响。
"""

import json
from pathlib import Path

import torch
from PIL import Image, ImageOps

from src.config import (
    MODEL_DIR, CLASS_DIRS, class_name_zh, DISEASE_KNOWLEDGE_PATH,
    CALIBRATION_PATH, INFER_BACKEND, PRUNED_MODEL_PATH, QUANTIZED_MODEL_PATH,
    BEST_MODEL_PATH, LAST_MODEL_PATH, NUM_CLASSES, PRESCALE_SIZE,
)
from src.data.transforms import get_val_transform
from src.model.resnet import build_resnet18
from src.model.quantize import load_quantized_model
from src.model.prune import build_pruned_model, PRUNE_RATIO


def _prescale(img, max_side):
    """推理前预缩放：最长边超过 max_side 时按等比例缩放到 max_side（LANCZOS 高质量下采样）。
    仅用于降低超大上传图的内存占用，不改变模型输入尺寸（仍为 IMAGE_SIZE=224）。"""
    w, h = img.size
    long = max(w, h)
    if long <= max_side:
        return img
    scale = max_side / float(long)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


class Predictor:
    """单例式推理器，懒加载模型。"""

    _instance = None

    def __init__(self, device=None):
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
        self.num_classes = NUM_CLASSES

    @classmethod
    def get_instance(cls, device=None):
        # 注意：top_k 不在此缓存（见 predict_image），否则首次调用后 top_k 即被冻结、后续传参静默失效
        if cls._instance is None:
            cls._instance = cls(device=device)
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
    def predict_image(self, pil_image, top_k=3):
        """输入 PIL.Image，返回 Top-K 结果列表。统一 fp32 推理 + 温度缩放校准。
        top_k 为逐次调用参数（非单例缓存），k 自动夹紧到 [1, 类别数]，避免越界。"""
        # 1) 校正 EXIF 方向：手机照片常带旋转元数据，PIL 默认不应用，否则会把正立叶片喂歪导致误判
        img = ImageOps.exif_transpose(pil_image).convert("RGB")
        # 2) 预缩放：超大图（如 6000x4000）先缩到最长边 <= PRESCALE_SIZE，降低内存占用；
        #    不改变模型输入尺寸（仍为 IMAGE_SIZE=224），对精度无实质影响
        img = _prescale(img, PRESCALE_SIZE)
        x = self.transform(img).unsqueeze(0).to(self.device)
        logits = self.model(x)                  # 统一 fp32，不使用 AMP（与评估一致）
        logits = logits / self.temperature      # 温度缩放：T>1 抑制过自信，使置信度更可靠
        probs = torch.softmax(logits, dim=1)[0]
        k = max(1, min(int(top_k), self.num_classes))
        topk = torch.topk(probs, k)
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
    """便捷函数：懒加载全局 Predictor 并预测。top_k 逐次生效（不受单例缓存影响）。"""
    predictor = Predictor.get_instance()
    return predictor.predict_image(pil_image, top_k=top_k)
