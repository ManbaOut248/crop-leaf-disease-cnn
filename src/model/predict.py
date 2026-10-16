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
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from src.config import (
    MODEL_DIR, CLASS_DIRS, class_name_zh, DISEASE_KNOWLEDGE_PATH,
    CALIBRATION_PATH, INFER_BACKEND, PRUNED_MODEL_PATH, QUANTIZED_MODEL_PATH,
    BEST_MODEL_PATH, LAST_MODEL_PATH, NUM_CLASSES, PRESCALE_SIZE,
    OOD_MSP_THRESHOLD, OOD_CHROMA_THRESHOLD,
)
from src.data.transforms import get_val_transform
from src.model.resnet import build_resnet18
from src.model.quantize import load_quantized_model
from src.model.prune import build_pruned_model, PRUNE_RATIO


def _allow_safe_numpy_globals():
    """放行 numpy 的纯数值类型，使 weights_only=True 能加载本项目 checkpoint。

    训练脚本在 checkpoint 里写入了 numpy 标量（如 val_f1 为 np.float64），
    而 weights_only 反序列化默认只允许张量 / 基础类型。这些类型仅承载数值、
    不含可执行代码，显式放行不会引入 pickle RCE 风险。
    """
    try:
        import numpy as np
        allowed = [np.core.multiarray.scalar, np.ndarray, np.dtype]
        for name in ("Float64DType", "Float32DType", "Int64DType", "Int32DType"):
            obj = getattr(np.dtypes, name, None)
            if isinstance(obj, type):
                allowed.append(obj)
        torch.serialization.add_safe_globals(allowed)
    except Exception:  # numpy 版本差异 / 旧版 torch 无该 API —— 交给下层兜底
        pass


def _load_checkpoint(path, map_location):
    """安全加载模型 checkpoint。

    安全加固（审计 P2）：torch 2.5 的 torch.load 默认 weights_only=False，
    会对权重文件执行完整 pickle 反序列化 —— 一旦权重文件被替换即可导致任意代码
    执行。这里优先以 weights_only=True 加载（只允许张量与白名单类型）。

    兼容性：本项目 checkpoint 含 numpy 数值标量，需先放行相应纯数值类型；
    若仍失败（例如来自其他训练器的权重），退回传统加载并打印告警，
    避免因安全加固反而令模型无法加载。
    """
    _allow_safe_numpy_globals()
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"以 weights_only=True 安全模式加载 {path} 失败（{exc}），"
            "已退回传统 pickle 模式，请确认该权重文件来源可信。",
            stacklevel=2,
        )
        return torch.load(path, map_location=map_location, weights_only=False)


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
            # 安全加载（weights_only=True，见 _load_checkpoint）
            ckpt = _load_checkpoint(PRUNED_MODEL_PATH, self.device)
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
        # 安全加载（weights_only=True，见 _load_checkpoint）；torch 2.5 默认 False
        state = _load_checkpoint(best_path, self.device)
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

    @staticmethod
    def _mean_chroma(img, size=160):
        """平均色度 = mean(max(R,G,B) - min(R,G,B))，取值约 0~255。

        灰阶图（黑白漫画 / 线稿 / 扫描件 / 纯色块）接近 0；
        彩色叶片照片显著大于 0 —— 实测真实叶片最低 10.6，而非叶片大量为 0。
        先等比缩到最长边 size 再统计：色度对分辨率不敏感，缩图可显著提速。
        """
        small = img.copy()
        small.thumbnail((size, size), Image.BILINEAR)
        a = np.asarray(small).astype(np.float32)
        return float((a.max(axis=2) - a.min(axis=2)).mean())

    @torch.no_grad()
    def _infer(self, pil_image, top_k=3):
        """内部推理实现：返回 (Top-K 结果列表, 最大 softmax 概率 MSP, 平均色度)。

        MSP（top-1 概率）与平均色度共同构成「非叶片输入」拒识判据，
        见 predict_image_with_ood()。
        """
        # 1) 校正 EXIF 方向：手机照片常带旋转元数据，PIL 默认不应用，否则会把正立叶片喂歪导致误判
        img = ImageOps.exif_transpose(pil_image).convert("RGB")
        # 2) 预缩放：超大图（如 6000x4000）先缩到最长边 <= PRESCALE_SIZE，降低内存占用；
        #    不改变模型输入尺寸（仍为 IMAGE_SIZE=224），对精度无实质影响
        img = _prescale(img, PRESCALE_SIZE)
        chroma = self._mean_chroma(img)
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
        # topk.values[0] 即全局最大概率（k>=1），作为 MSP 返回
        return results, float(topk.values[0].item()), chroma

    def predict_image(self, pil_image, top_k=3):
        """输入 PIL.Image，返回 Top-K 结果列表（对外接口保持不变）。

        top_k 为逐次调用参数（非单例缓存），k 自动夹紧到 [1, 类别数]，避免越界。
        """
        results, _, _ = self._infer(pil_image, top_k)
        return results

    def predict_image_with_ood(self, pil_image, top_k=3):
        """返回 (results, ood_info)，额外给出「这是不是叶片照片」的判断。

        为什么需要它：
            模型是 38 类闭集 softmax，没有"以上都不是"这个选项，对任何输入都会
            吐出一个类别。实测把一张动漫插画判成「番茄叶霉病 96.87%」，并显示
            "结果可信" —— 对农户是误导性输出。

        判据 = MSP **或** 平均色度（两者互补，任一命中即提示）：

            · MSP（最大 softmax 概率）—— 抓「模型自己也没把握」的输入，阈值 0.98。
              实测 400 张真实叶片 vs 363 张非叶片：误杀 6.2%、拦截 82.9%。

            · 平均色度 —— 抓「灰阶图」这类 MSP 会失效的输入。黑白漫画 / 线稿 /
              扫描件的色度接近 0，却可能被模型以高置信度误判（实测某漫画
              MSP=0.984 > 0.98 直接漏过）；而真实叶片照片实测色度最低 10.6，
              故阈值取 8.0，几乎零误伤。

            组合后实测：误杀仍为 6.2%，拦截率由 82.9% 提升到 93.1%，
            其中色度补拦了 MSP 漏掉样本的 59.7%。详见 docs/08_OOD拒识评估.md。

        返回字段：
            leaf_prob       —— MSP，前端用于展示「模型认为这是叶片的把握」
            mean_chroma     —— 平均色度，前端用于区分提示语（灰阶图 / 低置信）
            grayscale       —— 是否判定为灰阶图
            likely_not_leaf —— 组合判定结果
        """
        results, msp, chroma = self._infer(pil_image, top_k)
        grayscale = chroma < OOD_CHROMA_THRESHOLD
        return results, {
            "leaf_prob": round(msp, 4),
            "mean_chroma": round(chroma, 1),
            "grayscale": grayscale,
            "likely_not_leaf": (msp < OOD_MSP_THRESHOLD) or grayscale,
        }


def predict_image(pil_image, top_k=3):
    """便捷函数：懒加载全局 Predictor 并预测。top_k 逐次生效（不受单例缓存影响）。"""
    predictor = Predictor.get_instance()
    return predictor.predict_image(pil_image, top_k=top_k)


def predict_image_with_ood(pil_image, top_k=3):
    """便捷函数：返回 (results, ood_info)，供 API 层做「非叶片输入」提示。"""
    predictor = Predictor.get_instance()
    return predictor.predict_image_with_ood(pil_image, top_k=top_k)
