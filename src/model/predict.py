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
    OOD_HEAD_PATH, OOD_HEAD_ENABLE, OOD_HEAD_THRESHOLD,
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


class _LeafHead:
    """叶片 / 非叶片二分类检测头（方案 C2）。

    在**冻结的** 38 类模型 512 维池化特征上做逻辑回归，与病害识别完全解耦。

    为什么不再依赖 MSP：
        38 类 softmax 的 top-1 概率衡量的是「在 38 个病害里最像哪个」，对非叶片
        输入也必须挑一个，置信度虚高、且跨域不可用。实测真实田间叶片与非叶片的
        MSP 分布几乎重合（AUROC 0.553，误杀 10% 只拦下 16.8%）。
    本检测头直接学「是不是叶片」：真实叶 vs 非叶 AUROC 0.99（同分布留出）、
    0.77~0.99（负样本留一来源的严格验证），且不修改原模型任何权重。

    产物为纯数值 npz（mean / scale / coef / intercept / threshold），
    加载时 allow_pickle=False，不存在 pickle 反序列化风险。
    """

    def __init__(self, path, threshold_override=None):
        self.ready = False
        self.threshold = 0.5
        try:
            d = np.load(path, allow_pickle=False)
            self.mean = d["mean"].astype(np.float32)
            self.scale = d["scale"].astype(np.float32)
            self.coef = d["coef"].astype(np.float32)
            self.bias = float(d["intercept"][0])
            self.threshold = float(d["threshold"][0])
            # 运行期阈值覆盖（OOD_HEAD_THRESHOLD），只影响判定、不改模型
            if threshold_override is not None:
                self.threshold = float(threshold_override)
            self.ready = True
        except Exception as e:  # noqa: BLE001
            warnings.warn(
                f"叶片检测头未加载（{path}）：{e}。将回退到 MSP 判据 —— "
                "该判据在真实田间照片上误杀率很高，建议运行 "
                "scripts/train_ood_head.py 重新生成。",
                stacklevel=2,
            )

    def score(self, feat):
        """feat: torch.Tensor (1, 512) → 「是叶片」的概率（0~1）。"""
        x = feat.detach().cpu().numpy().reshape(-1).astype(np.float32)
        z = float(((x - self.mean) / self.scale) @ self.coef + self.bias)
        return 1.0 / (1.0 + float(np.exp(-z)))


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
        # 叶片检测头只在 fp32 后端启用：量化 / 剪枝模型的池化特征分布与训练时
        # 不一致，复用会失准，这两个后端一律回退到 MSP。
        if OOD_HEAD_ENABLE and INFER_BACKEND == "fp32":
            self.leaf_head = _LeafHead(OOD_HEAD_PATH, OOD_HEAD_THRESHOLD)
        else:
            self.leaf_head = None
            # 必须显式告警：MSP 判据在真实田间只有 AUROC 0.553、误杀率可达 50%+，
            # 静默降级会让使用者误以为防护仍然存在。
            if OOD_HEAD_ENABLE:
                warnings.warn(
                    f"INFER_BACKEND={INFER_BACKEND} 下不启用叶片检测头，已回退到 MSP 判据。"
                    "MSP 在真实田间照片上不可靠（误杀率可能超过 50%）；"
                    "需要该能力请改用 INFER_BACKEND=fp32。",
                    stacklevel=2,
                )
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
        """内部推理实现：返回 (Top-K 结果列表, MSP, 平均色度, 池化特征)。

        MSP 与平均色度是**回退判据**（检测头不可用时使用）；
        池化特征是**主判据**的输入 —— 交给 _LeafHead 判断「是不是叶片」。
        未启用检测头时池化特征为 None。
        """
        # 1) 校正 EXIF 方向：手机照片常带旋转元数据，PIL 默认不应用，否则会把正立叶片喂歪导致误判
        img = ImageOps.exif_transpose(pil_image).convert("RGB")
        # 2) 预缩放：超大图（如 6000x4000）先缩到最长边 <= PRESCALE_SIZE，降低内存占用；
        #    不改变模型输入尺寸（仍为 IMAGE_SIZE=224），对精度无实质影响
        img = _prescale(img, PRESCALE_SIZE)
        chroma = self._mean_chroma(img)
        x = self.transform(img).unsqueeze(0).to(self.device)
        # 3) 用 forward hook 顺带取出 avgpool 的 512 维池化特征，
        #    避免为了拿特征而多跑一次前向（成本翻倍）
        pooled = {}
        hook = None
        if getattr(self, "leaf_head", None) is not None and hasattr(self.model, "avgpool"):
            def _grab(_module, _inp, out):
                pooled["f"] = out.flatten(1)
            hook = self.model.avgpool.register_forward_hook(_grab)
        try:
            logits = self.model(x)              # 统一 fp32，不使用 AMP（与评估一致）
        finally:
            if hook is not None:
                hook.remove()
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
        return results, float(topk.values[0].item()), chroma, pooled.get("f")

    def predict_image(self, pil_image, top_k=3):
        """输入 PIL.Image，返回 Top-K 结果列表（对外接口保持不变）。

        top_k 为逐次调用参数（非单例缓存），k 自动夹紧到 [1, 类别数]，避免越界。
        """
        results, _, _, _ = self._infer(pil_image, top_k)
        return results

    def predict_image_with_ood(self, pil_image, top_k=3):
        """返回 (results, ood_info)，额外给出「这是不是叶片照片」的判断。

        为什么需要它：
            模型是 38 类闭集 softmax，没有「以上都不是」这个选项，对任何输入都会
            吐出一个类别。实测把一张动漫插画判成「番茄叶霉病 96.87%」，并显示
            「结果可信」—— 对农户是误导性输出。

        三级判据（优先级从高到低）：

            1. **叶片检测头**（主判据，_LeafHead）
               在冻结模型的 512 维池化特征上做二分类，直接学「是不是叶片」。
               实测真实叶 vs 非叶 AUROC 0.99（同分布留出）、0.77~0.99
               （负样本留一来源的严格验证）—— 对真实田间场景是唯一可用判据。
               且不修改原模型，PlantVillage 精度保持 99.34%。

            2. **MSP**（回退判据，阈值 0.98）
               38 类 top-1 概率。跨域不可靠：真实田间 AUROC 仅 0.553，
               误杀 10% 只能拦下 16.8%。仅在检测头缺失/后端不支持时启用。

            3. **平均色度**（仅用于文案，**不参与判定**）
               标记「灰阶图」（黑白漫画 / 线稿 / 扫描件），供前端选择提示语措辞，
               并对灰阶图额外给一条「结果可能受影响」的温和提醒。
               注意语义变更：它**不再**是拦截条件 —— 该判据隐含「叶片照片必为彩色」
               这一强假设，实测把 40 张真实叶片转灰度后会 100% 误杀
               （而灰度增强后的检测头能判对 93%）。

        判定语义：**只看主判据**（检测头或 MSP），色度不参与。
        「提示而非硬阻断」：误杀真实叶片的代价（重拍一张）远小于漏放
        （据此施药），但也不直接拒绝，交由用户判断。

        返回字段：
            leaf_prob        —— 「是叶片」的把握（检测头输出；回退时为 MSP）
            leaf_prob_source —— "head" / "msp"，标明判据来源（便于诊断）
            mean_chroma      —— 平均色度
            grayscale        —— 是否标记为灰阶图（仅影响文案与提醒，不影响判定）
            likely_not_leaf  —— 判定结果（由主判据决定）
            msp              —— 原始 softmax 置信度（诊断 / 日志用）
        """
        results, msp, chroma, feat = self._infer(pil_image, top_k)
        grayscale = chroma < OOD_CHROMA_THRESHOLD

        if self.leaf_head is not None and self.leaf_head.ready and feat is not None:
            leaf_prob = self.leaf_head.score(feat)
            likely_not_leaf = leaf_prob < self.leaf_head.threshold
            source = "head"
        else:
            leaf_prob = msp
            likely_not_leaf = msp < OOD_MSP_THRESHOLD
            source = "msp"

        return results, {
            "leaf_prob": round(float(leaf_prob), 4),
            "leaf_prob_source": source,
            "mean_chroma": round(chroma, 1),
            "grayscale": grayscale,
            "likely_not_leaf": bool(likely_not_leaf),
            "msp": round(float(msp), 4),
        }


def predict_image(pil_image, top_k=3):
    """便捷函数：懒加载全局 Predictor 并预测。top_k 逐次生效（不受单例缓存影响）。"""
    predictor = Predictor.get_instance()
    return predictor.predict_image(pil_image, top_k=top_k)


def predict_image_with_ood(pil_image, top_k=3):
    """便捷函数：返回 (results, ood_info)，供 API 层做「非叶片输入」提示。"""
    predictor = Predictor.get_instance()
    return predictor.predict_image_with_ood(pil_image, top_k=top_k)
