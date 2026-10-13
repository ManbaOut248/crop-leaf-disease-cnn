# -*- coding: utf-8 -*-
"""
配置中心：数据路径、38 类病害中英映射、训练/推理超参数。
所有模块统一 import 本文件，避免硬编码散落各处。
"""

import os
from pathlib import Path

# ===================== 路径配置 =====================
# 项目根目录（本文件位于 src/ 下，向上一级即为项目根）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 原始数据集根目录：默认直接引用 F:\实训产品\... 下的 PlantVillage，不拷贝（节省磁盘）。
# 可用环境变量 DATA_ROOT 覆盖（换机器 / 数据集迁移时无需改代码，见 #5）。
_DATA_ROOT_ENV = os.getenv("DATA_ROOT", "").strip()
DATA_ROOT = Path(_DATA_ROOT_ENV) if _DATA_ROOT_ENV else Path(r"F:\实训产品\基于CNN的农作物叶片病害识别系统\PlantVillage")

# 中间产物与产出目录
INDEX_DIR = PROJECT_ROOT / "outputs" / "index"      # 训练/验证/测试索引
MODEL_DIR = PROJECT_ROOT / "models"                  # 模型权重
REPORT_DIR = PROJECT_ROOT / "outputs"                # 评估报告、混淆矩阵图

# 模型权重文件名（集中管理，避免硬编码散落多处，见 #7）
BEST_MODEL_PATH = MODEL_DIR / "best_resnet18_38.pth"
LAST_MODEL_PATH = MODEL_DIR / "last_resnet18_38.pth"

# 推理用的病害知识库（中文防治建议）
DISEASE_KNOWLEDGE_PATH = PROJECT_ROOT / "src" / "data" / "disease_knowledge.json"

# ===================== 类别映射 =====================
# 英文文件夹名 -> 中文名称（与数据库 disease_knowledge 表一致）
# 顺序无关：代码运行时按文件夹名字典序排序生成 0..37 的模型输出索引，
# 但中文名必须按“英文文件夹名”精确对应。
CLASS_NAMES_ZH = {
    "Apple___Apple_scab": "苹果黑星病",
    "Apple___Black_rot": "苹果黑腐病",
    "Apple___Cedar_apple_rust": "苹果雪松锈病",
    "Apple___healthy": "苹果（健康）",
    "Blueberry___healthy": "蓝莓（健康）",
    "Cherry_(including_sour)___Powdery_mildew": "樱桃白粉病",
    "Cherry_(including_sour)___healthy": "樱桃（健康）",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "玉米灰斑病",
    "Corn_(maize)___Common_rust_": "玉米普通锈病",
    "Corn_(maize)___Northern_Leaf_Blight": "玉米北方叶枯病",
    "Corn_(maize)___healthy": "玉米（健康）",
    "Grape___Black_rot": "葡萄黑腐病",
    "Grape___Esca_(Black_Measles)": "葡萄埃斯卡病（黑麻疹）",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "葡萄叶枯病（房枯病）",
    "Grape___healthy": "葡萄（健康）",
    "Orange___Haunglongbing_(Citrus_greening)": "柑橘黄龙病",
    "Peach___Bacterial_spot": "桃细菌性斑点病",
    "Peach___healthy": "桃（健康）",
    "Pepper,_bell___Bacterial_spot": "甜椒细菌性斑点病",
    "Pepper,_bell___healthy": "甜椒（健康）",
    "Potato___Early_blight": "马铃薯早疫病",
    "Potato___Late_blight": "马铃薯晚疫病",
    "Potato___healthy": "马铃薯（健康）",
    "Raspberry___healthy": "树莓（健康）",
    "Soybean___healthy": "大豆（健康）",
    "Squash___Powdery_mildew": "南瓜白粉病",
    "Strawberry___Leaf_scorch": "草莓叶焦病",
    "Strawberry___healthy": "草莓（健康）",
    "Tomato___Bacterial_spot": "番茄细菌性斑点病",
    "Tomato___Early_blight": "番茄早疫病",
    "Tomato___Late_blight": "番茄晚疫病",
    "Tomato___Leaf_Mold": "番茄叶霉病",
    "Tomato___Septoria_leaf_spot": "番茄 Septoria 叶斑病",
    "Tomato___Spider_mites Two-spotted_spider_mite": "番茄红蜘蛛（二斑叶螨）",
    "Tomato___Target_Spot": "番茄靶斑病",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "番茄黄化曲叶病毒病",
    "Tomato___Tomato_mosaic_virus": "番茄花叶病毒病",
    "Tomato___healthy": "番茄（健康）",
}

# 运行时解析类别列表（保证与模型输出 0..37 一致）
def get_class_dirs():
    """返回类别文件夹名列表（即 0..37 索引顺序）。优先按 DATA_ROOT 真实目录；
    无数据集 / 类数异常时回退到内置中英映射键，避免导入即崩、整包不可导入（#5）。"""
    if DATA_ROOT.exists():
        dirs = sorted([p.name for p in DATA_ROOT.iterdir() if p.is_dir()])
        if len(dirs) == 38:
            return dirs
    # 回退：用内置 CLASS_NAMES_ZH 的键（同样 38 类，字典序一致）
    return sorted(CLASS_NAMES_ZH.keys())

CLASS_DIRS = get_class_dirs()          # 长度 38，索引 i 对应类别 dirs[i]
NUM_CLASSES = len(CLASS_DIRS)

def class_name_zh(en_name: str) -> str:
    """英文文件夹名 -> 中文名，缺省回退英文。"""
    return CLASS_NAMES_ZH.get(en_name, en_name)

# ===================== 数据增强 / 预处理 =====================
IMAGE_SIZE = 224
# ImageNet 预训练均值/标准差（与预训练权重一致，必需）
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 分层抽样比例（对齐任务书 8:1:1）
SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0.8, 0.1, 0.1
SPLIT_SEED = 42

# ===================== 防泄漏分组切分参数（C-1：group-aware split） =====================
GROUP_HASH_SIZE = 8     # 归组用平均哈希尺寸：8x8 = 64bit 指纹
GROUP_THRESH = 2        # 汉明距离 <=2 视为近重复，并查集归为同一“同叶组”
                        # （整组进 train/val/test 之一，杜绝跨切分泄漏；阈值2较严，避免误连不同叶）


# ===================== 训练超参数（适配 RTX 4060 8GB） =====================
BATCH_SIZE = 64            # 224x224 ResNet18 fp32 约 1.5GB 显存；AMP 下可更稳
NUM_EPOCHS = 30            # 配合早停，实际通常 15~25 轮收敛
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 6               # 验证集 F1 连续 PATIENCE 轮无提升则早停
NUM_WORKERS = 4
USE_AMP = True             # 混合精度，降低显存、提速
GRAD_CLIP = 1.0

# 类别不均衡应对：训练时按“逆频率”计算类别权重（在 train.py 中应用）
USE_CLASS_WEIGHT = True

# ===================== 推理 / 后端服务配置 =====================
MODEL_VERSION = "resnet18-plantvillage-38-v1"   # 经 API 暴露，便于灰度/回滚（API 文档 4）

# 置信度校准（Temperature Scaling）参数：由 evaluate.py 在验证集拟合温度 T 后产出，
# 推理端（predict.py）加载并用 T 缩放 logits，使 softmax 置信度更可靠（缓解过自信）。
CALIBRATION_PATH = PROJECT_ROOT / "outputs" / "calibration.json"
THRESHOLD_LOW_CONF = 0.5                         # 置信度低于此值 → 提示人工复核（API 文档 2.2）

# ===================== 模型压缩推理后端（D 方案：量化 / 剪枝） =====================
# 推理后端选择：fp32（默认，原始防泄漏 best） / quantized（INT8 量化，CPU 专用） / pruned（结构化通道剪枝）
# 由本开关统一切换 predict.py 的加载逻辑，predict_image() 对外接口不变。
INFER_BACKEND = "fp32"   # 可选值："fp32" | "quantized" | "pruned"
PRUNED_MODEL_PATH = MODEL_DIR / "pruned_resnet18_38.pth"        # D-1 结构化剪枝产出
QUANTIZED_MODEL_PATH = MODEL_DIR / "quantized_resnet18_38.pt"  # D-2 INT8 量化产出

# 上传文件存储目录（服务端受控，文件名随机化，禁止暴露用户目录）
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_UPLOAD_MB = 10                                # 单文件上限（API 文档 5）

# ===================== JWT / 安全 =====================
# 密钥优先读环境变量 JWT_SECRET（生产必须设置）；缺失时回退到开发用默认值并打印告警，
# 仅用于本地演示，绝不用于生产。用法：set JWT_SECRET=<一段长度>=32的随机串>
_JWT_SECRET_ENV = os.getenv("JWT_SECRET", "").strip()
if _JWT_SECRET_ENV:
    JWT_SECRET = _JWT_SECRET_ENV
else:
    import warnings
    JWT_SECRET = "dev-only-insecure-secret-DO-NOT-USE-IN-PROD"
    warnings.warn(
        "JWT_SECRET 未通过环境变量设置，已使用不安全的开发默认值。"
        "任何能读到该默认值的人都可伪造 token。生产环境务必执行："
        "set JWT_SECRET=<一段长度>=32的随机串>",
        stacklevel=2,
    )
JWT_EXPIRE_DAYS = int(os.getenv("JWT_EXPIRE_DAYS", "7"))
RATE_LIMIT_PER_MIN = 60                           # 单 IP 识别接口限流（API 文档 5）

# ===================== CORS（跨域白名单，替代原本 allow_origins=["*"]） =====================
# 允许的前端来源，逗号分隔；演示可临时设为 *，生产应收窄为具体前端域名。
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",") if o.strip()]
CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "false").strip().lower() == "true"
# 说明：本项目鉴权用 Bearer Token（非 Cookie），故默认 allow_credentials=False 即可，
#       即便设置具体来源也不会触发「* + credentials」的非法危险组合。

if __name__ == "__main__":
    print("项目根目录 :", PROJECT_ROOT)
    print("数据集目录 :", DATA_ROOT, "存在:", DATA_ROOT.exists())
    print("类别数     :", NUM_CLASSES)
    print("类别顺序(前5) :", CLASS_DIRS[:5])
