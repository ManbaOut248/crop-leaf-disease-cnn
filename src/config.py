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
# 推理预缩放上限：上传图最长边超过此值先缩放到此值（保持比例），用于降低超大图内存占用。
# 仅作用于推理入口（predict.py），不改变模型输入尺寸（仍由 get_val_transform 统一缩到 224）；
# 设为 None 可关闭预缩放（直接走 transform）。
PRESCALE_SIZE = 512

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

# ===================== OOD（非叶片输入）拒识 =====================
# 背景：模型是 38 类闭集 softmax，没有"以上都不是"选项，对任何输入都会吐出
# 一个类别。实测把动漫插画判成「番茄叶霉病 96.87%」—— 对农户是误导性输出。
#
# 判据：最大 softmax 概率（MSP）。在 500 张真实叶片 + 363 张非叶片样本上实测：
#   MSP 的 AUROC = 0.9425（优于 Energy 0.9273 / MaxLogit 0.9281）；
#   阈值 0.98 时真实叶片误杀率≈5%、非叶片拦截率≈83%。
# 详见 scripts/eval_ood.py 与 outputs/ood_eval.json。
#
# 注意：这里只做「标记 + 前端强提示」，不硬阻断识别 —— 误杀真实叶片时
# 用户仍可查看结果，避免把可用功能一刀切掉。
OOD_MSP_THRESHOLD = float(os.getenv("OOD_MSP_THRESHOLD", "0.98"))

# 补充判据：平均色度（mean(max(R,G,B) - min(R,G,B))）。
# 起因：用户上传**黑白漫画**被判「桃（健康）98.4%」而未被拒识 —— MSP=0.984 恰好越过
# 0.98 阈值，说明 MSP 在非叶片样本上存在长尾（评估集上漏检约 17%）。
# 色度与 MSP 互补：灰阶图（黑白漫画 / 线稿 / 扫描件）色度接近 0，而彩色叶片照片
#   实测色度最低 10.6 —— 取阈值 8.0 时对 755 张真实叶片**零误伤**。
# 组合效果（400 张真实叶片 vs 363 张非叶片实测）：
#   仅 MSP        → 误杀 6.2%、拦截 82.9%
#   仅 色度        → 误杀 0.0%、拦截 51.8%
#   MSP 或 色度    → 误杀 6.2%、拦截 93.1%（色度补拦了 MSP 漏掉样本的 59.7%）
# 详见 docs/08_OOD拒识评估.md。
OOD_CHROMA_THRESHOLD = float(os.getenv("OOD_CHROMA_THRESHOLD", "8.0"))

# ===================== 叶片/非叶片检测头（方案 C2，当前主判据） =====================
# 起因：MSP + 色度在**真实田间**场景失效 —— 实测 651 张真实田间叶片中 53.8% 被判
# 「非叶片」（色度零误伤，误杀全部由 MSP 触发）。根因是 PlantVillage（黑底居中）
# 与真实田间是两个域，softmax 置信度无法跨域使用；而对 38 类模型做参数微调来修
# 会陷入跷跷板（改善 OOD 就把 PV 精度从 99.34% 打到 86.89%）。
#
# 方案：训练独立的二分类检测头（冻结模型的 512 维池化特征 → 逻辑回归），
# 与 38 类识别**解耦**，原模型权重分毫不动。
#   真实叶 vs 非叶 AUROC：0.99（同分布留出）/ 0.77~0.99（负样本留一来源的严格验证）
#   对照：38 类模型 MSP 仅 0.553
# 训练：scripts/train_ood_head.py  →  产物 models/ood_head.npz（纯数值，无 pickle）
OOD_HEAD_PATH = MODEL_DIR / "ood_head.npz"
OOD_HEAD_ENABLE = os.getenv("OOD_HEAD_ENABLE", "1").strip() == "1"

# 决策阈值覆盖：默认 None → 使用 npz 内记录的 0.5（逻辑回归概率的自然边界）。
# 设为 0~1 的值可在运行期调整：调高 → 拦得更严；调低 → 更宽松。
# 只影响判定，不改模型，可随时回退。
_HEAD_THR_ENV = os.getenv("OOD_HEAD_THRESHOLD", "").strip()
OOD_HEAD_THRESHOLD = float(_HEAD_THR_ENV) if _HEAD_THR_ENV else None

# ⚠️ 语义变更（2026-10-19 审计后）：平均色度**不再参与「是否非叶片」的判定**。
# 原因：该判据隐含「叶片照片必为彩色」这一强假设，而实测 40 张真实叶片转灰度后
# 检测头（灰度增强版）可判对 93%，色度却会一票否决、造成 100% 误杀。
# 现色度只用于**选择提示语措辞**（"这是黑白图" vs "把握不足"），以及给灰阶图
# 加一条"结果可能受影响"的温和提醒，不影响是否拦截。

# ===================== 作物先验（提高分类准确率） =====================
# 背景：实测 120 张 PlantDoc test 真实照片，模型 Top-1 仅 17.5%。错误模式主要是
# 「只看斑点纹理、不看作物」——苹果雪松锈病的橙黄斑点被判成玉米灰斑病，置信度 0.96~1.00。
# 若用户先指定作物，则把 softmax 限制在该作物的候选类内，多类别作物上实测
# 15.8% → 43.6%（+27.7pp），且**不动模型权重**、可随时关闭。
#
# 为什么必须由用户指定而不是模型自动判断：实测模型自动识别作物只有 33.3% 准确
# （玉米 92%、苹果 8%、樱桃/树莓 0%），拿它做掩码会把正确答案直接排除（收益归零）。
#
# ⚠️ 关键约束：掩码只作用于**展示用的概率**，OOD 判据（是否叶片）必须用
# 未掩码的全类 softmax —— 否则候选变少会让置信度虚高，非叶片图片更容易蒙混过关。
CROP_PRIOR_ENABLE = os.getenv("CROP_PRIOR_ENABLE", "1").strip() == "1"

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
# 密钥优先读环境变量 JWT_SECRET（生产必须设置）。
#
# 安全修复（审计 P0）：此前缺失环境变量时会回退到**写死在源码里、随仓库分发**的
# 常量 "dev-only-insecure-secret-DO-NOT-USE-IN-PROD"。任何读到该常量的人都能离线
# 伪造任意用户的 token —— 实测用它签发 {"uid":1} 可直接读取他人账户资料。
# 现改为：未设置时**为本次进程生成随机密钥**：
#   · 默认配置下不再存在「公开可预测的密钥」，离线伪造失效；
#   · 不阻断启动，本地开发体验不受影响；
#   · 代价：进程重启后此前签发的 token 全部失效，需重新登录（开发场景可接受）。
# 生产/长期部署仍应固定设置 JWT_SECRET，否则每次重启即全员登出。
_JWT_SECRET_ENV = os.getenv("JWT_SECRET", "").strip()
if _JWT_SECRET_ENV:
    JWT_SECRET = _JWT_SECRET_ENV
else:
    import secrets as _secrets
    import warnings
    JWT_SECRET = _secrets.token_urlsafe(48)
    warnings.warn(
        "JWT_SECRET 未通过环境变量设置，已为本次进程生成随机密钥。"
        "进程重启后所有已签发 token 将失效（需重新登录）。"
        "长期部署请固定设置环境变量 JWT_SECRET（建议 32 位以上随机串）。",
        stacklevel=2,
    )
JWT_EXPIRE_DAYS = int(os.getenv("JWT_EXPIRE_DAYS", "7"))
RATE_LIMIT_PER_MIN = 60                           # 单 IP 识别接口限流（API 文档 5）

# ===================== HTTPS / 传输加密（任务书 6.2） =====================
# 两种部署方式（详见 docs/13_部署运维手册.md）：
#   A. 反向代理终止 TLS（推荐）：Nginx 持证书，后端仍监听 127.0.0.1:8000，
#      并设置 TRUST_PROXY=1 让限流按 X-Forwarded-For 取真实来源。
#   B. 应用直连 TLS：设置 TLS_CERT / TLS_KEY，启动时带
#      --ssl-keyfile/--ssl-certfile；适合无代理的小规模部署。
TLS_CERT = os.getenv("TLS_CERT", "").strip()
TLS_KEY = os.getenv("TLS_KEY", "").strip()
# 设为 1 后：非 HTTPS 请求一律 308 跳转到 https（代理已终止 TLS 的场景靠
# X-Forwarded-Proto 判断，故需同时开启 TRUST_PROXY，否则会被反复跳转）。
REQUIRE_HTTPS = os.getenv("REQUIRE_HTTPS", "0").strip() == "1"
HSTS_MAX_AGE = int(os.getenv("HSTS_MAX_AGE", "31536000"))

# ===================== 数据库字段加密（任务书 6.2「数据库存储加密」） =====================
# 对 users.email 等 PII 字段做 Fernet(AES) 加密后落库。
# 与 JWT_SECRET 的关键差异：**密钥必须稳定**。缺失时只能禁用加密并告警，
# 不能像 JWT 那样随机生成 —— 否则进程重启后将无法解密历史数据。
# 生成：py -3.12 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
DB_FIELD_KEY = os.getenv("DB_FIELD_KEY", "").strip()
RATE_LIMIT_LOGIN_PER_MIN = 10                     # 单 IP 登录/注册限流（防口令爆破）

# 是否信任反向代理透传的 X-Forwarded-For / X-Real-IP。
# 安全修复（审计 P1）：默认 **不信任**。此前无条件采信这两种请求头，攻击者每次
# 请求换一个 X-Forwarded-For 即可让限流完全失效（实测连续 65 次请求一次未拦）。
# 仅当确实部署在可信反向代理（Nginx / 网关）之后时才设为 1；
# 否则保持 0，限流按真实 TCP 来源地址计算。
#
# ⚠️ 必须与启动参数配合：uvicorn 自身默认启用 ProxyHeadersMiddleware，且默认信任
# 来自 127.0.0.1 的转发头 —— 它会在 ASGI 层抢先覆写 scope["client"]，使应用层读到
# 的 request.client **已经是伪造值**（应用层无从察觉）。因此本开关为 0 时，
# 启动命令必须带 --no-proxy-headers（见 README 第 6.1 节）。
TRUST_PROXY = os.getenv("TRUST_PROXY", "0").strip() == "1"

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
