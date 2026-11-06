# -*- coding: utf-8 -*-
"""
交付包构建脚本：把工作空间里的交付物按《项目任务书》的十项任务与 5.2 文档成果分类，
复制到 F:\\实训产品\\基于CNN的农作物叶片病害识别系统\\ 。

设计原则：
1. 只复制、不移动 —— 原工作空间保持可运行，避免破坏开发环境。
2. 不复制 node_modules / __pycache__ / 中间 checkpoint 备份 —— 体积大且可重建。
3. PlantVillage 原地不动 —— src/config.py 的 DATA_ROOT 默认值直指该目录。
4. 幂等：重复执行只做覆盖，不产生重复目录。
用法：X:\\Anconda3\\python.exe scripts/build_delivery_package.py
"""

import os
import shutil
import sys
from pathlib import Path

# ============ 源与目标 ============
SRC = Path(r"F:\WorkBuddy工作空间\基于CNN的农作物叶片病害识别系统")
DST = Path(r"F:\实训产品\基于CNN的农作物叶片病害识别系统")

# 复制时全局忽略的目录名 / 文件后缀
IGNORE_DIRS = {"node_modules", "__pycache__", ".pytest_cache", ".git", "dist", "dist-verify",
               "backup_preC", "_tmp_tls", "_tmp_verify_gray"}
IGNORE_SUFFIX = {".pyc", ".pyo", ".log"}

# ============ 目录骨架 ============
DIRS = [
    "00_交付总览",
    "01_需求与设计（任务1、5）",
    "02_数据集与预处理（任务2）",
    "02_数据集与预处理（任务2）/数据集索引",
    "02_数据集与预处理（任务2）/预处理与增强代码",
    "03_模型训练与评估（任务3）",
    "03_模型训练与评估（任务3）/训练与评估代码",
    "03_模型训练与评估（任务3）/训练与实验脚本",
    "03_模型训练与评估（任务3）/评估报告/真实域评估",
    "03_模型训练与评估（任务3）/评估报告/OOD与校准实验",
    "03_模型训练与评估（任务3）/实验与分析文档",
    "04_后端服务与部署（任务4、6、8）",
    "04_后端服务与部署（任务4、6、8）/后端源码",
    "04_后端服务与部署（任务4、6、8）/推理服务",
    "04_后端服务与部署（任务4、6、8）/部署配置",
    "04_后端服务与部署（任务4、6、8）/依赖与环境",
    "05_前端Web界面（任务7）",
    "05_前端Web界面（任务7）/界面截图",
    "06_系统测试与质量（任务9）",
    "06_系统测试与质量（任务9）/自动化测试",
    "06_系统测试与质量（任务9）/测试图像样例",
    "06_系统测试与质量（任务9）/兼容性与主题实测截图",
    "07_演示与答辩（任务10）",
    "07_演示与答辩（任务10）/PPT截图素材",
    "08_RAG语料库（5.2-⑥）",
    "09_AI协作教程（5.2-⑤）",
    "10_完整工程源码",
]

# ============ 复制清单：(相对源路径, 相对目标路径) 或 (源, 目标, 是否需要改名) ============
FILES = [
    # ---- 00 交付总览 ----
    ("README.md", "00_交付总览/README_项目说明（参数文档）.md"),
    ("overview.md", "00_交付总览/项目一页纸.md"),
    ("docs/12_任务书对照检查.md", "00_交付总览/任务书对照检查报告.md"),
    ("农作物叶片病害识别系统项目任务书V1.0.0.docx", "00_交付总览/项目任务书V1.0.0.docx"),

    # ---- 01 需求与设计 ----
    ("docs/01_PRD.md", "01_需求与设计（任务1、5）/01_PRD需求文档.md"),
    ("docs/02_UIUX设计.md", "01_需求与设计（任务1、5）/02_UIUX设计文档.md"),
    ("docs/03_数据库设计.md", "01_需求与设计（任务1、5）/03_数据库设计文档.md"),
    ("docs/assets/architecture.svg", "01_需求与设计（任务1、5）/系统架构设计图.svg"),
    ("docs/prototype/ui-prototype.html", "01_需求与设计（任务1、5）/交互原型（5画板）.html"),

    # ---- 02 数据集与预处理 ----
    ("docs/10_叶片病害数据集获取指南.md", "02_数据集与预处理（任务2）/数据集获取指南.md"),
    ("outputs/index/dataset_index.json", "02_数据集与预处理（任务2）/数据集索引/dataset_index.json"),
    ("outputs/index/dataset_index_mixed.json", "02_数据集与预处理（任务2）/数据集索引/dataset_index_mixed.json"),
    ("outputs/index/dataset_index_realtest.json", "02_数据集与预处理（任务2）/数据集索引/dataset_index_realtest.json"),
    ("src/data/transforms.py", "02_数据集与预处理（任务2）/预处理与增强代码/transforms.py"),
    ("src/data/datasets.py", "02_数据集与预处理（任务2）/预处理与增强代码/datasets.py"),
    ("src/data/build_index.py", "02_数据集与预处理（任务2）/预处理与增强代码/build_index.py"),
    ("src/data/materialize_split.py", "02_数据集与预处理（任务2）/预处理与增强代码/materialize_split.py"),
    ("src/data/measure_leakage.py", "02_数据集与预处理（任务2）/预处理与增强代码/measure_leakage.py"),
    ("src/data/disease_knowledge.json", "02_数据集与预处理（任务2）/预处理与增强代码/disease_knowledge.json"),

    # ---- 03 模型训练与评估 ----
    ("src/model/resnet.py", "03_模型训练与评估（任务3）/训练与评估代码/resnet.py"),
    ("src/model/train.py", "03_模型训练与评估（任务3）/训练与评估代码/train.py"),
    ("src/model/evaluate.py", "03_模型训练与评估（任务3）/训练与评估代码/evaluate.py"),
    ("src/model/compare.py", "03_模型训练与评估（任务3）/训练与评估代码/compare.py"),
    ("src/model/prune.py", "03_模型训练与评估（任务3）/训练与评估代码/prune.py"),
    ("src/model/quantize.py", "03_模型训练与评估（任务3）/训练与评估代码/quantize.py"),
    ("outputs/confusion_matrix.png", "03_模型训练与评估（任务3）/评估报告/混淆矩阵.png"),
    ("outputs/classification_report.txt", "03_模型训练与评估（任务3）/评估报告/分类报告_classification_report.txt"),
    ("outputs/eval_report.json", "03_模型训练与评估（任务3）/评估报告/eval_report.json"),
    ("outputs/error_samples.csv", "03_模型训练与评估（任务3）/评估报告/error_samples.csv"),
    ("outputs/class_mapping.json", "03_模型训练与评估（任务3）/评估报告/class_mapping.json"),
    ("outputs/compare.json", "03_模型训练与评估（任务3）/评估报告/模型压缩对比_compare.json"),
    ("outputs/web_eval_report.json", "03_模型训练与评估（任务3）/评估报告/真实域评估/web_eval_report.json"),
    ("outputs/ood_eval.json", "03_模型训练与评估（任务3）/评估报告/OOD与校准实验/ood_eval.json"),
    ("outputs/ood_head_result.json", "03_模型训练与评估（任务3）/评估报告/OOD与校准实验/ood_head_result.json"),
    ("outputs/ood_head_xsrc.json", "03_模型训练与评估（任务3）/评估报告/OOD与校准实验/ood_head_xsrc.json"),
    ("outputs/ood_ab_result.json", "03_模型训练与评估（任务3）/评估报告/OOD与校准实验/ood_ab_result.json"),
    ("outputs/crop_prior_report.json", "03_模型训练与评估（任务3）/评估报告/OOD与校准实验/crop_prior_report.json"),
    ("outputs/improvement_probe.json", "03_模型训练与评估（任务3）/评估报告/OOD与校准实验/improvement_probe.json"),
    ("outputs/domain_head_eval.json", "03_模型训练与评估（任务3）/评估报告/OOD与校准实验/domain_head_eval.json"),
    ("outputs/oe_adapt_result.json", "03_模型训练与评估（任务3）/评估报告/OOD与校准实验/oe_adapt_result.json"),
    ("outputs/oe_final_result.json", "03_模型训练与评估（任务3）/评估报告/OOD与校准实验/oe_final_result.json"),
    ("outputs/oe_final_result_t150.json", "03_模型训练与评估（任务3）/评估报告/OOD与校准实验/oe_final_result_t150.json"),
    ("docs/06_模型压缩对比.md", "03_模型训练与评估（任务3）/实验与分析文档/06_模型压缩对比.md"),
    ("docs/08_OOD拒识评估.md", "03_模型训练与评估（任务3）/实验与分析文档/08_OOD拒识评估.md"),
    ("docs/11_域适应与OOD检测实验.md", "03_模型训练与评估（任务3）/实验与分析文档/11_域适应与OOD检测实验.md"),
    ("docs/14_置信度校准.md", "03_模型训练与评估（任务3）/实验与分析文档/14_置信度校准.md"),
    ("docs/15_真实域混合微调.md", "03_模型训练与评估（任务3）/实验与分析文档/15_真实域混合微调.md"),
    ("docs/16_320训练续训备忘.md", "03_模型训练与评估（任务3）/实验与分析文档/16_320训练续训备忘.md"),
    ("docs/17_双分辨率模式.md", "03_模型训练与评估（任务3）/实验与分析文档/17_双分辨率模式.md"),
    # 以下 8 篇此前是手工补进去的，2026-10-29 一并写进映射，重跑打包才不会漏
    ("docs/18_逐作物识别率短板分析.md", "03_模型训练与评估（任务3）/实验与分析文档/18_逐作物识别率短板分析.md"),
    ("docs/19_224与320实测对比.md", "03_模型训练与评估（任务3）/实验与分析文档/19_224与320实测对比.md"),
    ("docs/20_历史页图表功能.md", "03_模型训练与评估（任务3）/实验与分析文档/20_历史页图表功能.md"),
    ("docs/21_裁剪有效性实验.md", "03_模型训练与评估（任务3）/实验与分析文档/21_裁剪有效性实验.md"),
    ("docs/22_真实域加权与防泄漏对照实验.md", "03_模型训练与评估（任务3）/实验与分析文档/22_真实域加权与防泄漏对照实验.md"),
    ("docs/23_外部域评估_DigiGreen农户实拍.md", "03_模型训练与评估（任务3）/实验与分析文档/23_外部域评估_DigiGreen农户实拍.md"),
    ("docs/24_拍法引导与单片裁剪.md", "03_模型训练与评估（任务3）/实验与分析文档/24_拍法引导与单片裁剪.md"),
    ("docs/25_能力边界评估_DigiGreen全量.md", "03_模型训练与评估（任务3）/实验与分析文档/25_能力边界评估_DigiGreen全量.md"),

    # ---- 04 后端与部署 ----
    ("docs/04_API接口文档.md", "04_后端服务与部署（任务4、6、8）/API接口文档.md"),
    ("docs/13_部署运维手册.md", "04_后端服务与部署（任务4、6、8）/部署运维手册.md"),
    ("src/config.py", "04_后端服务与部署（任务4、6、8）/后端源码/config.py"),
    ("src/api/main.py", "04_后端服务与部署（任务4、6、8）/后端源码/main.py"),
    ("src/api/auth.py", "04_后端服务与部署（任务4、6、8）/后端源码/auth.py"),
    ("src/api/crypto.py", "04_后端服务与部署（任务4、6、8）/后端源码/crypto.py"),
    ("src/api/db.py", "04_后端服务与部署（任务4、6、8）/后端源码/db.py"),
    ("src/api/schemas.py", "04_后端服务与部署（任务4、6、8）/后端源码/schemas.py"),
    ("src/api/thumbs.py", "04_后端服务与部署（任务4、6、8）/后端源码/thumbs.py"),
    ("src/model/predict.py", "04_后端服务与部署（任务4、6、8）/推理服务/predict.py"),
    ("src/model/resnet.py", "04_后端服务与部署（任务4、6、8）/推理服务/resnet.py"),
    ("deploy/nginx.conf.example", "04_后端服务与部署（任务4、6、8）/部署配置/nginx.conf.example"),
    ("requirements.txt", "04_后端服务与部署（任务4、6、8）/依赖与环境/requirements.txt"),
    ("pytest.ini", "04_后端服务与部署（任务4、6、8）/依赖与环境/pytest.ini"),

    # ---- 06 测试 ----
    ("docs/05_测试报告.md", "06_系统测试与质量（任务9）/05_测试报告.md"),
    ("docs/09_代码审计报告.md", "06_系统测试与质量（任务9）/09_代码审计报告.md"),
    ("outputs/compat/report.json", "06_系统测试与质量（任务9）/兼容性与主题实测截图/compat_report.json"),

    # ---- 07 演示与答辩 ----
    ("03.ppt/农作物叶片病害识别系统/农作物叶片病害识别系统_项目汇报.pptx",
     "07_演示与答辩（任务10）/项目汇报PPT_精简8页.pptx"),
    ("03.ppt/农作物叶片病害识别系统/农作物叶片病害识别系统_项目汇报_详细版16页.pptx",
     "07_演示与答辩（任务10）/项目汇报PPT_详细版16页.pptx"),
    ("03.ppt/农作物叶片病害识别系统/农作物叶片病害识别系统.pptx",
     "07_演示与答辩（任务10）/项目汇报PPT_V3演示版15页.pptx"),
    ("03.ppt/农作物叶片病害识别系统/DESIGN.md", "07_演示与答辩（任务10）/PPT设计说明_DESIGN.md"),
    ("03.ppt/农作物叶片病害识别系统/STORY.md", "07_演示与答辩（任务10）/PPT讲稿_STORY.md"),

    # ---- 09 AI 协作教程 ----
    ("docs/07_AI协作教程.md", "09_AI协作教程（5.2-⑤）/AI协作教程.md"),

    # ---- 10 完整工程 ----
    ("requirements.txt", "10_完整工程源码/requirements.txt"),
    ("pytest.ini", "10_完整工程源码/pytest.ini"),
    ("README.md", "10_完整工程源码/README.md"),
    ("overview.md", "10_完整工程源码/overview.md"),
]

# 目录整体复制：(源相对目录, 目标相对目录)
TREES = [
    ("scripts", "03_模型训练与评估（任务3）/训练与实验脚本"),
    ("src/web", "05_前端Web界面（任务7）/前端源码"),
    ("03.ppt/农作物叶片病害识别系统/assets_v4", "05_前端Web界面（任务7）/界面截图"),
    ("outputs/compat", "06_系统测试与质量（任务9）/兼容性与主题实测截图"),
    ("outputs/theme_shots", "06_系统测试与质量（任务9）/兼容性与主题实测截图/主题配色截图"),
    ("tests", "06_系统测试与质量（任务9）/自动化测试"),
    ("test_images", "06_系统测试与质量（任务9）/测试图像样例"),
    ("03.ppt/农作物叶片病害识别系统/assets", "07_演示与答辩（任务10）/PPT截图素材"),
    ("rag", "08_RAG语料库（5.2-⑥）"),
    # 完整工程源码
    ("src", "10_完整工程源码/src"),
    ("scripts", "10_完整工程源码/scripts"),
    ("tests", "10_完整工程源码/tests"),
    ("rag", "10_完整工程源码/rag"),
    ("models", "10_完整工程源码/models"),
]


def _ignore(dirname, names):
    """shutil.copytree 的 ignore 回调：过滤缓存目录与编译产物。"""
    skipped = set()
    for n in names:
        p = Path(dirname) / n
        if p.is_dir() and n in IGNORE_DIRS:
            skipped.add(n)
        elif p.is_file() and p.suffix.lower() in IGNORE_SUFFIX:
            skipped.add(n)
    return skipped


def copy_file(src_rel: str, dst_rel: str) -> str:
    s, d = SRC / src_rel, DST / dst_rel
    if not s.exists():
        return f"  跳过（不存在）: {src_rel}"
    d.parent.mkdir(parents=True, exist_ok=True)
    # 同一文件已被别处复制过就不重复写（如 scripts 复制到 03 与 10 两处属正常，需各写一次）
    shutil.copy2(s, d)
    return None


def main() -> int:
    if not SRC.exists():
        print(f"源目录不存在: {SRC}")
        return 2
    DST.mkdir(parents=True, exist_ok=True)

    # 1) 建目录
    for rel in DIRS:
        (DST / rel).mkdir(parents=True, exist_ok=True)
    print(f"[1/4] 目录骨架 {len(DIRS)} 个已就绪")

    # 2) 单文件复制
    misses = []
    for item in FILES:
        src_rel, dst_rel = item[0], item[1]
        msg = copy_file(src_rel, dst_rel)
        if msg:
            misses.append(msg)
    print(f"[2/4] 单文件复制完成，成功 {len(FILES) - len(misses)} / {len(FILES)}")
    for m in misses:
        print(m)

    # 3) 目录树复制
    ok = 0
    for src_rel, dst_rel in TREES:
        s, d = SRC / src_rel, DST / dst_rel
        if not s.exists():
            print(f"  跳过（不存在）: {src_rel}")
            continue
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(s, d, ignore=_ignore, dirs_exist_ok=True)
        ok += 1
    print(f"[3/4] 目录树复制完成 {ok} / {len(TREES)}")

    # 4) 统计
    total = sum(1 for _ in DST.rglob("*") if _.is_file())
    size = sum(f.stat().st_size for f in DST.rglob("*") if f.is_file())
    print(f"[4/4] 交付包共 {total} 个文件，{size / 1024 / 1024:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
