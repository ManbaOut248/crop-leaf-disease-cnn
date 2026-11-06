# -*- coding: utf-8 -*-
"""
单张推理延迟基准：对比当前 IMAGE_SIZE（224 / 320）的端到端耗时。

用法（PowerShell / CMD 两次运行，各自独立进程）：
    python scripts/bench_resolution.py                 # 默认 224
    set IMAGE_SIZE=320&& python scripts/bench_resolution.py

计时口径：从 PIL.Image 输入到返回结果字典，含预处理与前向，不含模型加载与磁盘读取。
预热 5 次排除首轮 CUDA kernel 编译开销，正式计时 30 次取均值。
"""

import glob
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

from src import config
from src.model.predict import Predictor

WARMUP = 5
ROUNDS = 30


def main():
    # 固定同一张图，避免不同图片解码耗时干扰
    cands = glob.glob("test_images/01_叶片识别/*.jpg")
    if not cands:
        print("[错误] 未找到 test_images/01_叶片识别/*.jpg")
        return
    img_path = sorted(cands)[0]
    img = Image.open(img_path).convert("RGB")

    # 与 Predictor 保持一致：is_available() 为 True 但 device_count() 为 0 时仍算 CPU
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    device = "cuda" if use_cuda else "cpu"
    predictor = Predictor()

    # 预热
    for _ in range(WARMUP):
        predictor.predict_image(img, crop=None)

    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(ROUNDS):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        predictor.predict_image(img, crop=None)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)  # 转毫秒

    mean = statistics.mean(times)
    p50 = statistics.median(times)
    p95 = sorted(times)[int(len(times) * 0.95) - 1]

    print(f"IMAGE_SIZE   : {config.IMAGE_SIZE}")
    print(f"设备         : {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"测试图       : {os.path.basename(img_path)}")
    print(f"样本数       : {ROUNDS} 次（预热 {WARMUP} 次已排除）")
    print(f"平均延迟     : {mean:.1f} ms")
    print(f"中位 P50     : {p50:.1f} ms")
    print(f"P95          : {p95:.1f} ms")
    print(f"理论吞吐     : {1000 / mean:.1f} 张/秒（单请求串行）")


if __name__ == "__main__":
    main()
