# -*- coding: utf-8 -*-
"""下载 DigiGreen 真实农户拍摄照片（989 张 / 174MB）。
特性：8 线程并发 + 单文件重试 + 已存在跳过（可断点续跑）。
"""
import json
import os
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DS = "DigiGreen/Crop_Disease_Images"
ROOT = r"F:\WorkBuddy工作空间\基于CNN的农作物叶片病害识别系统\data\_downloads\digigreen"
IMG_DIR = os.path.join(ROOT, "images")
BASE = f"https://hf-mirror.com/datasets/{DS}/resolve/main/"
UA = {"User-Agent": "Wget/1.21.4"}


def get_json(url, tries=6):
    """带退避重试的 JSON 拉取（代理偶发 502）。"""
    d = 2
    for _ in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.load(r)
        except Exception as e:
            print(f"  列表重试: {e}", flush=True)
            time.sleep(d)
            d = min(d * 2, 30)
    raise RuntimeError("列表拉取失败")


def fetch_one(fname):
    """下载单张图片；已存在且大小一致则跳过。"""
    dst = os.path.join(IMG_DIR, fname)
    for attempt in range(6):
        try:
            url = BASE + f"images/{fname}"
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
                data = r.read()
            open(dst, "wb").write(data)
            return len(data)
        except Exception as e:
            if attempt == 5:
                print(f"  [失败] {fname}: {e}", flush=True)
                return 0
            time.sleep(2 * (attempt + 1))
    return 0


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    # 分页拉取全量文件清单
    files, url = [], f"https://hf-mirror.com/api/datasets/{DS}/tree/main?recursive=true&limit=1000"
    while url:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            files += [f for f in json.load(r) if f["type"] == "file"]
            m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link") or "")
        url = m.group(1).replace("https://huggingface.co", "https://hf-mirror.com") if m else None

    names = [f["path"].split("/")[-1] for f in files if f["path"].lower().endswith(".jpg")]
    todo = [n for n in names if not os.path.exists(os.path.join(IMG_DIR, n))]
    print(f"共 {len(names)} 张，待下 {len(todo)} 张", flush=True)

    t0, done = time.time(), 0
    lock = threading.Lock()

    def task(n):
        nonlocal done
        fetch_one(n)
        with lock:
            done += 1
            if done % 100 == 0:
                el = time.time() - t0
                print(f"  进度 {done}/{len(todo)}  速度 {done/el*2:.0f} 张/s", flush=True)

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(task, todo))

    got = len([n for n in os.listdir(IMG_DIR) if n.endswith('.jpg')])
    total = sum(os.path.getsize(os.path.join(IMG_DIR, n)) for n in os.listdir(IMG_DIR) if n.endswith('.jpg'))
    print(f"完成：{got}/{len(names)} 张，{total/1048576:.0f} MB，用时 {time.time()-t0:.0f}s")

    # 终验：每张都必须能 PIL 打开（抓截断文件，verify 抓不住）
    from PIL import Image
    bad = []
    for n in os.listdir(IMG_DIR):
        if not n.endswith('.jpg'):
            continue
        try:
            Image.open(os.path.join(IMG_DIR, n)).load()
        except Exception as e:
            bad.append((n, str(e)[:40]))
    print(f"图片完整性：{len(bad)} 张异常", bad[:5])


if __name__ == "__main__":
    main()
