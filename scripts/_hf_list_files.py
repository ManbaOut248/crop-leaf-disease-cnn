# -*- coding: utf-8 -*-
"""列出 HF 数据集全量文件（分页 + 502 重试），结果落 JSON 便于后续筛选。"""
import json
import re
import sys
import time
import urllib.request

ds = sys.argv[1] if len(sys.argv) > 1 else "KeenForgeAI/PlantDoc-corrected"
out = sys.argv[2] if len(sys.argv) > 2 else "outputs/_hf_tree.json"
UA = {"User-Agent": "Wget/1.21.4"}


def get(url, tries=8):
    """带指数退避的 GET（沙箱代理会偶发 502）。"""
    delay = 2
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.load(r), r.headers.get("Link") or ""
        except Exception as e:
            print(f"  重试{i+1}: {e}", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 30)
    raise RuntimeError(f"多次重试仍失败: {url}")


url = f"https://hf-mirror.com/api/datasets/{ds}/tree/main?recursive=true&limit=1000"
files = []
while url:
    batch, link = get(url)
    files += [f for f in batch if f["type"] == "file"]
    m = re.search(r'<([^>]+)>;\s*rel="next"', link)
    # 分页链接由后端返回的是 huggingface.co 原站地址，必须改写为镜像域名
    url = m.group(1).replace("https://huggingface.co", "https://hf-mirror.com") if m else None
    print(f"  累计 {len(files)}", flush=True)

json.dump(files, open(out, "w", encoding="utf-8"), ensure_ascii=False)
print(f"完成：{len(files)} 个文件，合计 {sum(f.get('size',0) for f in files)/1048576:.0f} MB -> {out}")
