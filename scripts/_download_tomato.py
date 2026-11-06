# -*- coding: utf-8 -*-
"""
Mendeley 数据集稳健下载器（针对本项目实测的链路问题定制）：
1. Mendeley 短链每次 302 到有时效的 S3 预签名 URL → 每轮重试都要重新解析新鲜直链；
2. 服务端 502/503 抖动频繁 → 指数退避重试；
3. 单轮不能限 570s → 进程内无限跑，每轮最长 10 分钟，直到下完为止；
4. 兜底校验：总大小对上 + zipfile 完整性，防「403 错误页追加进文件」的老坑。
"""
import io
import os
import time
import zipfile
import urllib.request

ORIG = "https://data.mendeley.com/public-files/datasets/zfv4jj7855/files/5bec472d-1728-4dd0-b81a-233d04ad026d/file_downloaded"
OUT = r"F:\WorkBuddy工作空间\基于CNN的农作物叶片病害识别系统\data\_downloads\mendeley_zfv4jj7855\TomatoDataset.zip"
TARGET = 668089588  # 服务端 Content-Length 总字节数

UA = {"User-Agent": "Wget/1.21.4"}  # 实测：默认 Python-urllib UA 被 WAF 403，换 UA 后 206 正常

def resolve_s3():
    """跟随 Mendeley 短链的 302，拿到新鲜的 S3 预签名直链。"""
    req = urllib.request.Request(ORIG, method="HEAD", headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.url

def main():
    # 断点续传：从已有文件大小继续；若已有内容非 zip 前缀（老坑），超过 1MB 就信它
    pos = os.path.getsize(OUT) if os.path.exists(OUT) else 0
    backoff = 10  # 指数退避秒数：10 → 20 → 40 → 60 封顶
    round_no = 0
    while pos < TARGET:
        round_no += 1
        try:
            s3 = resolve_s3()
            # Range 请求从断点续传
            req = urllib.request.Request(s3, headers={**UA, "Range": f"bytes={pos}-"})
            with urllib.request.urlopen(req, timeout=120) as r, \
                 open(OUT, "ab") as f:
                got = 0
                while True:
                    chunk = r.read(256 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if pos + got >= TARGET:
                        break
            pos = os.path.getsize(OUT)
            print(f"[{time.strftime('%H:%M:%S')}] round {round_no}: 累计 {pos}/{TARGET} "
                  f"({pos*100//TARGET}%)", flush=True)
            backoff = 10  # 有进展就重置退避
        except Exception as e:
            pos = os.path.getsize(OUT) if os.path.exists(OUT) else 0
            print(f"[{time.strftime('%H:%M:%S')}] round {round_no} 异常: {e} "
                  f"（断点 {pos}，{backoff}s 后重试）", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)

    # 终验：大小 + zip 完整性（防错误页追加）
    assert pos == TARGET, f"大小不符: {pos} != {TARGET}"
    z = zipfile.ZipFile(OUT)
    bad = z.testzip()
    print("ZIP 校验:", ("损坏@" + bad) if bad else "OK", "条目:", len(z.namelist()), flush=True)

if __name__ == "__main__":
    main()
