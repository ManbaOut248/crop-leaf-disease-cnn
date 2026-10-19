# -*- coding: utf-8 -*-
"""HTTPS 与强制跳转的端到端验证（任务书 6.2）。

用法（先跑 scripts/gen_selfsigned_cert.py 生成证书）：
    py -3.12 scripts/verify_tls.py

用 subprocess 起 uvicorn，再用 urllib 发请求，避免 Git Bash 的变量/后台化陷阱。
验证四件事：
  1. TLS 直连可用（自签名证书 → 用不校验证书的 context）
  2. REQUIRE_HTTPS=1 时 HTTP 请求 308 跳转到 https
  3. HTTPS 响应带 HSTS
  4. 代理场景（TRUST_PROXY=1 + X-Forwarded-Proto: https）不跳转，且带 HSTS
"""

import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request

PY = sys.executable
HEALTH = "/api/v1/health"


def start(port, env_extra, ssl_args=()):
    import os
    env = os.environ.copy()
    env.update(env_extra)
    cmd = [PY, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1",
           "--port", str(port), "--no-proxy-headers", *ssl_args]
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    # 用 TCP 连通性判断就绪：HTTP 方式在"强制跳转"场景会被 308 打乱
    # （urllib 会自动跟随到 https://127.0.0.1:port，而该端口没有 TLS）
    import socket
    for _ in range(90):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                time.sleep(1.5)      # 再等 uvicorn 完成 startup
                return p
        except OSError:
            time.sleep(1)
    p.terminate()
    raise RuntimeError(f"端口 {port} 启动超时")


def probe(url, headers=None, ctx=None):
    """返回 (状态码, 头字典, 响应体前 100 字节)。不自动跟随跳转。"""
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None

    handlers = [NoRedirect]
    if ctx is not None:
        # 注意：OpenerDirector.open() 不接受 context 参数，
        # 必须通过 HTTPSHandler 注入（否则会抛 TypeError）
        handlers.append(urllib.request.HTTPSHandler(context=ctx))
    opener = urllib.request.build_opener(*handlers)
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with opener.open(req, timeout=10) as r:
            return r.status, {k.lower(): v for k, v in dict(r.headers).items()}, r.read(100)
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in dict(e.headers).items()}, e.read(100)


results = []

# ---------- A) TLS 直连 + HSTS ----------
p = start(8443, {"REQUIRE_HTTPS": "1"},
          ["--ssl-keyfile", "certs/dev-key.pem", "--ssl-certfile", "certs/dev-cert.pem"])
try:
    ctx = ssl._create_unverified_context()
    code, hdr, body = probe("https://127.0.0.1:8443" + HEALTH, ctx=ctx)
    results.append(("HTTPS 直连（自签证书）", code, code == 200))
    hsts = hdr.get("strict-transport-security", "")
    results.append(("HTTPS 响应带 HSTS", hsts or "(无)", bool(hsts)))
finally:
    p.terminate()
    p.wait(timeout=10)

# ---------- B) HTTP → HTTPS 跳转 ----------
p = start(8001, {"REQUIRE_HTTPS": "1"})
try:
    code, hdr, _ = probe("http://127.0.0.1:8001" + HEALTH)
    loc = hdr.get("location", "")
    results.append(("HTTP 请求跳转", f"{code} → {loc}", code == 308 and loc.startswith("https://")))
    results.append(("跳转响应不带 HSTS", hdr.get("strict-transport-security") or "(无)", not hdr.get("strict-transport-security")))
finally:
    p.terminate()
    p.wait(timeout=10)

# ---------- C) 代理场景（Nginx 终止 TLS）----------
p = start(8002, {"REQUIRE_HTTPS": "1", "TRUST_PROXY": "1"})
try:
    code, hdr, _ = probe("http://127.0.0.1:8002" + HEALTH,
                         headers={"X-Forwarded-Proto": "https"})
    hsts = hdr.get("strict-transport-security", "")
    results.append(("代理场景不跳转（X-Forwarded-Proto=https）", code, code == 200))
    results.append(("代���场景下带 HSTS", hsts or "(无)", bool(hsts)))

    # 伪造该头但 TRUST_PROXY=0 时应被忽略（对照：上面 B 组未带该头，已验跳转）
    code2, _, _ = probe("http://127.0.0.1:8002" + HEALTH,
                        headers={"X-Forwarded-Proto": "http"})
    results.append(("代理声明 http 时仍跳转", code2, code2 == 308))
finally:
    p.terminate()
    p.wait(timeout=10)

print("=" * 74)
print("HTTPS / 传输加密验证")
print("=" * 74)
ok = 0
for name, got, passed in results:
    ok += bool(passed)
    print(f"  {'✓' if passed else '✗'} {name:<44} {got}")
print("-" * 74)
print(f"通过 {ok}/{len(results)}")
sys.exit(0 if ok == len(results) else 1)
