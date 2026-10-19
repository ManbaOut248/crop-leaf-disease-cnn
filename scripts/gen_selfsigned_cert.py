# -*- coding: utf-8 -*-
"""生成开发/内网用的自签名 TLS 证书（任务书 6.2 传输加密）。

为什么需要它
---------------------------------------------------------------
任务书要求「HTTPS 加密传输」。本地/内网验收没有公网域名，拿不到 CA 证书，
自签名证书是唯一能立刻跑通的方式：浏览器会提示"不安全"，但**链路确实加密**，
足以验证 TLS 配置、HSTS、跳转逻辑是否正确。

正式环境请用 Let's Encrypt（acme.sh / certbot）或云厂商证书替换本脚本产物。

用法
    py -3.12 scripts/gen_selfsigned_cert.py
    # 产物：certs/dev-key.pem（私钥，勿提交）、certs/dev-cert.pem

    然后启动后端：
    py -3.12 -m uvicorn src.api.main:app --host 127.0.0.1 --port 8443 \
        --ssl-keyfile certs/dev-key.pem --ssl-certfile certs/dev-cert.pem \
        --no-proxy-headers

    # 验证（-k 跳过自签名校验）
    curl -k https://127.0.0.1:8443/api/v1/health

注意
    · 私钥文件权限应限制为当前用户可读（Windows 下建议放非共享目录）；
    · 证书默认 825 天，过期后浏览器会直接拒绝连接，需重新生成；
    · `certs/` 已加入 .gitignore —— 私钥**绝不能**进版本库。
"""

import datetime
import ipaddress
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

OUT_DIR = Path("certs")
CERT_PATH = OUT_DIR / "dev-cert.pem"
KEY_PATH = OUT_DIR / "dev-key.pem"
DAYS = 825          # 浏览器对自签名证书的容忍上限约 825 天（Apple 平台限制）


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "CN"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "LeafDisease Dev"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))   # 容忍客户端时钟偏差
        .not_valid_after(now + datetime.timedelta(days=DAYS))
        # SAN 必须显式包含 127.0.0.1：新版浏览器已不再回退到 CN 匹配
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                x509.IPAddress(ipaddress.ip_address("::1")),
            ]),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )

    KEY_PATH.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    CERT_PATH.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    print(f"已生成：\n  {CERT_PATH}  （证书，{DAYS} 天有效）\n  {KEY_PATH}  （私钥，勿提交）")
    print("\n启动 HTTPS：")
    print("  py -3.12 -m uvicorn src.api.main:app --host 127.0.0.1 --port 8443 "
          "--ssl-keyfile certs/dev-key.pem --ssl-certfile certs/dev-cert.pem --no-proxy-headers")
    print("\n验证：curl -k https://127.0.0.1:8443/api/v1/health")


if __name__ == "__main__":
    main()
