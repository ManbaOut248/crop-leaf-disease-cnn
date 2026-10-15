# -*- coding: utf-8 -*-
"""
认证模块：密码 bcrypt 哈希 + 轻量 JWT（HMAC 签名，无第三方依赖）。
- 密码：bcrypt（自带盐，抗暴力/抗 GPU，符合 PRD NFR-SEC 安全基线）。
        旧账户（sha256+salt 方案）仍可被 verify_password 兼容校验，
        并在登录成功时自动升级为 bcrypt（见 src/api/main.py 的 login）。
- Token：base64url(header).base64url(payload).HMAC-SHA256，含 exp 过期时间。
说明：JWT_SECRET 已由环境变量注入（见 src/config.py），生产请启用 HTTPS。
"""

import hashlib
import hmac
import json
import secrets
import time
from base64 import urlsafe_b64encode, urlsafe_b64decode

import bcrypt

from src.config import JWT_SECRET, JWT_EXPIRE_DAYS

# bcrypt 哈希以 $2 开头（如 $2b$12$...），用于区分新旧账户存储格式
_BCRYPT_PREFIXES = ("$2a$", "$2b$", "$2y$")

# 占位哈希：用于「用户名不存在」分支做一次等价 bcrypt 计算。
# 目的：抹平「用户存在（走 bcrypt，约 300ms）」与「用户不存在（直接返回，约 6ms）」
# 之间的耗时差，否则攻击者可用响应时间枚举已注册用户名（审计 P2，实测差异 324ms）。
_DUMMY_HASH = bcrypt.hashpw(b"constant-time-placeholder", bcrypt.gensalt())


def dummy_verify(password: str) -> None:
    """对不存在的用户执行一次等价的 bcrypt 校验，仅用于消耗等量时间（抹平时序差异）。"""
    try:
        bcrypt.checkpw(password.encode("utf-8"), _DUMMY_HASH)
    except ValueError:
        pass


# ---------- base64url 工具 ----------
def _b64url_encode(b: bytes) -> bytes:
    return urlsafe_b64encode(b).rstrip(b"=")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return urlsafe_b64decode(s + pad)


# ---------- 密码哈希 ----------
def hash_password(password: str, salt: str = None):
    """返回 (salt, hash)。新账户使用 bcrypt（盐内嵌于哈希，salt 字段留空）。
    保留 salt 形参仅为兼容旧调用点；新账户 salt 恒为 ""。"""
    pw = password.encode("utf-8")
    hashed = bcrypt.hashpw(pw, bcrypt.gensalt())
    return "", hashed.decode("utf-8")


def is_legacy_hash(stored_hash: str) -> bool:
    """判断存储的哈希是否为旧的 sha256(salt+pw) 格式（用于登录时自动升级）。"""
    if not stored_hash:
        return False
    return not stored_hash.startswith(_BCRYPT_PREFIXES)


def verify_password(password: str, salt: str, expected_hash: str) -> bool:
    """校验密码。自动兼容 bcrypt 新账户与旧 sha256(salt+pw) 账户。"""
    if not expected_hash:
        return False
    # 新账户：bcrypt 校验（哈希内已含盐，忽略传入 salt）
    if expected_hash.startswith(_BCRYPT_PREFIXES):
        try:
            return bcrypt.checkpw(password.encode("utf-8"), expected_hash.encode("utf-8"))
        except ValueError:
            return False
    # 旧账户：sha256(salt + password)
    digest = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return hmac.compare_digest(digest, expected_hash)


# ---------- JWT ----------
def create_token(payload: dict, expire_days: int = JWT_EXPIRE_DAYS) -> str:
    """生成 HMAC 签名的 JWT（header.alg=HS256）。"""
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    body = dict(payload)
    body["iat"] = now
    body["exp"] = now + expire_days * 86400
    signing_input = _b64url_encode(json.dumps(header, separators=(",", ":")).encode()) \
        + b"." \
        + _b64url_encode(json.dumps(body, separators=(",", ":")).encode())
    sig = hmac.new(JWT_SECRET.encode(), signing_input, hashlib.sha256).digest()
    return (signing_input + b"." + _b64url_encode(sig)).decode()


def decode_token(token: str) -> dict:
    """校验签名与过期时间，成功返回 payload，失败返回 None。"""
    try:
        parts = token.encode().split(b".")
        if len(parts) != 3:
            return None
        signing_input = parts[0] + b"." + parts[1]
        sig = _b64url_decode(parts[2].decode())
        expected = hmac.new(JWT_SECRET.encode(), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(_b64url_decode(parts[1].decode()))
        if payload.get("exp", 0) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


if __name__ == "__main__":
    salt, h = hash_password("123456")
    print("salt:", salt)
    print("verify 正确:", verify_password("123456", salt, h))
    print("verify 错误:", verify_password("wrong", salt, h))
    tok = create_token({"uid": 1, "username": "test"})
    print("token:", tok)
    print("decode:", decode_token(tok))
