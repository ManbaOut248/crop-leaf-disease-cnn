# -*- coding: utf-8 -*-
"""
PII 字段的静态加密（对应任务书 6.2「数据库存储加密」）。

为什么是「字段级」而不是「整库加密」
---------------------------------------------------------------
整库加密的标准做法是 SQLCipher，但本机实测 `sqlcipher3-binary` 在
**Python 3.12 上没有可用 wheel**（`No matching distribution found`），
自行编译需要 OpenSSL 开发链，不适合作为交付依赖。

故采用字段级加密：只对 `users.email` 这类 PII 列做 AES 加密后落库，
其余列（密码哈希本身已不可逆）保持明文，查询与索引不受影响。
整库加密的迁移路径写在 `docs/13_部署运维手册.md`。

密钥管理（这里有个与 JWT_SECRET 不同的硬约束）
---------------------------------------------------------------
JWT_SECRET 缺失时可以「每次启动随机生成」——代价只是重启后要重新登录。
**加密密钥不能这么做**：随机生成意味着重启后就解不开历史数据了。
所以 `DB_FIELD_KEY` 未设置时**不启用加密**并告警，而不是生成临时密钥。

生成密钥：
    py -3.12 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    set DB_FIELD_KEY=<上面输出的 44 字符>

存储格式
---------------------------------------------------------------
`enc:v1:<Fernet 密文>`。带前缀的目的是**兼容历史明文数据**：
读取时若无前缀则原样返回，这样加密封装可以在不改库的情况下逐步生效。
"""

import warnings

from cryptography.fernet import Fernet, InvalidToken

from src.config import DB_FIELD_KEY

# 版本前缀：将来换算法（v2）时可并存，读取端按前缀分派
PREFIX = "enc:v1:"


class _FieldCipher:
    def __init__(self):
        self.enabled = False
        self._fernet = None
        if not DB_FIELD_KEY:
            warnings.warn(
                "DB_FIELD_KEY 未设置，PII 字段加密**未启用**（users.email 将以明文存储）。"
                "生产部署请生成并固定设置该密钥，详见 docs/13_部署运维手册.md。",
                stacklevel=2,
            )
            return
        try:
            self._fernet = Fernet(DB_FIELD_KEY.encode())
            self.enabled = True
        except Exception as e:  # noqa: BLE001
            # 密钥格式错误不应阻断启动（否则配错一个字就整站起不来），
            # 但必须显式告警，避免"以为开了加密其实没开"。
            warnings.warn(
                f"DB_FIELD_KEY 格式非法（{type(e).__name__}），PII 字段加密未启用。"
                "应为 Fernet.generate_key() 生成的 44 字符 urlsafe base64。",
                stacklevel=2,
            )

    def encrypt(self, plain):
        """加密一个字段值。空值、已加密值原样返回（幂等）。"""
        if not self.enabled or plain is None or plain == "":
            return plain
        s = str(plain)
        if s.startswith(PREFIX):
            return s
        return PREFIX + self._fernet.encrypt(s.encode("utf-8")).decode("ascii")

    def decrypt(self, stored):
        """解密。无前缀视为历史明文，原样返回；密钥变更导致无法解密时返回 None。"""
        if stored is None or stored == "":
            return stored
        s = str(stored)
        if not s.startswith(PREFIX):
            return s
        if not self.enabled:
            return None
        try:
            return self._fernet.decrypt(s[len(PREFIX):].encode("ascii")).decode("utf-8")
        except (InvalidToken, ValueError):
            # 换过密钥 / 数据被篡改：不能把密文当明文返回，也不能让整个请求 500
            return None

    def is_ciphertext(self, value):
        return isinstance(value, str) and value.startswith(PREFIX)


cipher = _FieldCipher()
