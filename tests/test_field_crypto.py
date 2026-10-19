# -*- coding: utf-8 -*-
"""PII 字段加密的单测（任务书 6.2「数据库存储加密」）。

覆盖四件事：
  1. 加解密往返 + 落库格式（带 `enc:v1:` 前缀）
  2. 历史明文兼容：无前缀原样返回（否则加密封装一上线老用户就登不进）
  3. 幂等：对已加密值重复加密不会套娃
  4. 与 db 层集成：create_user 落库是密文，读出来是明文；启动迁移能把明文补加密
"""

import sqlite3

import pytest
from cryptography.fernet import Fernet

import src.api.crypto as crypto_mod
from src.api import db


@pytest.fixture()
def keyed(monkeypatch):
    """构造一个已启用的 cipher，并注入到 db 模块。"""
    key = Fernet.generate_key().decode()
    monkeypatch.setattr(crypto_mod, "DB_FIELD_KEY", key)
    c = crypto_mod._FieldCipher()
    assert c.enabled, "测试用密钥应能正常启用加密"
    monkeypatch.setattr(db, "cipher", c)
    return c


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "crypto_test.db")
    db.init_db()
    return tmp_path / "crypto_test.db"


def test_roundtrip_and_prefix(keyed):
    c = keyed
    plain = "farmer@example.com"
    enc = c.encrypt(plain)

    assert enc.startswith(crypto_mod.PREFIX)      # 落库形态可判别
    assert plain not in enc                       # 明文不得出现在密文里
    assert c.decrypt(enc) == plain


def test_legacy_plaintext_passthrough(keyed):
    """没有前缀的值视为历史明文，原样返回 —— 保证加密封装可平滑上线。"""
    assert keyed.decrypt("old@example.com") == "old@example.com"


def test_encrypt_is_idempotent(keyed):
    once = keyed.encrypt("a@b.com")
    twice = keyed.encrypt(once)
    assert once == twice, "已加密值不应被二次加密（否则套娃且无法还原）"


def test_disabled_cipher_passthrough(monkeypatch):
    """未配置密钥时不加密，且读取密文返回 None（不能把密文当明文吐给上游）。"""
    monkeypatch.setattr(crypto_mod, "DB_FIELD_KEY", "")
    c = crypto_mod._FieldCipher()
    assert not c.enabled
    assert c.encrypt("x@y.com") == "x@y.com"
    assert c.decrypt("x@y.com") == "x@y.com"
    # 别人加密过的数据在无密钥时不可读
    assert c.decrypt(crypto_mod.PREFIX + "garbage") is None


def test_db_stores_ciphertext_but_returns_plaintext(keyed, temp_db):
    uid = db.create_user("u_enc", "hash", "salt", email="secret@example.com")
    # 直接读原始库：必须是密文
    conn = sqlite3.connect(str(temp_db))
    raw = conn.execute("SELECT email FROM users WHERE id=?", (uid,)).fetchone()[0]
    conn.close()
    assert raw.startswith(crypto_mod.PREFIX)
    assert "secret@example.com" not in raw

    # 走 ORM 层：必须还原成明文
    assert db.get_user_by_id(uid)["email"] == "secret@example.com"
    assert db.get_user_by_username("u_enc")["email"] == "secret@example.com"


def test_startup_migration_encrypts_existing_plaintext(keyed, temp_db):
    """模拟「上线前已有明文数据」，再跑一次 init_db 应就地加密且不重复。"""
    conn = sqlite3.connect(str(temp_db))
    conn.execute(
        "INSERT INTO users (username, password_hash, salt, email) VALUES (?,?,?,?)",
        ("legacy", "h", "s", "legacy@example.com"),
    )
    conn.commit()
    conn.close()

    db.init_db()          # 触发 _migrate_encrypt_pii
    conn = sqlite3.connect(str(temp_db))
    raw = conn.execute("SELECT email FROM users WHERE username='legacy'").fetchone()[0]
    conn.close()
    assert raw.startswith(crypto_mod.PREFIX)

    db.init_db()          # 再跑一次：幂等，不应重复加密
    conn = sqlite3.connect(str(temp_db))
    raw2 = conn.execute("SELECT email FROM users WHERE username='legacy'").fetchone()[0]
    conn.close()
    assert raw2.count(crypto_mod.PREFIX) == 1
