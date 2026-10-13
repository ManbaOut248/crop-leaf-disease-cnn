# -*- coding: utf-8 -*-
"""请求体 Pydantic 模型（FastAPI 入参校验，对齐 docs/04_API接口文档.md）。"""

from typing import Optional
from pydantic import BaseModel, Field


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32, description="用户名")
    password: str = Field(..., min_length=6, max_length=64, description="密码")
    email: Optional[str] = Field(None, description="邮箱（可选）")
    nickname: Optional[str] = Field(None, description="昵称（可选，默认同用户名）")


class LoginRequest(BaseModel):
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")


class UpdateMeRequest(BaseModel):
    nickname: Optional[str] = Field(None, description="新昵称")
    old_password: Optional[str] = Field(None, description="旧密码（改密码时必填）")
    new_password: Optional[str] = Field(None, min_length=6, max_length=64, description="新密码")
