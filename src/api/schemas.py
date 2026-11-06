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


class UpdateRecordRequest(BaseModel):
    """修改识别记录（PATCH /records/{rid}）。

    三个字段都是三态：不传=不改 / null=清空 / 传值=更新。
    因此**不能**用 `Field(None)` 的默认值来判断「是否需要改」——
    调用方传 null 与不传，在模型里长得一样，必须靠 `model_fields_set` 区分。
    """

    custom_name: Optional[str] = Field(
        None, max_length=40, description="自定义命名，null 表示取消命名")
    verdict: Optional[str] = Field(
        None,
        description="人工核对结论：correct（识别正确）/ wrong（识别错误）/ null（取消核对）")
    ground_truth_class_id: Optional[int] = Field(
        None, ge=0, le=37,
        description=("用户已知的实际病害类别 id（0~37）；一般只在 verdict=wrong 时使用。"
                     "不传则由后端按 verdict 自动处理：判错保留、判对或取消核对则清空"))
