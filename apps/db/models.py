from sqlmodel import SQLModel, Field
from sqlalchemy import Column, String, Boolean, Text
from typing import Optional
from datetime import datetime


class ChatSession(SQLModel, table=True):
    id: str = Field(
        default=None,
        sa_column=Column(String(36), primary_key=True),  # 直接在 Column 中声明主键
        description="主键 UUID"
    )
    create_time: Optional[datetime] = Field(default=None, description="创建时间")
    chat_type: str = Field(default="chat", max_length=20, description="聊天类型: chat 或 datasource")
    llm_name: str = Field(default="sqlite", max_length=64, description="使用的大模型名称")
    create_by: Optional[str] = Field(default=None, description="创建者用户ID")
    brief: Optional[str] = Field(default=None, max_length=64, description="聊天记录标题")
    summary: Optional[str] = Field(default=None, sa_column=Column(Text), description="聊天记录摘要")


class ChatInfo(SQLModel, table=True):
    id: str = Field(
        default=None,
        sa_column=Column(String(36), primary_key=True),  # 直接在 Column 中声明主键
        description="主键 UUID"
    )
    chat_session_id: str = Field(sa_column=Column(String(36)), description="关联 Chat ID")
    create_time: Optional[datetime] = Field(default=None, description="创建时间")
    llm_name: Optional[str] = Field(default=None, sa_column=Column(String(36)))
    create_by: Optional[str] = Field(default=None, description="创建者用户ID")
    question: Optional[str] = Field(default=None, sa_column=Column(Text))
    analysis: Optional[str] = Field(default=None, sa_column=Column(Text))
    answer: Optional[str] = Field(default=None, sa_column=Column(Text))
    error: Optional[str] = Field(default=None, sa_column=Column(Text))
