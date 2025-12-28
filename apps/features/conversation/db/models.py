from datetime import datetime
from typing import Optional

from sqlalchemy import Column, String, Text
from sqlmodel import Field, SQLModel


class ChatSession(SQLModel, table=True):
    """Chat session summary."""

    __tablename__ = "chatsession"
    __table_args__ = {"extend_existing": True}

    id: str = Field(
        default=None,
        sa_column=Column(String(36), primary_key=True),
        description="Primary key UUID",
    )
    create_time: Optional[datetime] = Field(default=None, description="Created time")
    chat_type: str = Field(default="chat", max_length=20, description="chat | datasource")
    llm_name: str = Field(default="sqlite", max_length=64, description="Model name")
    create_by: Optional[str] = Field(default=None, description="Creator user id")
    brief: Optional[str] = Field(default=None, max_length=64, description="Session title")
    summary: Optional[str] = Field(default=None, sa_column=Column(Text), description="Session summary")


class ChatInfo(SQLModel, table=True):
    """Chat message details."""

    __tablename__ = "chatinfo"
    __table_args__ = {"extend_existing": True}

    id: str = Field(
        default=None,
        sa_column=Column(String(36), primary_key=True),
        description="Primary key UUID",
    )
    chat_session_id: str = Field(sa_column=Column(String(36)), description="FK ChatSession.id")
    create_time: Optional[datetime] = Field(default=None, description="Created time")
    llm_name: Optional[str] = Field(default=None, sa_column=Column(String(36)))
    create_by: Optional[str] = Field(default=None, description="Creator user id")
    question: Optional[str] = Field(default=None, sa_column=Column(Text))
    analysis: Optional[str] = Field(default=None, sa_column=Column(Text))
    answer: Optional[str] = Field(default=None, sa_column=Column(Text))
    error: Optional[str] = Field(default=None, sa_column=Column(Text))
