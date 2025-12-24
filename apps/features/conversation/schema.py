from typing import Annotated, List, Optional

from pydantic import BaseModel, Field
from langgraph.graph import MessagesState, add_messages
from langchain_core.messages import AnyMessage


def unique_merge(current: List[str], new: List[str]) -> List[str]:
    return current + new


class ChatState(MessagesState):
    chat_type: Annotated[str, Field(default='chat')]
    user_input: Annotated[List[AnyMessage | str | dict], Field(default=[])]
    recent_history: Annotated[List[AnyMessage | str | dict], Field(default=[]), unique_merge]
    similar_chats: Annotated[List[str], Field(default=[])]
    session_summary: Annotated[str, Field(default='')]
    rewrite_num: Annotated[int, Field(default=-1)]
    regenerate_num: Annotated[int, Field(default=-1)]


class Validate(BaseModel):
    """相关性检查的二元评分"""
    binary_score: str = Field(description="相关性评分 'yes' 或 'no'")

