from lancedb.pydantic import LanceModel, Vector, Any, List
from pydantic import BaseModel, Field
from typing import Optional, Annotated, List
from langgraph.graph import MessagesState
from langchain_core.messages import AnyMessage


class RAGVector(LanceModel):
    file_name: Optional[str] = None
    title_path: Optional[List[str]] = None
    create_time: Optional[str] = None
    content: Optional[str] = None
    vector: Any


class Grade(BaseModel):
    """相关性检查的二元评分"""
    binary_score: str = Field(description="相关性评分 'yes' 或 'no'")


def unique_merge(current: List[str], new: List[str]) -> List[str]:
    return current + new


class RAGState(MessagesState):
    history: Annotated[List[AnyMessage | str], unique_merge]
    rewrite_num: Annotated[int, Field(default=-1)]
    documents: str
    regenerate_num: Annotated[int, Field(default=-1)]
