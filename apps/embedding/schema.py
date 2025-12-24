from lancedb.pydantic import LanceModel, Vector
from pydantic import BaseModel
from typing import Optional


class ChatVector(LanceModel):
    id: Optional[str] = None
    role: str
    content: str
    create_time: Optional[str] = None
    session_id: str
    embedding_model_name: Optional[str] = None


class EmbCheck(BaseModel):
    """
    检查输入的url参数
    """
    url: str
    api_key: str
    api_name: str
