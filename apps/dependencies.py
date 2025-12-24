from dotenv import load_dotenv
import os

from apps.chat.pipeline import ChatPipeline
from apps.chat.memory import Memory
from apps.embedding.edb import EdbManager
from apps.db.db import DatabaseManager
from apps.embedding.embedding_model import Embedding

load_dotenv()
url = os.getenv("EMB_API_URL")
key = os.getenv("EMB_API_KEY")
name = os.getenv("EMB_API_NAME")
llm_name = os.getenv("MODEL")
_embed_model = Embedding({"url": url,
                          "api_key": key,
                          "api_name": name})

_edb_manager = EdbManager(embed_model=_embed_model)
_db_manager = DatabaseManager()
_memory = Memory(edb_manager=_edb_manager,
                 db_manager=_db_manager,
                 llm_name=llm_name,
                 emb_mdl_name=name)
_chat_pipeline = ChatPipeline(memory=_memory)


# --- FastAPI Depends ---
def get_chat_pipeline() -> ChatPipeline:
    return _chat_pipeline  # 返回同一个实例
