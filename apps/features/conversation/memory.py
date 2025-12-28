# -*- coding: utf-8 -*-
from .db.models import ChatInfo, ChatSession
from .db.db import DatabaseManager
from .embedding.schema import ChatVector
from .embedding.edb import EdbManager
from apps.logs.logs import get_logger
from apps.common import custom_bge

logger = get_logger(__name__)


class Memory:
    def __init__(self, db_manager, edb_manager, llm_name="gpt", emb_mdl_name="BAAI/bge-large-zh-v1.5"):
        self.db_manager = db_manager
        self.edb_manager = edb_manager
        self.llm_name = llm_name
        self.emb_mdl_name = emb_mdl_name

    def save_memory(self, message: dict, create_time, chat_session_id: str, session_brief: str, chat_type: str,
                    summary: str = None, new_session: bool = False):
        """
        将会话写入数据库与向量库；批量写入并去重，避免同一列表重复写导致报错。
        """
        records = []
        if summary is not None or new_session:
            records.append(
                ChatSession(
                    id=chat_session_id,
                    create_time=create_time,
                    llm_name=self.llm_name,
                    brief=session_brief,
                    summary=summary,
                    chat_type=chat_type,
                )
            )

        chat_records = [
            ChatInfo(
                chat_session_id=chat_session_id,
                create_time=create_time,
                question=message.get("user", ""),
                llm_name=self.llm_name,
            ),
            ChatInfo(
                chat_session_id=chat_session_id,
                create_time=create_time,
                answer=message.get("system", ""),
                llm_name=self.llm_name,
            ),
        ]
        records.extend(chat_records)

        try:
            ids = self.db_manager.insert_chats(records)
            logger.info("写入ChatSession/ChatInfo成功，session=%s, count=%s, ids=%s", chat_session_id, len(records), ids)
        except Exception as exc:
            logger.exception("保存ChatSession/ChatInfo失败: session=%s, error=%s", chat_session_id, exc)
            return

        chat_vector_info = [
            ChatVector(
                role="user",
                content=message.get("user", ""),
                session_id=chat_session_id,
                embedding_model_name=self.emb_mdl_name,
            ),
            ChatVector(
                role="system",
                content=message.get("system", ""),
                session_id=chat_session_id,
                embedding_model_name=self.emb_mdl_name,
            ),
        ]
        try:
            self.edb_manager.insert(chat_vector_info)
        except Exception as exc:
            logger.exception("写入embedding失败: session=%s, error=%s", chat_session_id, exc)

    def load_memory(self, session_id, query) -> dict:
        """
        1.加载最近的10条信息
        2.加载语义最接近的5条信息
        3.加载summary
        """
        try:
            latest_messages = self.db_manager.get_latest_messages(session_id=session_id, limit=10)
            similar_messages = self.edb_manager.search_similar(query=query, top_k=5) or []
            latest_messages["similar_messages"] = similar_messages
        except Exception as e:
            logger.info("读取聊天记录失败: %s", e)
            latest_messages = {"chat_list": [], "summary": "", "similar_messages": []}
        return latest_messages


edb = EdbManager(custom_bge)
db = DatabaseManager()

memory = Memory(db, edb)
