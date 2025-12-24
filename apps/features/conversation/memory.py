from db.models import ChatInfo, ChatSession
from db.db import DatabaseManager
from embedding.schema import ChatVector
from embedding.edb import EdbManager
from apps.logs.logs import get_logger
from apps.common import custom_bge

logger = get_logger("main")


class Memory:
    def __init__(self, db_manager, edb_manager, llm_name="gpt", emb_mdl_name="BAAI/bge-large-zh-v1.5"):
        self.db_manager = db_manager
        self.edb_manager = edb_manager
        self.llm_name = llm_name
        self.emb_mdl_name = emb_mdl_name

    def save_memory(self, message: dict, create_time, chat_session_id: str, session_brief: str, chat_type: str,
                    summary: str = None, new_session: bool = False):

        # 往ChatSession表里存
        if summary or new_session:
            chat_session_memory = ChatSession(id=chat_session_id, create_time=create_time, llm_name=self.llm_name,
                                              brief=session_brief, summary=summary, chat_type=chat_type)
            self.db_manager.insert_chats(chat_session_memory)
            logger.info(f"写入ChatSession表成功")
        # 根据信息类型写入ChatInfo
        chat_info = [ChatInfo(chat_session_id=chat_session_id, create_time=create_time, question=message["user"],
                              llm_name=self.llm_name),
                     ChatInfo(chat_session_id=chat_session_id, create_time=create_time, answer=message["system"],
                              llm_name=self.llm_name)]
        _ = self.db_manager.insert_chats(chat_info)
        # 写入信息的embedding
        chat_vector_info = [ChatVector(role="user", content=message["user"], session_id=chat_session_id,
                                       embedding_model_name=self.emb_mdl_name),
                            ChatVector(role="system", content=message["system"], session_id=chat_session_id,
                                       embedding_model_name=self.emb_mdl_name)]
        self.edb_manager.insert(chat_vector_info)

    def load_memory(self, session_id, query) -> dict:
        """
        1.加载最近的10条信息
        2.加载语义最相近的5条信息
        3.加载summary
        """
        try:
            latest_messages = self.db_manager.get_latest_messages(session_id=session_id, limit=10)
            similar_messages = self.edb_manager.search_similar(query=query, top_k=5)
            latest_messages["similar_messages"] = similar_messages
        except Exception as e:
            logger.info(f"读取聊天记录失败:{e}")
            latest_messages = {"chat_list": '', "summary": '', "similar_messages": ''}
        return latest_messages


edb = EdbManager(custom_bge)
db = DatabaseManager()

memory = Memory(db, edb)
# if __name__ == '__main__':
#     from apps.embedding.edb import EdbManager
#     from apps.embedding.embedding_model import Embedding
#     from apps.db.db import DatabaseManager
#     import os
#     from dotenv import load_dotenv
#
#     load_dotenv()
#     url = os.getenv("EMB_API_URL")
#     key = os.getenv("EMB_API_KEY")
#     name = os.getenv("EMB_API_NAME")
#     embed_model = Embedding({"url": url,
#                              "api_key": key,
#                              "api_name": name})
#     edb = EdbManager(embed_model)
#     db = DatabaseManager()
#     memory = Memory(db, edb)
#     memory.load_memory()
#     a = 1
