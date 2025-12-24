from pathlib import Path
from typing import List, Optional, Union, TypeVar
from datetime import datetime
from sqlmodel import SQLModel, Session, create_engine, select, desc, and_, asc

import uuid

from apps.logs.logs import get_logger
from .models import ChatSession, ChatInfo

logger = get_logger("db")
# ------------------ 配置 ------------------
BASE_DIR = Path(__file__).parent
logger.info(f"数据库路径：{BASE_DIR}")
DB_PATH = BASE_DIR / "app_data.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

T = TypeVar("T", bound=SQLModel)


# 如果文件不存在自动创建空数据库文件
class DatabaseManager:
    def __init__(self):
        self._create_engine()
        self._init_db()

    def _create_engine(self):
        self.engine = create_engine(
            f"sqlite:///{DB_PATH}",
            echo=False,
            connect_args={"check_same_thread": False},
        )

    def _init_db(self):
        if not DB_PATH.exists():  # 检测数据库文件是否存在
            logger.info("数据库不存在，创建表...")
            SQLModel.metadata.create_all(self.engine)
            logger.info("数据库创建完成")
        else:
            logger.info("数据库已存在，直接使用")

    def _get_session(self) -> Session:
        return Session(self.engine)

    @staticmethod
    def _generate_unique_uuid(session: Session) -> str:
        while True:
            new_id = str(uuid.uuid4())
            existing = session.exec(select(ChatSession).where(ChatSession.id == new_id)).first()
            if not existing:
                return new_id

    def insert_chats(self, records: Union[T, List[T]], create_by: Optional[str] = "栗子") -> List[str]:
        """
        支持单条或批量插入
        返回主键 UUID 列表
        """
        if not isinstance(records, list):
            records = [records]

        now = datetime.now()
        ids = []

        with self._get_session() as session:
            try:
                for record in records:
                    if record.create_time is None:
                        record.create_time = now
                    if record.create_by is None:
                        record.create_by = create_by
                    if not record.id:
                        record.id = self._generate_unique_uuid(session)

                    if isinstance(record, ChatSession):
                        existing = session.get(ChatSession, record.id)
                        if existing:
                            existing.create_time = record.create_time
                            existing.llm_name = record.llm_name
                            existing.summary = record.summary
                            existing.chat_type = record.chat_type
                            existing.brief = record.brief
                            existing.create_by = record.create_by
                            logger.info(f"更新 ChatSession: id={existing.id}")
                        else:
                            session.add(record)
                            logger.info(f"插入新 ChatSession: id={record.id}")
                    else:
                        session.add(record)
                        ids.append(record.id)
                        logger.info(f"插入 Chat: id={record.id}")

                session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f"插入 Chat 失败, error={e}")
                raise

        return ids

    def get_latest_messages(self, session_id: str, limit: int = 10) -> dict:
        """
        从 DB 按时间倒序取最近 limit 条（返回按时间正序，方便阅读）
        """
        latest_messages_dict = {}
        with self._get_session() as session_db:
            chat_session_summary = (session_db.exec(select(ChatSession.summary, ChatSession.create_time)
                                                    .filter(ChatSession.id == session_id)
                                                    ).first())
            rows_chat_info = (session_db.exec(select(ChatInfo)
                                              .filter(and_(ChatInfo.chat_session_id == session_id,
                                                           ChatInfo.create_time > chat_session_summary[1]))  # 使用正确的字段名
                                              .order_by(asc(ChatInfo.create_time))).all())
            chat_list = [
                {"system": _.answer} if _.answer else {"user": _.question}
                for _ in rows_chat_info
            ]
            latest_messages_dict.update({"chat_list": chat_list, "summary": chat_session_summary[0]})
        return latest_messages_dict

    def get_session_info(self, chat_session_id: str):
        """
        获取会话的摘要
        """
        try:
            with self._get_session() as session_db:
                chat_session = (
                    session_db.exec(
                        select(ChatSession)
                        .filter(ChatSession.id == chat_session_id)
                    ).first()
                )

                rows_chat_session = (
                    session_db.exec(
                        select(ChatInfo)
                        .filter(and_(ChatInfo.chat_session_id == chat_session_id,
                                     ChatInfo.create_time > chat_session.create_time))
                    ).all()
                )
            return chat_session.brief, rows_chat_session
        except ValueError:
            pass
        except Exception as e:
            logger.error(f"获取会话信息失败: {e}")
            return None, []


# ------------------ 测试 ------------------
if __name__ == "__main__":
    db_manager = DatabaseManager()
    aaa = db_manager.get_latest_messages(session_id="1234", limit=10)

    aaa = db_manager.get_session_info("1234")
    chat1 = ChatSession(id="1234", brief="单条UUID测试", summary="单条UUID测试111", llm_name="gpt")
    ids1 = db_manager.insert_chats(chat1)
    print(f"单条插入成功，id={ids1}")
    # chats = [ChatSession(brief=f"批量UUID测试{i}", chat_type="chat", engine_type="gpt") for i in
    #          range(1, 6)]
    # ids2 = db_manager.insert_chats(chats, create_by=1)
    # print(f"批量插入成功，id列表={ids2}")
    chat_info1 = [ChatInfo(chat_session_id="1234", question=f"批量UUID测试{i}", llm_name="gpt") for i in
                  range(1, 20)]
    for chat_info1_ in chat_info1:
        ids3 = db_manager.insert_chats(chat_info1_)
        print(f"chat_info单条插入成功，id={ids3}")
