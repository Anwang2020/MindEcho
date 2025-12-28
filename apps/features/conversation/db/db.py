# -*- coding: utf-8 -*-
from datetime import datetime
from pathlib import Path
from typing import List, Optional, TypeVar, Union
import uuid

from sqlmodel import SQLModel, Session, and_, asc, create_engine, select
from .models import ChatInfo, ChatSession

from apps.logs.logs import get_logger

logger = get_logger(__name__)

# ------------------ config ------------------
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "app_data.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

T = TypeVar("T", bound=SQLModel)


class DatabaseManager:
    """Wrap SQLite access so chat sessions and messages are persisted reliably."""

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
        # create_all is idempotent; always run to guarantee tables exist
        logger.info("Ensuring database and tables exist at %s", DB_PATH)
        SQLModel.metadata.create_all(self.engine)

    def _get_session(self) -> Session:
        return Session(self.engine)

    @staticmethod
    def _generate_unique_uuid(session: Session) -> str:
        """Generate a UUID that is not already used in ChatSession."""
        while True:
            new_id = str(uuid.uuid4())
            existing = session.exec(select(ChatSession).where(ChatSession.id == new_id)).first()
            if not existing:
                return new_id

    def insert_chats(self, records: Union[T, List[T]], create_by: Optional[str] = "system") -> List[str]:
        """
        Insert one or many ChatSession/ChatInfo records. Duplicates in the same batch
        are skipped, and existing primary keys are updated instead of raising errors.
        Returns the list of processed record ids.
        """
        if not isinstance(records, list):
            records = [records]

        filtered_records: List[T] = []
        seen_keys = set()
        for record in records:
            if record is None:
                continue
            key = (type(record), getattr(record, "id", None) or id(record))
            if key in seen_keys:
                logger.info("Skip duplicated record in batch: %s", key)
                continue
            seen_keys.add(key)
            filtered_records.append(record)

        now = datetime.now()
        ids: List[str] = []

        with self._get_session() as session:
            try:
                for record in filtered_records:
                    if getattr(record, "create_time", None) is None:
                        record.create_time = now
                    if getattr(record, "create_by", None) is None:
                        record.create_by = create_by
                    if not getattr(record, "id", None):
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
                            logger.info("Update ChatSession: id=%s", existing.id)
                        else:
                            session.add(record)
                            logger.info("Insert ChatSession: id=%s", record.id)
                        ids.append(record.id)
                    else:
                        existing_chat = session.get(ChatInfo, record.id) if getattr(record, "id", None) else None
                        if existing_chat:
                            existing_chat.create_time = record.create_time
                            existing_chat.llm_name = getattr(record, "llm_name", existing_chat.llm_name)
                            existing_chat.question = getattr(record, "question", existing_chat.question)
                            existing_chat.analysis = getattr(record, "analysis", existing_chat.analysis)
                            existing_chat.answer = getattr(record, "answer", existing_chat.answer)
                            existing_chat.error = getattr(record, "error", existing_chat.error)
                            existing_chat.chat_session_id = getattr(record, "chat_session_id", existing_chat.chat_session_id)
                            existing_chat.create_by = record.create_by
                            logger.info("Update ChatInfo: id=%s", existing_chat.id)
                        else:
                            session.add(record)
                            logger.info("Insert ChatInfo: id=%s", record.id)
                        ids.append(record.id)

                session.commit()
            except Exception as e:
                session.rollback()
                logger.error("Insert Chat failed: %s", e)
                raise

        return ids

    def get_latest_messages(self, session_id: str, limit: int = 10) -> dict:
        """Return the latest messages and summary for a session in chronological order."""
        latest_messages_dict = {"chat_list": [], "summary": ""}
        with self._get_session() as session_db:
            chat_session_summary = (
                session_db.exec(
                    select(ChatSession.summary, ChatSession.create_time).filter(ChatSession.id == session_id)
                ).first()
            )
            if not chat_session_summary:
                return latest_messages_dict

            rows_chat_info = (
                session_db.exec(
                    select(ChatInfo)
                    .filter(and_(ChatInfo.chat_session_id == session_id, ChatInfo.create_time > chat_session_summary[1]))
                    .order_by(asc(ChatInfo.create_time))
                    .limit(limit)
                ).all()
            )
            chat_list = [{"system": _.answer} if _.answer else {"user": _.question} for _ in rows_chat_info]
            latest_messages_dict.update({"chat_list": chat_list, "summary": chat_session_summary[0]})
        return latest_messages_dict

    def get_session_info(self, chat_session_id: str):
        """Fetch brief info and all chat messages for a session."""
        try:
            with self._get_session() as session_db:
                chat_session = session_db.exec(select(ChatSession).filter(ChatSession.id == chat_session_id)).first()
                if not chat_session:
                    return None, []

                rows_chat_session = (
                    session_db.exec(
                        select(ChatInfo).filter(
                            and_(
                                ChatInfo.chat_session_id == chat_session_id,
                                ChatInfo.create_time > chat_session.create_time,
                            )
                        )
                    ).all()
                )
            return chat_session.brief, rows_chat_session
        except ValueError:
            return None, []
        except Exception as e:
            logger.error("Fetch session info failed: %s", e)
            return None, []


# ------------------ tests ------------------
if __name__ == "__main__":
    db_manager = DatabaseManager()
    aaa = db_manager.get_latest_messages(session_id="1234", limit=10)

    aaa = db_manager.get_session_info("1234")
    chat1 = ChatSession(id="1234", brief="single UUID test", summary="single UUID test111", llm_name="gpt")
    ids1 = db_manager.insert_chats(chat1)
    logger.info(f"single insert success, id={ids1}")
    chat_info1 = [ChatInfo(chat_session_id="1234", question=f"batch UUID test{i}", llm_name="gpt") for i in range(1, 20)]
    for chat_info1_ in chat_info1:
        ids3 = db_manager.insert_chats(chat_info1_)
        logger.info(f"chat_info single insert success, id={ids3}")