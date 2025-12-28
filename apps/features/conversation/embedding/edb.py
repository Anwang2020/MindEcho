from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Union
from uuid import uuid4
import lancedb
import os
import json

from .schema import ChatVector
from apps.logs.logs import get_logger

logger = get_logger("edb")
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "embedding_db"
db = lancedb.connect(DB_PATH)
REGISTRY_PATH = BASE_DIR / 'table_registry.json'
logger.info(f"edb路径: {DB_PATH}\ntable_registry路径: {REGISTRY_PATH}")


class EdbManager:
    def __init__(self, embed_model):
        self.db = lancedb.connect(DB_PATH)
        self.embed_model = embed_model
        self.table_registry = self._read_registry()

    @staticmethod
    def _read_registry():
        if not os.path.exists(REGISTRY_PATH):
            with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
                json.dump([{}], f, ensure_ascii=False, indent=4)
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid registry file: {e}")
                data = [{}]  # 如果文件
        return data[0]

    def _get_table(self, vector_dim: int):
        """获取对应维度的表"""
        vector_dim = str(vector_dim)
        if vector_dim in self.table_registry:
            return self.db.open_table(self.table_registry[vector_dim])

    def insert(self, data: Union[ChatVector, List[ChatVector]]):
        """单条或批量插入聊天记录"""
        if not isinstance(data, list):
            data = [data]

        for record in data:
            # 自动生成 id 和 create_time
            if not record.id:
                record.id = str(uuid4())
            if not record.create_time:
                record.create_time = datetime.utcnow().isoformat()

            vector = self.embed_model.embed_query(record.content)
            data = [
                {"id": record.id, "role": record.role, "content": record.content, "create_time": record.create_time,
                 "vector": vector, "session_id": record.session_id}]
            vector_dim = len(vector)
            try:
                table = self.db.open_table(self.table_registry[str(vector_dim)])
                table.add(data=data)
            except (ValueError, KeyError):
                table_name = f"chatmemory_{vector_dim}d"
                # 动态生成 LanceModel 子类
                self.table_registry[str(vector_dim)] = table_name
                with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
                    json.dump([self.table_registry], f, ensure_ascii=False, indent=4)
                self.db.create_table(table_name, data=data)

    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, str]]:
        """按语义相似度检索历史消息"""
        query_vec = self.embed_model.embed_query(query)
        table = self._get_table(len(query_vec))
        if table:
            res = table.search(query_vec).limit(top_k).to_pandas()
            similar_messages_dict = res.to_dict(orient="records")
            similar_messages_list = [{_["role"]: _["content"]} for _ in similar_messages_dict]
            return similar_messages_list
