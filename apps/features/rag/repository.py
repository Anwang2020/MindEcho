from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Union
from uuid import uuid4
import lancedb
import os
import json
import logging
import numpy as np

from .schema import RAGVector

logger = logging.getLogger('server.rag.repository')
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "embedding_db"
db = lancedb.connect(DB_PATH)
REGISTRY_PATH = BASE_DIR / 'table_registry.json'


class LancedbManager:
    def __init__(self, embed_model):
        self.db = lancedb.connect(DB_PATH)
        self.embed_model = embed_model

    @staticmethod
    def registry_table(description: str, vector_dim: int):
        table_id = str(uuid4())
        if not os.path.exists(REGISTRY_PATH):
            with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
                json.dump([{f"{description}_{vector_dim}": table_id}], f, ensure_ascii=False, indent=4)
        else:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            data[0][f"{description}_{vector_dim}"] = table_id
            with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        return table_id

    @staticmethod
    def _get_table():
        """获取表"""
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data[0]

    def insert(self, table_id: str, data: list):
        """单条或批量插入聊天记录"""
        for record in data:
            # 自动create_time
            create_time = datetime.utcnow().isoformat()

            vector = self.embed_model.embed_query(record["content"])
            record_data = {"file_name": record["file_name"],
                           "title_path": record["title_path"],
                           "create_time": create_time,
                           "vector": vector,
                           "content": record["content"]}
            RAGVector(**record_data)
            try:
                table = self.db.open_table(table_id)
                table.add(data=[record_data])
            except (ValueError, KeyError):
                self.db.create_table(table_id, data=[record_data])

    def search_similar(self, query: str, top_k: int = 5, table_id="") -> List[Dict[str, str]]:
        query_vec = self.embed_model.embed_query(query)
        if not table_id:
            table_id = self.top_similar_above_threshold(query_vec)
        if table_id:
            table = self.db.open_table(table_id)
            res = table.search(query_vec).limit(top_k).to_pandas()
            similar_messages_dict = res.to_dict(orient="records")
            return similar_messages_dict

    def top_similar_above_threshold(self, query_vec, threshold=0.5):
        data = self._get_table()
        table_descriptions = list(data.keys())
        table_names = list(data.values())
        doc_vecs = self.embed_model.embed_documents(table_descriptions)
        q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        docs = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)
        sims = np.dot(docs, q)

        # 筛选大于 threshold
        mask = sims >= threshold
        if not np.any(mask):
            return None  # 没有符合条件的

        filtered_sims = sims[mask]

        # 找到最大值
        idx = np.argmax(filtered_sims)
        return table_names[idx]
