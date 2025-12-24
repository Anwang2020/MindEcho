from typing import Dict
import requests
import json

from apps.embedding.schema import EmbCheck
from apps.logs.logs import get_logger

logger = get_logger("edb")


class Embedding:
    def __init__(self, emb_model: Dict | None = None):
        if not emb_model:
            """本地搭建"""
            pass
        else:
            try:
                emb_obj = EmbCheck(**emb_model)
                self.emb_model = emb_obj
                self.emb_model_name = emb_obj.api_name
            except Exception as e:
                self.emb_model = None
                logger.error(f"Invalid embedding model: {e}\n---{emb_model}")

    def embed(self, text: str) -> list:
        try:
            if not self.emb_model:
                return []
            else:
                payload = {
                    "model": self.emb_model.api_name,
                    "input": text
                }
                headers = {
                    "Authorization": f"Bearer {self.emb_model.api_key}",
                    "Content-Type": "application/json"
                }
                response = requests.post(self.emb_model.url, json=payload, headers=headers)
                logger.info(f"embedding模型返回值：{response}")
                data = json.loads(response.text)
                return data["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"embedding模型调用失败：{e}")
