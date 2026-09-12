import json
import requests
import os
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from pathlib import Path

from apps.config import LLMConfig

COMMON_DIR = Path(__file__).parent
# 先实现功能 再根据要求进行优化
llm = ChatOpenAI(model_name=LLMConfig.model_name,
                 api_key=LLMConfig.api_key,
                 base_url=LLMConfig.base_url,
                 temperature=0)


class BgeEmbeddings(Embeddings):
    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY")
        self.base_url = base_url or os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1/embeddings")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        emds = []
        for text in texts:
            emb = self.embed_query(text)
            emds.append(emb)
        return emds

    def embed_query(self, text: str) -> List[float]:
        if not self.api_key:
            raise RuntimeError("EMBEDDING_API_KEY is required when using BgeEmbeddings")
        payload = {
            "model": self.model,
            "input": text
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        response = requests.post(self.base_url,
                                 json=payload,
                                 headers=headers,
                                 timeout=30)
        response.raise_for_status()
        data = json.loads(response.text)
        return data["data"][0]["embedding"]


class CustomBgeEmbeddings(Embeddings):
    def __init__(self, model_path: str | None = None):
        self.model_path = Path(model_path or os.getenv(
            "BGE_MODEL_PATH", str(COMMON_DIR / "models_dir" / "bge-base-zh-v1.5")
        ))
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if not self.model_path.exists():
                raise RuntimeError(
                    "Local BGE model is missing. Set BGE_MODEL_PATH or place "
                    "bge-base-zh-v1.5 under apps/common/models_dir/."
                )
            self._model = SentenceTransformer(str(self.model_path))
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        p_embeddings = list(self.model.encode(texts, normalize_embeddings=True))
        return p_embeddings

    def embed_query(self, text: str) -> List[float]:
        p_embeddings = self.embed_documents([text])
        return p_embeddings[0]
