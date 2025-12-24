import json
import requests
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
    def __init__(self, model: str):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        emds = []
        for text in texts:
            emb = self.embed_query(text)
            emds.append(emb)
        return emds

    def embed_query(self, text: str) -> List[float]:
        payload = {
            "model": self.model,
            "input": text
        }
        headers = {
            "Authorization": f"Bearer sk-urjzseolxpsczjynkyxrztmpdhdxklrdvnszywwforljglqx",
            "Content-Type": "application/json"
        }
        response = requests.post("https://api.siliconflow.cn/v1/embeddings",
                                 json=payload,
                                 headers=headers)
        data = json.loads(response.text)
        return data["data"][0]["embedding"]


class CustomBgeEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer(f'{COMMON_DIR}/models_dir/bge-base-zh-v1.5')

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        p_embeddings = list(self.model.encode(texts, normalize_embeddings=True))
        return p_embeddings

    def embed_query(self, text: str) -> List[float]:
        p_embeddings = self.embed_documents([text])
        return p_embeddings[0]
