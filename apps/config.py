from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()


def app_cross(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 允许所有域名访问
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


class LLMConfig:
    """
    LLMConfig: LLM配置类
    """
    model_name: str = os.getenv("MODEL")
    api_key: str = os.getenv("LLM_API_KEY")
    base_url: str = os.getenv("LLM_BASE_URL")
