from langchain_core.tools import tool
from typing import Annotated, Callable
from pathlib import Path
import json
import numpy as np

from .service import rag_workflow

REGISTRY_PATH = Path(__file__).parent / 'table_registry.json'


def create_tools(name: str, description: str, table_id: str) -> Callable:
    @tool(name, description=description)
    def search_knowledge_tool(query: Annotated[str, "搜索内容"], limit: Annotated[int, "搜索结果数量"] = 3):
        result = rag_workflow.search(query, limit, table_id)
        return result

    return search_knowledge_tool


def load_tools() -> list[Callable]:
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    tools = []
    for idx, (description, table_id) in enumerate(data[0].items()):
        search_tool = create_tools(f"search_knowledge_{idx}", description, table_id)
        tools.append(search_tool)
    return tools
