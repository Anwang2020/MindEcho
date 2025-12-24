from typing import Annotated
from langchain.tools import tool
import time

from apps.features.rag import search_func, rag_agent


@tool
def get_current_time(city: Annotated[str, "城市名称"] = "上海"):
    """涉及当前系统时间或当前日期相关问题时必须调用此工具获取当前时间"""
    timestamp = time.time()
    local_time = time.localtime(timestamp)
    return time.strftime("%Y-%m-%d %H:%M:%S", local_time)


@tool('web_search', description='需要进行网络搜索时调用此工具')
def search(query: Annotated[str, "搜索内容"], limit: Annotated[int, "搜索结果数量"] = 3):
    return search_func(query, limit)
