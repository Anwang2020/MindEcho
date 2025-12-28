from typing import Annotated
from langgraph.types import Command
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.prebuilt import InjectedState
from pathlib import Path
import json

from apps.logs.logs import get_logger
from .schema import ChatState

logger = get_logger(__name__)

TABLE_REGISTRY_PATH = Path(__file__).parents[1] / 'rag/table_registry.json'


def process_history(state):
    history_dict = {}
    for _ in ['recent_history', 'similar_chats', 'session_summary']:
        __ = state.get(_, '')
        match _:
            case 'recent_history':
                recent_history = []
                for temp in __:
                    msg = None
                    if isinstance(temp, BaseMessage):
                        recent_history.append(temp)
                        continue
                    if isinstance(temp, str):
                        if temp.startswith("system:"):
                            msg = AIMessage(content=temp[len("system:"):])
                        elif temp.startswith("user:"):
                            msg = HumanMessage(content=temp[len("user:"):])
                    elif isinstance(temp, dict):
                        if temp.get('system'):
                            msg = AIMessage(content=temp['system'])
                        elif temp.get('user'):
                            msg = HumanMessage(content=temp['user'])
                    if msg and msg.content is not None:
                        recent_history.append(msg)
                    else:
                        # 忽略无有效内容的历史项，避免 None 触发校验错误
                        continue

                history_dict[_] = recent_history

            case 'similar_chats':
                history_dict[_] = ', '.join([str(temp) for temp in __]) if __ else ''
            case 'session_summary':
                history_dict[_] = "【" + __.replace("\n", " ") + "】" if __ else ''
    return history_dict


def create_handoff_tool(*, agent_name: str, description: str | None = None):
    """
    创建一个用于将当前会话转接到指定代理的工具函数。

    该函数返回一个装饰器包装的工具函数，当调用时会生成一个工具消息并返回转接命令，
    指示流程控制器将控制权转移给指定的代理。

    参数:
        agent_name (str): 目标代理的名称，用于标识要转接的代理
        description (str | None): 工具的描述信息，如果未提供则使用默认描述

    返回:
        handoff_tool: 一个装饰器包装的工具函数，用于执行转接操作
    """
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
            state: Annotated[ChatState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """
        执行实际的转接操作。

        创建一个工具消息表明转接成功，并返回一个命令对象指示流程控制器
        将控制权转移给指定代理，同时更新会话状态。

        参数:
            state (MessagesState): 当前会话状态，包含消息历史等信息
            tool_call_id (str): 工具调用的唯一标识符

        返回:
            Command: 包含转接指令和状态更新的命令对象
        """
        # 构造工具消息，记录转接操作的成功执行
        logger.info("转接成功: %s", name)
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": name,
            "tool_call_id": tool_call_id,
        }
        # 返回转接命令，指定目标代理和更新后的状态
        return Command(
            goto=agent_name,
            update={**state, "messages": state["messages"] + [tool_message]},
            graph=Command.PARENT,
        )

    return handoff_tool


def load_save_knowledge():
    with open(TABLE_REGISTRY_PATH, 'r', encoding='utf-8') as f:
        table_registry = json.load(f)[0]
    return ' | '.join('【' + _.split('_')[:-1][0] + '】' for _ in table_registry) if table_registry else '当前知识库内容为空'


if __name__ == '__main__':
    logger.info(load_save_knowledge())
