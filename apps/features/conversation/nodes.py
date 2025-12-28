# -*- coding: utf-8 -*-
import threading
import asyncio
from datetime import datetime
from langgraph.prebuilt import create_react_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, AIMessageChunk
from langgraph.config import get_stream_writer
from pathlib import Path

from apps.logs.logs import get_logger
from apps.common import llm, load_prompt
from .utils import process_history, create_handoff_tool, load_save_knowledge
from .tools import search, get_current_time, rag_agent
from .memory import memory

APPS_DIR = Path(__file__).parents[2]
logger = get_logger(__name__)


async def get_history_agent(state):
    user_input = state["messages"][-1].content
    user_input_dict = eval(user_input)
    user_input_content = user_input_dict.get("content")
    session_id = user_input_dict.get("session_id")
    chat_type = user_input_dict.get("type")
    latest_messages = memory.load_memory(session_id, user_input_content)
    return {
        'chat_type': chat_type,
        'user_input': [HumanMessage(content=user_input_content)],
        'recent_history': latest_messages['chat_list'],
        'similar_chats': latest_messages['similar_messages'],
        'session_summary': latest_messages['summary']
    }


async def text2sql_agent(state):
    pass
    return {"messages": [AIMessage(content="当前尚未实现text2sql")]}


async def rag_search_agent(state):
    inputs = {
        "user_input": state['user_input'],
        'recent_history': state['recent_history'],
        'similar_chats': state['similar_chats'],
        'session_summary': state['session_summary']
    }
    rag_response = await rag_agent.ainvoke(inputs)

    # rag_response is a dict from the RAG graph; extract the final AI message
    final_message = rag_response.get("messages", [])[-1] if isinstance(rag_response, dict) else rag_response
    # Pass the RAG result forward so supervisor_agent can craft the final reply
    return {"messages": [final_message], "recent_history": [final_message]}


web_search_agent = create_react_agent(
    model=llm,
    tools=[get_current_time, search],
    prompt=(
        "你是一个网络搜索的智能体(Agent)。\n\n"
        "指令：\n"
        "- 仅网络数据获取、网络查询、数据查询相关的任务\n"
        "- 回复时仅包含工作结果，不要包含任何其他文字"
    ),
    name="web_search_agent",
)

assign_to_text2sql_agent = create_handoff_tool(
    agent_name="text2sql_agent",
    description="将任务分配给：text2sql_agent智能体。",
)

assign_to_rag_agent = create_handoff_tool(
    agent_name="rag_search_agent",
    description="将任务分配给：rag_search_agent智能体。",
)
assign_to_search_agent = create_handoff_tool(
    agent_name="web_search_agent",
    description="将任务分配给：web_search_agent智能体。",
)


async def supervisor_agent(state):
    user_input = state["user_input"][-1]
    prompt_dict = process_history(state)
    prompt_dict['save_knowledge'] = load_save_knowledge()
    recent_history = prompt_dict.pop('recent_history')
    recent_history.append(user_input)
    character_setting_template = load_prompt(f"{APPS_DIR}/template/CharacterSetting.yaml")
    character_setting_prompt = character_setting_template.invoke(prompt_dict)
    # print('=' * 20)
    # print(character_setting_prompt.text)
    # print('=' * 20)
    supervisor = create_react_agent(
        llm,
        tools=[assign_to_text2sql_agent, assign_to_rag_agent, assign_to_search_agent],
        prompt=character_setting_prompt.text
    )
    resp = await supervisor.ainvoke({"messages": recent_history})
    return {"messages": resp['messages']}


async def _generate_summary_content(session_id: str, message: dict):
    latest_messages = memory.load_memory(session_id, message["user"])
    summary_template = load_prompt(f"{APPS_DIR}/template/SummarySession.yaml")
    summary_chain = summary_template | llm | StrOutputParser()
    summary_input_dict = {
        "user_input": str(message),
        'recent_messages': latest_messages['chat_list'],
        'similar_messages': latest_messages['similar_messages'],
        'summary_history': latest_messages['summary']
    }
    return await summary_chain.ainvoke(summary_input_dict)


async def _generate_session_brief(message: dict):
    brief_template = load_prompt(f"{APPS_DIR}/template/SessionBrief.yaml")
    brief_chain = brief_template | llm | StrOutputParser()
    brief_input_dict = {"recent_messages": str(message)}
    return await brief_chain.ainvoke(brief_input_dict)


async def save_messages(payload):
    session_id = payload.get("session_id")
    message = payload.get("message") or {}
    chat_type = payload.get("chat_type") or "chat"

    if not session_id or not isinstance(message, dict):
        logger.error("save_messages payload missing session_id or message dict: %s", payload)
        return

    if "user" not in message or "system" not in message:
        logger.error("save_messages payload missing user/system content for session_id=%s", session_id)
        return

    create_time = datetime.now()
    session_brief, rows_chat_session = memory.db_manager.get_session_info(session_id)
    session_exists = session_brief is not None
    summary_content = None

    if session_exists and len(rows_chat_session) >= 8:
        try:
            summary_content = await _generate_summary_content(session_id, message)
        except Exception as exc:
            logger.exception("Failed to generate summary for session %s: %s", session_id, exc)

    if not session_exists:
        try:
            session_brief = await _generate_session_brief(message)
        except Exception as exc:
            logger.exception("Failed to generate brief for session %s: %s", session_id, exc)
            session_brief = str(message.get("user", ""))[:64]

    try:
        memory.save_memory(
            message=message,
            create_time=create_time,
            chat_session_id=session_id,
            session_brief=session_brief,
            chat_type=chat_type,
            summary=summary_content,
            new_session=not session_exists
        )
        logger.info(
            "Saved message for session %s (new_session=%s, summary_created=%s)",
            session_id,
            not session_exists,
            bool(summary_content),
        )
    except Exception as exc:
        logger.exception("Persisting chat failed for session %s: %s", session_id, exc)


# if __name__ == '__main__':
#
#     session_id = "123456"
#     message = {"user": "你的性格怎么样", "system": "我的性格。。。"}
#
#     chat_type = "chat"
#     asyncio.run(save_messages(session_id, message, chat_type))
