import threading
import asyncio
from datetime import datetime
from langgraph.prebuilt import create_react_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, AIMessageChunk
from langgraph.config import get_stream_writer
from pathlib import Path

from apps.common import llm, load_prompt
from utils import process_history, create_handoff_tool, load_save_knowledge
from schema import Validate
from tools import search, get_current_time, rag_agent
from memory import memory

APPS_DIR = Path(__file__).parents[2]


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
    question = state["messages"][-1].content
    inputs = {
        "messages": [
            ("user", question),
        ]
    }
    rag_response = await rag_agent.ainvoke(inputs)
    return {"messages": [rag_response]}


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


async def save_messages(payload):
    try:
        session_id = payload["session_id"]
        message = payload["message"]
        chat_type = payload["chat_type"]
        create_time = datetime.now()
        session_brief, rows_chat_session = memory.db_manager.get_session_info(session_id)
        print("保存会话信息:", session_brief)
        if session_brief:
            print("保存会话信息:")
            if len(rows_chat_session) >= 8:
                print("会话已满8条，请清理历史记录")
                # 异步生成 summary
                latest_messages = memory.load_memory(session_id, message["user"])
                summary_template = load_prompt(f"{APPS_DIR}/template/SummarySession.yaml")
                summary_chain = summary_template | llm | StrOutputParser()
                summary_input_dict = {
                    "user_input": str(message),
                    'recent_messages': latest_messages['chat_list'],
                    'similar_messages': latest_messages['similar_messages'],
                    'summary_history': latest_messages['summary']
                }
                summary_content = await summary_chain.ainvoke(summary_input_dict)

                memory.save_memory(
                    message=message,
                    create_time=create_time,
                    chat_session_id=session_id,
                    session_brief=session_brief,
                    chat_type=chat_type,
                    summary=summary_content
                )
                print("save_memory > 8 stop ")
            else:
                print("huihua ")
                memory.save_memory(
                    message=message,
                    create_time=create_time,
                    chat_session_id=session_id,
                    session_brief=session_brief,
                    chat_type=chat_type
                )
                print("save_memory < 8 stop ")

        else:
            # 异步生成 brief
            print("生成会话brief")
            brief_template = load_prompt(f"{APPS_DIR}/template/SessionBrief.yaml")
            brief_chain = brief_template | llm | StrOutputParser()
            brief_input_dict = {"recent_messages": str(message)}
            brief_content = await brief_chain.ainvoke(brief_input_dict)

            memory.save_memory(
                message=message,
                create_time=create_time,
                chat_session_id=session_id,
                session_brief=brief_content,
                chat_type=chat_type,
                new_session=True
            )
            print("异步生成 < 8 stop ")

    except Exception as e:
        print("异步处理 summary/brief 出错: %s", e)


# if __name__ == '__main__':
#
#     session_id = "123456"
#     message = {"user": "你的性格怎么样", "system": "我的性格。。。"}
#
#     chat_type = "chat"
#     asyncio.run(save_messages(session_id, message, chat_type))
