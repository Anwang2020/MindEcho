import asyncio
import time
from langgraph.graph import START, END, StateGraph
from schema import ChatState
from nodes import *

agent = (StateGraph(ChatState)
         .add_node("get_history", get_history_agent)
         .add_node("supervisor", supervisor_agent,
                   destinations=("text2sql_agent", 'rag_search_agent', 'web_search_agent', END))
         .add_node("text2sql_agent", text2sql_agent, )
         .add_node("rag_search_agent", rag_search_agent, )
         .add_node("web_search_agent", web_search_agent, )
         .add_edge(START, "get_history")
         .add_edge("get_history", "supervisor")
         .compile())


async def invoke_agent(user_input, chat_type, session_id):
    st = time.time()
    input_dict = {'content': user_input, 'type': chat_type, 'session_id': session_id}
    inputs = {
        "messages": [HumanMessage(content=f'{input_dict}')]
    }
    async for event in agent.astream_events(inputs, version="v2"):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            yield chunk.content

        elif event["event"] == "on_tool_start":
            print(f"\n[调用工具 {event['name']}]")

        elif event["event"] == "on_tool_end":
            print(f"\n[工具 {event['name']} 完成]")
    print(f'用时{time.time() - st:.2f}s')


if __name__ == '__main__':
    while True:
        question = input("请输入问题：")
        if question == "exit":
            break
        asyncio.run(invoke_agent(question, 'chat', '123456'))
