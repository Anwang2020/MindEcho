import asyncio
import time
from langgraph.graph import START, END, StateGraph

from apps.logs.logs import get_logger
from .schema import ChatState
from .nodes import *

logger = get_logger(__name__)

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


async def _stream_text(content: str, chunk_size: int = 80):
    for idx in range(0, len(content), chunk_size):
        yield content[idx: idx + chunk_size]


async def invoke_agent(user_input, chat_type, session_id):
    st = time.time()
    input_dict = {'content': user_input, 'type': chat_type, 'session_id': session_id}
    inputs = {
        "messages": [HumanMessage(content=f"{input_dict}")]
    }
    # Run the full graph first, then stream only the final answer back to the client
    result = await agent.ainvoke(inputs)
    final_messages = result.get("messages") or []
    final_content = ""
    if final_messages:
        last_message = final_messages[-1]
        final_content = getattr(last_message, "content", "") or str(last_message)

    async for chunk in _stream_text(final_content):
        yield chunk
    logger.info("elapsed %.2fs", time.time() - st)



if __name__ == '__main__':
    while True:
        question = input("请输入问题：")
        if question == "exit":
            break
        asyncio.run(invoke_agent(question, 'chat', '123456'))
