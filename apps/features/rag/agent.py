import uuid
from langgraph.graph import StateGraph, MessagesState, START, END
# from langgraph.checkpoint.memory import InMemorySaver
# from langgraph.prebuilt import ToolNode, tools_condition

from .schema import RAGState
from .nodes import *

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

search_tools = load_tools()

graph = StateGraph(RAGState)
graph.add_node('agent', agent_node)
graph.add_node('retrieve', tool_node)
# graph.add_node('retrieve', ToolNode(search_tools))
graph.add_node('grade_documents', grade_documents)
graph.add_node('rewrite', rewrite)
graph.add_node('generate', generate)
graph.add_node('finally', finally_node)

# 3.根据问题判断是否搜索知识库，如果搜索生成调用搜索知识库工具指令/回答，如果回答直接回复
graph.add_edge(START, 'agent')
graph.add_conditional_edges(
    'agent',
    tools_condition,
    {
        'retrieve': 'retrieve',
        'END': END
    }
)
# graph.add_conditional_edges(
#     'agent',
#     tools_condition,
#     {
#         'tools': 'retrieve',
#         END: END
#     }
# )

# 4.如果搜索：逐个判断判断知识库搜索结果和问题的相关性，将相关文本加入state
graph.add_edge('retrieve', 'grade_documents')

# 5.如果搜索：当相关文本不为空的时候，结合搜索结果针对用户问题生成回复/当相关文本为空的时候，改写用户问题，重新搜索
graph.add_conditional_edges(
    'grade_documents',
    judge_rewrite,
)

# 6.如果搜索，如果文本为空：重写用户输入并重新搜索，记录重新搜索次数，搜索两次后，相关文本仍为空，依靠模型自己能力回答
graph.add_edge('rewrite', 'agent')

# 7.生成回复如果知识库搜索结果大于两次，则没有相关内容
# 8.当知识库搜索结果不大于两次，针对回复判断是否依赖搜索结果回答并判断回复是否对用户输入有用/当搜索结果大于两次，判断回复是否对用户输入有用
graph.add_conditional_edges(
    'generate',
    judge_generate,
    {
        "useful": "finally",
        "not_useful": "rewrite",
        "not_supported": "generate"
    }
)

# 9.更新状态值
graph.add_edge('finally', END)
# 检查点让状态图可以持久化其状态
# 这是整个状态图的完整内存
# memory = InMemorySaver()

# 编译状态图，配置检查点为memory, 配置中断点
graph = graph.compile()

if __name__ == '__main__':
    config = {
        "configurable": {
            # 检查点由session_id访问
            "thread_id": str(uuid.uuid4()),
        }
    }
    while True:
        is_debug = False
        question = input('用户：')
        if question.lower() in ['q', 'exit', 'quit']:
            logger.info('对话结束，拜拜！')
            break
        else:
            inputs = {
                "messages": [
                    ("user", question),
                ]
            }
            if is_debug:
                events = graph.stream(inputs, config=config, stream_mode='values')
                # 打印消息
                for event in events:
                    print(event["messages"][-1].content)
            else:
                resp = graph.invoke(inputs, config=config)
                print(resp["messages"][-1].content)


    def draw_graph(graph, file_name: str):
        mermaid_code = graph.get_graph().draw_mermaid_png()
        with open(file_name, "wb") as f:
            f.write(mermaid_code)


    draw_graph(graph, 'graph.png')
