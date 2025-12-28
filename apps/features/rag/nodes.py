from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import create_react_agent
from langchain.tools import BaseTool
from typing import Literal, List, Any
from pathlib import Path

from apps.logs.logs import get_logger
from .tools import load_tools
from .schema import Grade
from apps.common import llm, load_prompt

logger = get_logger(__name__)

APPS_DIR = Path(__file__).parents[2]


async def summary_history_agent(state):
    summary_history_template = load_prompt(f"{APPS_DIR}/template/SummaryHistory.yaml")
    user_input = state["user_input"]
    human_input = user_input[-1].content
    history_dict = {"session_summary": state["session_summary"]}
    history_dict["similar_chats"] = "【" + ", ".join(str(temp) for temp in state["similar_chats"]) + "】"
    history_dict["recent_history"] = "【" + ", ".join(str(temp) for temp in state["recent_history"]) + "】"
    logger.info(f"history_dict:{history_dict}")
    summary_chain = summary_history_template | llm
    resp = await summary_chain.ainvoke({"user_input": human_input, **history_dict})
    logger.info(f"改写后的输入：{resp.content}")
    rewrite_num = state.get('rewrite_num', -1) + 1
    return {"messages": [resp], "rewrite_num": rewrite_num}


async def validate_summary(state):
    user_input = state["user_input"]
    user_input = user_input[-1].content
    rewrite_num = int(state.get('rewrite_num', -1))
    history_dict = {"session_summary": state["session_summary"]}
    history_dict["similar_chats"] = "【" + ", ".join(str(temp) for temp in state["similar_chats"]) + "】"
    history_dict["recent_history"] = "【" + ", ".join(str(temp) for temp in state["recent_history"]) + "】"
    history_dict["user_input"] = user_input
    messages = state["messages"]
    summary: AIMessage = messages[-1].content
    validate_summary_template = load_prompt(f"{APPS_DIR}/template/ValidateSummary.yaml")
    validate_summary_chain = validate_summary_template | llm.with_structured_output(Grade, method="function_calling")
    resp = await validate_summary_chain.ainvoke({"rewritten_input": summary, **history_dict})
    if resp.binary_score == "yes" or rewrite_num > 2:
        logger.info("总结历史记录正确")
        return "agent"
    else:
        logger.info("总结历史记录错误")
        return "summary_history"


# 3.根据问题判断搜索知识库，如果搜索生成调用工具指令/回答，如果回答直接回复
def agent_node(state):
    """
    创建节点进行问答
    :param state:
    :return:
    """
    logger.info("start RAG")
    human_message: AIMessage | HumanMessage = state["messages"][-1]
    logger.info(f"start RAG human_message:{human_message}")
    search_tools = load_tools()
    search_tools_str = '\n'.join([f"{tool}" for tool in search_tools])
    logger.info("search_tools_len: %s", len(search_tools))
    logger.info("search_tools: %s", search_tools_str)
    llm_with_tool = llm.bind_tools(search_tools)
    response = llm_with_tool.invoke([human_message])
    if hasattr(response, "tool_calls") and response.tool_calls:
        return {"messages": [response], "history": [f"human:{human_message}"]}
    return {"messages": [response], "history": [f"AI:{response.content}"]}


def tools_condition(state):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return 'retrieve'
    return 'END'


def tool_node(state):
    tool_calls = state["messages"][-1].tool_calls
    search_tools = load_tools()
    tools_map = {tool.name: tool for tool in search_tools}
    tool_messages = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        args = tool_call.get("args", {})
        call_id = tool_call.get("id")
        tool = tools_map.get(tool_name)
        if not tool:
            tool_messages.append(
                ToolMessage(
                    tool_call_id=call_id,
                    content=f"Tool `{tool_name}` not found. Available tools: {list(tools_map)}",
                    status="error"
                )
            )
            continue
        try:
            if tool.args_schema:
                validated = tool.args_schema(**args)
                args = validated.model_dump()

            result = tool.func(**args)

            tool_messages.append(
                ToolMessage(
                    tool_call_id=call_id,
                    content=str(result),
                    status="success"
                )
            )
        except Exception as e:
            schema_info = ""
            if tool.args_schema:
                schema_info = (
                        "Expected args schema:\n"
                        + "\n".join(
                    f"- {name}: {field.annotation}"
                    for name, field in tool.args_schema.model_fields.items()
                )
                )

            tool_messages.append(
                ToolMessage(
                    tool_call_id=call_id,
                    content=(
                        f"Tool `{tool_name}` execution failed.\n"
                        f"Error: {e}\n"
                        f"{schema_info}"
                    ),
                    status="error"
                )
            )
            # content, name=requested_tool, tool_call_id=call["id"], status="error"
    return {"messages": tool_messages}


# 4.如果搜索：判断知识库搜索结果和问题的相关性
def grade_documents(state):
    """
    判断检索到的文档是否与问题相关。
    参数:
        state (messages): 当前状态
    返回:
        str: 判断结果，文档是否相关
    """
    logger.info("---检查document的相关性---")
    #  带结构化输出的LLM
    llm_with_structured = llm.with_structured_output(Grade, method="function_calling")

    # 提示模板
    prompt = PromptTemplate(
        template="""你是一个评估检索文档与用户问题相关性的评分器。\n
            这是检索到的文档：\n\n {context} \n\n
            这是用户的问题：{question} \n
            如果文档包含与用户问题相关的关键词或语义含义，则评为相关。\n
            给出二元评分 'yes' 或 'no' 来表示文档是否与问题相关。""",
        input_variables=["context", "question"],
    )

    # 处理链
    chain = prompt | llm_with_structured

    messages = state["messages"]
    tools_messages = []
    for message in messages[::-1]:
        if isinstance(message, ToolMessage):
            tools_messages.append(message)
        break

    question = state["history"][-1]

    related_docs = []
    for tool_message in tools_messages:
        docs = tool_message.content.split('----')
        for doc in docs:
            scored_result = chain.invoke({"question": question, "context": doc})
            score = scored_result.binary_score

            if score == "yes":
                logger.info("文档相关")
                related_docs.append(doc)

            else:
                logger.info("文档不相关")
    content = "\n----\n".join(related_docs)
    return {"messages": [AIMessage(content=content)], "documents": content}


#  5.如果搜索：当相关文本不为空的时候，结合搜索结果针对用户问题生成回复/当相关文本为空的时候，改写用户问题，重新搜索
def judge_rewrite(state) -> Literal["generate", "rewrite"]:
    logger.info("---判断问题是否重写---")
    rewrite_num = state.get("rewrite_num", -1)
    if rewrite_num >= 2:
        logger.info("重写次数过多，放弃重写")
        return "generate"
    if state["messages"][-1].content:
        logger.info("---不需要重写---")
        return "generate"
    logger.info("---需要重写---")
    return "rewrite"


# 6.如果搜索，如果文本为空：重写用户输入并重新搜索。记录重新搜索次数，搜索两次后，相关文本仍为空，依靠模型自己能力回答
def rewrite(state):
    """
    转换查询以生成更好的问题。

    参数:
        state (messages): 当前状态
    返回:
        dict: 包含重述问题的更新后状态
    """
    logger.info("---转换查询---")
    rewrite_num = state.get("rewrite_num", -1)
    rewrite_num += 1
    # messages = state["messages"]
    # question = get_last_human_message(messages).content
    question = state["history"][-1]
    msg = [
        HumanMessage(
            content=f""" \n 
    分析输入并尝试理解潜在的语义意图/含义。\n 
    这是初始问题:
    \n ------- \n
    {question} 
    \n ------- \n
    请提出一个改进后的问题: """,
        )
    ]

    response = llm.invoke(msg)
    return {"messages": [response], "rewrite_num": rewrite_num}


# 7.生成回复如果知识库搜索结果大于两次，则没有相关内容
def generate(state):
    """
    生成答案

    参数:
        state (messages): 当前状态

    返回:
         dict: 包含重述问题的更新后状态
    """
    logger.info("---生成答案---")
    messages = state["messages"]
    # question = get_last_human_message(messages).content
    question = state["history"][-1]
    last_message = messages[-1]

    rewrite_num = state.get("rewrite_num", -1)
    docs = last_message.content if rewrite_num < 2 else "当前问题没有上下文"

    regenerate_num = state.get("regenerate_num", -1)
    if regenerate_num >= 2:
        logger.info("---生成次数过多，放弃生成---")
        return {"messages": [AIMessage(content="很抱歉我不会这个问题")]}

    # 提示模板
    prompt = PromptTemplate(
        template="你是一个问答任务助手。请根据以下检索到的上下文内容回答问题。如果不知道答案，请直接说明。回答保持简洁。\n问题：{question} \n上下文：{context} \n回答：",
        input_variables=["question", "context"],
    )
    # 处理链
    rag_chain = prompt | llm | StrOutputParser()
    # 执行
    response = rag_chain.invoke({"context": docs, "question": question})
    ai_message = AIMessage(content=response)
    return {"messages": [ai_message]}


# 8.当知识库搜索结果不大于两次，针对回复判断是否依赖搜索结果回答并判断回复是否对用户输入有用/当搜索结果大于两次，判断回复是否对用户输入有用
# 基于事实 有用 ->END
# 基于事实 无用 -> 重新检索
# 不基于事实 重新生成
def judge_generate(state):
    """
    评估生成结果是否基于文档并正确回答问题
    Args:
        state (dict): 当前图状态，包含问题、文档和生成结果
    Returns:
        str: 下一节点的名称（useful/not useful/not supported）
    """
    logger.info("---检查生成内容是否存在幻觉---")  # 阶段标识
    question = state["history"][-1]  # 获取用户问题
    documents = state["documents"]  # 获取参考文档
    generation = state["messages"][-1]  # 获取生成结果
    rewrite_num = state.get("rewrite_num", -1)

    regenerate_num = state.get("regenerate_num", -1)
    if regenerate_num >= 2:
        return "useful"  # finally_node

    if rewrite_num < 2:
        hallucination_template = """您是一个评估生成内容是否基于检索事实的评分器。
             给出'yes'或'no'的二元评分。'yes'表示回答是基于/支持于给定事实集的。
             事实：{documents}
             生成内容:{generation}"""
        hallucination_prompt = PromptTemplate.from_template(hallucination_template)

        # 构建幻觉检测工作流
        hallucination_grader_chain = hallucination_prompt | llm.with_structured_output(Grade, method="function_calling")
        # 检查生成是否基于文档
        score = hallucination_grader_chain.invoke({"documents": documents, "generation": generation})
        grade = score.binary_score
    else:
        grade = "yes"

    if grade == "yes":  # 如果生成基于文档
        answer_grader_template = """您是一个评估生成内容是否准确回答问题的评分器。
             给出'yes'或'no'的二元评分。'yes'表示回答与问题一致。
             问题：{question}
             生成内容:{generation}"""
        logger.info("---判定：生成内容基于参考文档---")
        answer_grader_prompt = PromptTemplate.from_template(answer_grader_template)
        answer_grader_chain = answer_grader_prompt | llm.with_structured_output(Grade, method="function_calling")
        # 检查是否准确回答问题
        logger.info("---评估：生成回答与问题的匹配度---")
        score = answer_grader_chain.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":  # 如果正确回答问题
            logger.info("---判定：生成内容准确回答问题---")
            return "useful"  # 返回有用结果 finally_node
        else:  # 如果没有回答问题
            logger.info("---判定：生成内容未能准确回答问题---")
            regenerate_num += 1
            return "not_useful"  # 返回无用结果 rewrite
    else:  # 如果生成不基于文档
        logger.info("---判定：生成内容未基于参考文档，将重新尝试---")
        return "not_supported"  # 返回不支持结果 generate


def finally_node(state):
    """
    改变state中的值
    """
    finally_message = state["messages"][-1]
    return {"history": [finally_message.content], "rewrite_num": -1, "regenerate_num": -1}
