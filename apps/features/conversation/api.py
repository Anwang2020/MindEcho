import httpx
from fastapi import FastAPI, WebSocket, BackgroundTasks

from agent import invoke_agent
from nodes import save_messages

app = FastAPI()


@app.websocket("/ws/chat")
async def websocket_echo(websocket: WebSocket):
    await websocket.accept()
    user_input = await websocket.receive_text()
    user_input_dict = eval(user_input)
    user_input = user_input_dict.get("content")
    chat_type = user_input_dict.get("type")
    session_id = user_input_dict.get("session_id")
    full_content = ''
    print(f"开始处理: {user_input}")
    async for chunk in invoke_agent(user_input, chat_type, session_id):
        full_content += chunk
        print(chunk, end='|')
        await websocket.send_text(chunk)
    message = {"user": user_input, "system": full_content}
    save_message_input = {"message": message, "session_id": session_id, "chat_type": chat_type}
    async with httpx.AsyncClient() as client:
        await client.post("http://127.0.0.1:8000/save_message", json=save_message_input)
    await websocket.close()


@app.post("/save_message")
async def save_message_api(payload: dict, background_tasks: BackgroundTasks):
    background_tasks.add_task(save_messages, payload)
    return {"status": "ok"}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
# from fastapi import APIRouter, WebSocket, Depends
#
#
# router = APIRouter(tags=["Data Q&A"], prefix="/chat")
#
#
# @router.websocket("/ws/chat")
# async def websocket_echo(websocket: WebSocket, llm: ChatPipeline = Depends(get_chat_pipeline)):
#     """先写自己的逻辑，后续再改进
#     1.获取用户输入
#     2.加载历史记录和对话模板
#     3.生成对话写入"""
#     await websocket.accept()
#     user_input = await websocket.receive_text()
#     async for chunk in llm.invoke(user_input):
#         await websocket.send_text(chunk)
#     await websocket.close()
