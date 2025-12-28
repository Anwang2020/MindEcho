import httpx
from fastapi import APIRouter, WebSocket, BackgroundTasks

from apps.logs.logs import get_logger
from .agent import invoke_agent
from .nodes import save_messages

logger = get_logger(__name__)

router = APIRouter(tags=["chat"], prefix="/chat")


@router.websocket("/ws/chat")
async def websocket_echo(websocket: WebSocket):
    await websocket.accept()
    user_input = await websocket.receive_text()
    user_input_dict = eval(user_input)
    user_input = user_input_dict.get("content")
    chat_type = user_input_dict.get("type")
    session_id = user_input_dict.get("session_id")
    full_content = ''
    logger.info(f"开始处理: {user_input}")
    async for chunk in invoke_agent(user_input, chat_type, session_id):
        full_content += chunk
        logger.info(chunk)
        await websocket.send_text(chunk)
    message = {"user": user_input, "system": full_content}
    save_message_input = {"message": message, "session_id": session_id, "chat_type": chat_type}
    async with httpx.AsyncClient() as client:
        await client.post("http://127.0.0.1:8000/apps/chat/save_message", json=save_message_input)
    await websocket.close()


@router.post("/save_message")
async def save_message_api(payload: dict, background_tasks: BackgroundTasks):
    background_tasks.add_task(save_messages, payload)
    return {"status": "ok"}


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)