from fastapi import FastAPI
import uvicorn

from apps.config import app_cross
from apps.chat import api as chat_api

app = FastAPI(title="MindEcho", version="1.0.0")
app = app_cross(app)
# 注册路由
app.include_router(chat_api.router)


@app.get("/")
async def root():
    return {"message": "Hello, FastAPI!"}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8001)
