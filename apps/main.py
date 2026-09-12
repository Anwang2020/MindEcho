from fastapi import FastAPI
import uvicorn

from apps.api import api_router
from apps.config import app_cross

app = FastAPI()
app_cross(app)
app.include_router(api_router)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
