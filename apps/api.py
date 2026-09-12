from fastapi import APIRouter

from apps.features import rag_router, conversation_router

api_router = APIRouter(tags=["apps"], prefix="/apps")
api_router.include_router(rag_router)
api_router.include_router(conversation_router)
