from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .service import text2sql_service

router = APIRouter(tags=["text2sql"], prefix="/text2sql")


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@router.get("/schema")
def get_schema():
    try:
        return text2sql_service.schema()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/query")
async def query_database(payload: QueryRequest):
    try:
        result = await text2sql_service.query(payload.question)
        return {"answer": text2sql_service.as_markdown(result), "sql": result.sql, "columns": result.columns, "rows": result.rows, "truncated": result.truncated}
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Text2SQL query failed") from exc
