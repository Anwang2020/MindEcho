"""Schema-aware, read-only Text2SQL service."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, inspect, text

from apps.common import llm


class SQLPlan(BaseModel):
    sql: str = Field(description="A single read-only SQL SELECT or WITH query")
    explanation: str = Field(description="Short Chinese explanation of what the query returns")


@dataclass(frozen=True)
class QueryResult:
    sql: str
    explanation: str
    columns: list[str]
    rows: list[dict[str, Any]]
    truncated: bool


class Text2SQLService:
    """Generate SQL from the configured database schema and execute it safely."""

    forbidden = re.compile(
        r"\b(insert|update|delete|merge|drop|alter|create|replace|truncate|grant|revoke|attach|detach|pragma|vacuum|call|exec(?:ute)?)\b",
        re.IGNORECASE,
    )

    def __init__(self, database_url: str | None = None, max_rows: int | None = None) -> None:
        self.database_url = database_url or os.getenv("TEXT2SQL_DATABASE_URL")
        self.max_rows = max_rows or int(os.getenv("TEXT2SQL_MAX_ROWS", "100"))

    def _engine(self):
        if not self.database_url:
            raise RuntimeError("TEXT2SQL_DATABASE_URL is not configured")
        return create_engine(self.database_url, pool_pre_ping=True)

    def schema(self) -> dict[str, list[dict[str, str]]]:
        engine = self._engine()
        try:
            inspector = inspect(engine)
            return {
                table_name: [
                    {"name": column["name"], "type": str(column["type"]), "nullable": str(column.get("nullable", True))}
                    for column in inspector.get_columns(table_name)
                ]
                for table_name in inspector.get_table_names()
            }
        finally:
            engine.dispose()

    def schema_text(self) -> str:
        schema = self.schema()
        return json.dumps(schema, ensure_ascii=False, indent=2) if schema else "数据库中没有可查询的数据表。"

    def validate_sql(self, sql: str) -> str:
        normalized = sql.strip()
        if normalized.endswith(";"):
            normalized = normalized[:-1].strip()
        if not normalized or ";" in normalized:
            raise ValueError("Only one SQL statement is allowed")
        if not re.match(r"^(select|with)\b", normalized, flags=re.IGNORECASE):
            raise ValueError("Only SELECT or WITH read-only queries are allowed")
        if self.forbidden.search(normalized):
            raise ValueError("Write, DDL, and administrative SQL are not allowed")
        return normalized

    def execute(self, sql: str) -> QueryResult:
        safe_sql = self.validate_sql(sql)
        engine = self._engine()
        try:
            with engine.connect() as connection:
                rows = connection.execute(text(safe_sql)).mappings().fetchmany(self.max_rows + 1)
        finally:
            engine.dispose()
        truncated = len(rows) > self.max_rows
        serialized = [{key: self._json_value(value) for key, value in row.items()} for row in rows[: self.max_rows]]
        return QueryResult(safe_sql, "", list(serialized[0]) if serialized else [], serialized, truncated)

    @staticmethod
    def _json_value(value: Any) -> Any:
        return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)

    async def query(self, question: str) -> QueryResult:
        prompt = f"""你是 Text2SQL 专家。根据数据库结构将用户问题转换为 SQL。
数据库结构：
{self.schema_text()}
用户问题：{question}
约束：只能生成单条 SELECT 或 WITH 查询；不能写入、修改结构或执行管理命令；只能使用已有表和字段；返回 SQL 及一句中文说明。"""
        plan = await llm.with_structured_output(SQLPlan).ainvoke([HumanMessage(content=prompt)])
        result = self.execute(plan.sql)
        return QueryResult(result.sql, plan.explanation, result.columns, result.rows, result.truncated)

    @staticmethod
    def as_markdown(result: QueryResult) -> str:
        heading = result.explanation or "查询已完成。"
        if not result.rows:
            return f"{heading}\n\n未查询到符合条件的数据。\n\n```sql\n{result.sql}\n```"
        columns = result.columns
        header = "| " + " | ".join(columns) + " |"
        divider = "| " + " | ".join("---" for _ in columns) + " |"
        body = "\n".join("| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |" for row in result.rows)
        notice = f"\n\n结果已按上限截取前 {len(result.rows)} 行。" if result.truncated else ""
        return f"{heading}\n\n{header}\n{divider}\n{body}{notice}\n\n```sql\n{result.sql}\n```"


text2sql_service = Text2SQLService()
