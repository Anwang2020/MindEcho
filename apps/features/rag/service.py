from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.prompts import load_prompt
import asyncio
import pathlib
import random

from .parse import pdf_parser, excel_parser, word_parser
from .repository import LancedbManager
from apps.common import custom_bge, llm
from apps.logs.logs import get_logger

tokenizer = custom_bge.model.tokenizer
APP_DIR = pathlib.Path(__file__).parents[2]
logger = get_logger(__name__)


class FileProcess:
    @staticmethod
    def _get_file_format(file_name):
        file_format, file_title = None, ''
        if isinstance(file_name, str):
            file_format = file_name.split('.')[-1]
            file_title = file_name.split('.')[:-1]
        if not file_format:
            raise TypeError('file_name must be a string')
        return file_format, file_title

    async def process_file(self, file_name):
        file_format, file_title = self._get_file_format(file_name)
        if file_format == 'pdf':
            file_response = await pdf_parser(file_name)

        elif file_format in ['docx', 'doc']:
            file_response = word_parser(file_name)

        elif file_format in ['xlsx', 'xls']:
            file_response = excel_parser(file_name), 'excel', file_title

        elif file_format in ['txt', 'md', 'html', 'py', 'log']:
            try:
                with open(file_name, 'r', encoding='utf-8') as f:
                    text = f.read()
            except:
                with open(file_name, 'r', encoding='gbk') as f:
                    text = f.read()

            file_response = [{"type": "paragraph", "text": text}]
        else:
            raise TypeError(f'{file_format} format not support')
        return file_response


class RAGWorkFlow(FileProcess):
    def __init__(self):
        self.lancedb_manager = LancedbManager(custom_bge)

    async def preprocess(self, file_names, min_tokens=500, max_tokens=2000):
        try:
            file_names = file_names if isinstance(file_names, list) else [file_names]
            multi_file_chunks = []
            for file_name in file_names:
                logger.info(f'processing {file_name}')
                file_response = await self.process_file(file_name)
                file_name = pathlib.Path(file_name).name
                chunks = self.build_rag_chunks(file_response, min_tokens, max_tokens, file_name)
                multi_file_chunks.extend(chunks)
            await self.save_chunks(multi_file_chunks)
            return 'upload success'
        except Exception as e:
            logger.exception("Preprocess failed")
            return 'upload failed'

    async def save_chunks(self, chunks):
        random_chunks = random.sample(range(len(chunks)), k=min(3, len(chunks)))
        random_chunks = '\n\n'.join([chunks[i]['content'] for i in random_chunks])
        try:
            template = load_prompt(f"{APP_DIR}/template/ToolDescription.yaml", encoding="utf-8")
        except:
            template = load_prompt(f"{APP_DIR}/template/ToolDescription.yaml", encoding="gbk")
        description = llm.invoke(f"{template.template}\n{random_chunks}").content
        embedding_dim = len(custom_bge.embed_query("你好"))
        table_id = self.lancedb_manager.registry_table(description, embedding_dim)
        self.lancedb_manager.insert(table_id, chunks)

    @staticmethod
    def build_rag_chunks(blocks, min_tokens=500, max_tokens=2000, file_name=""):
        if len(blocks) < 30:
            min_tokens = min(min_tokens, len(''.join(block['text'] for block in blocks)))
        logger.info(f"{file_name} is building rag chunks...")
        chunks = []
        title_stack = []
        current_chunk = {"title_path": [], "content": []}
        current_tokens = 0

        def flush():
            nonlocal current_tokens
            if current_chunk["content"]:
                chunk_text = "\n".join(current_chunk["content"])
                tokens = len(tokenizer.encode(chunk_text))
                # 只有满足最小 token 才 flush
                if tokens >= min_tokens:
                    if tokens > max_tokens:
                        text_splitter = SemanticChunker(
                            embeddings=custom_bge,  # 嵌入模型
                            breakpoint_threshold_type="percentile",  # 拆分阈值策略
                            breakpoint_threshold_amount=75,  # 阈值具体数值
                            min_chunk_size=min_tokens,  # 最小块大小
                            number_of_chunks=int(tokens / max_tokens) + 1,  # 最大块大小
                            sentence_split_regex=r"(?<=[。.?!])\s+|\n\n|\n",  # 句子分隔符（支持多语言）
                        )
                        docs = text_splitter.create_documents([chunk_text])
                        for doc in docs:
                            chunks.append({
                                "title_path": current_chunk["title_path"][:],
                                "content": doc.page_content,
                                "file_name": file_name
                            })
                            current_chunk["content"].clear()
                            current_tokens = 0
                    else:
                        chunks.append({
                            "title_path": current_chunk["title_path"][:],
                            "content": chunk_text,
                            "file_name": file_name
                        })
                        current_chunk["content"].clear()
                        current_tokens = 0

        for block in blocks:

            # ---- Heading 强制 flush ----
            if block["type"] == "title":
                flush()
                level = block.get("level", 1)
                title_stack[:] = title_stack[:level - 1]
                title_stack.append(block["text"])
                current_chunk["title_path"] = title_stack[:]
                continue

            block_text = block["text"]
            block_tokens = len(tokenizer.encode(block_text))
            # ---- 超大块单独 chunk ----
            if block_tokens >= max_tokens:
                current_chunk["content"].append(block_text)
                flush()
                # chunks.append({
                #     "title_path": current_chunk["title_path"][:],
                #     "content": block_text,
                #     "file_name": file_name
                # })
                continue

            # ---- 判断是否超 max token ----
            if current_tokens + block_tokens > max_tokens:
                flush()

            # ---- 加入当前 chunk ----
            current_chunk["content"].append(block_text)
            current_tokens += block_tokens

        flush()
        return chunks

    # ========== chunk → LLM 输入 ==========
    @staticmethod
    def format_chunk_for_llm(chunk):
        title_path = " > ".join(chunk["title_path"]) or ""

        header, title = "", ""
        if file_name := chunk.get("file_name", ""):
            header = f"以下内容来自文档《{file_name}》：\n\n"
        if title_path:
            title = f"章节路径：{title_path}\n\n"
        return f"""{header}{title}正文内容：\n{chunk['content']}"""

    def search(self, query: str, limit: int = 5, table_id=""):
        similar_chunks = self.lancedb_manager.search_similar(query, limit, table_id)
        formatted_chunks = '\n----\n'.join([self.format_chunk_for_llm(chunk) for chunk in similar_chunks])
        return formatted_chunks


rag_workflow = RAGWorkFlow()

if __name__ == '__main__':
    async def main():
        await rag_workflow.preprocess(
            r'C:\Users\27970\Documents\基于人工智能的生产成本量化分析方案.docx')
        formatted_chunks = rag_workflow.search()
        logger.info(formatted_chunks)


    asyncio.run(main())
