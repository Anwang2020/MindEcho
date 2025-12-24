from .service import rag_workflow, custom_bge
from .web_search import websearch
import numpy as np


def similar_search(embed_model, query, docs, top_k):
    query_vec = embed_model.embed_query(query)
    doc_vecs = embed_model.embed_documents(docs)
    q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    docs_normalized = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)
    sims = np.dot(docs_normalized, q)

    top_indices = np.argsort(sims)[-top_k:][::-1]
    top_docs = '\n----\n'.join([docs[idx] for idx in top_indices])
    return top_docs


def search_func(query, limit=3, llm_invoke=True):
    """需要进行网络搜索时调用此工具"""
    html_content = websearch(query, limit)
    if llm_invoke and len(html_content) > 6000:
        html_content = [{"type": "paragraph", "text": html_content.replace('----', '')}]
        html_content = rag_workflow.build_rag_chunks(html_content)
        html_content = [content['content'] for content in html_content]
        html_content = similar_search(custom_bge, query, html_content, limit)

    return html_content
