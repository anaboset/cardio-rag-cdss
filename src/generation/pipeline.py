from langchain_core.documents import Document
from src.retrieval.retriever import get_retriever
from src.utils.config_loader import CONFIG
from src.utils.logger import trace_task, logger
from src.generation.rewriter import generate_query_variants
from src.generation.generator import get_rag_chain, format_docs

@trace_task
async def rag_response(query: str, patient_summary: str, metadata_filter: dict | None = None):
    """Orchestrates the Hybrid Multi-Query RAG flow."""
    
    # 1. Setup Retrieval Components
    base_retriever = get_retriever(metadata_filter=metadata_filter)
    rag_chain = get_rag_chain()
    
    # 2. Hybrid Retrieval (Multi-Query + Vector/BM25)
    variants = generate_query_variants(query)
    
    
    # Gather Vector Candidates
    all_vector_docs = []
    for var in variants:
        all_vector_docs.extend(base_retriever.invoke(var))
    
    # 3. Deduplication
    seen = set()
    unique_candidates: list[Document] = []

    # Add Vector Chunks
    for doc in all_vector_docs:
        if doc.page_content not in seen:
            unique_candidates.append(doc)
            seen.add(doc.page_content)

    logger.info(f"🧬 Hybrid Recall: {len(unique_candidates)} unique chunks found.")

    # 4. Precision Reranking
    if CONFIG["retrieval"].get("rerank") and len(unique_candidates) > 0:
        from langchain_cohere import CohereRerank

        reranker = CohereRerank(model="rerank-english-v3.0", top_n=CONFIG["retrieval"]["rerank_top_k"])
        final_docs = reranker.compress_documents(unique_candidates, query)
    else:
        final_docs = unique_candidates[:CONFIG["retrieval"]["rerank_top_k"]]

    # 5. Generation
    response = rag_chain.invoke({
        "query": query, 
        "patient_summary": patient_summary,
        "context": format_docs(final_docs)
    })
    
    return response, final_docs, variants