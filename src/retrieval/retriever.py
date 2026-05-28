from threading import Lock
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from src.utils.config_loader import CONFIG
from src.utils.logger import logger

class RetrieverManager:
    _instance = None
    _lock = Lock()

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=CONFIG["embedding"]["model"], 
            model_kwargs={"device": "cpu"}
        )
        self.vectorstore = Chroma(
            collection_name=CONFIG["vectorstore"]["collection_name"],
            embedding_function=self.embeddings,
            persist_directory=CONFIG["vectorstore"]["persist_directory"]
        )
        self.bm25_retriever = None
        self._build_bm25()


    def _build_bm25(self):
        """Internal helper to build BM25 index from existing Chroma docs."""
        
        # fetch all documents currently in the vectorstore
        all_content = self.vectorstore.get()
        
        if all_content and len(all_content.get("documents", [])) > 0:
            docs = [
                Document(page_content=d, metadata=m) 
                for d, m in zip(all_content.get("documents", []), all_content.get("metadatas", []))
            ]

            doc_count = len(all_content.get("documents", []))
            if doc_count > 0:
                logger.info(f"📚 BM25: Indexing {doc_count} documents for keyword search.")
            else:
                logger.warning("⚠️ BM25: No documents found in ChromaDB. Keyword search will be disabled.")

            self.bm25_retriever = BM25Retriever.from_documents(docs)
            self.bm25_retriever.k = CONFIG["retrieval"]["k"]
        else:
            self.bm25_retriever = None


    def refresh_bm25(self):
        """Refresh BM25 index after new data ingestion."""
        self._build_bm25()
        print("🔄 BM25 index refreshed with new data.")


    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance


def get_retriever(metadata_filter: dict | None = None):
    
    manager = RetrieverManager.get_instance()
    
    # 1. Dense Retriever
    search_kwargs = {"k": CONFIG["retrieval"]["k"]}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter
    
    dense_retriever = manager.vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    # 2. Hybrid Ensemble Retriever
    if manager.bm25_retriever:
        base_retriever = EnsembleRetriever(
            retrievers=[manager.bm25_retriever, dense_retriever],
            weights=[1 - CONFIG["retrieval"]["hybrid_alpha"], CONFIG["retrieval"]["hybrid_alpha"]]
        )
    else:
        # Fallback to just dense if no docs are in BM25 yet
        base_retriever = dense_retriever
    
    return base_retriever