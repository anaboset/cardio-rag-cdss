from pathlib import Path
import asyncio
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from src.utils.config_loader import CONFIG
from src.utils.logger import trace_task, logger
from src.retrieval.retriever import RetrieverManager


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Fast sliding-window chunking (character-based)."""
    text = (text or "").strip()
    if not text:
        return []

    chunk_size = max(100, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size - 1))

    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks


@trace_task
def load_and_chunk_pdf(pdf_path: Path):
    """
    Parses a single PDF and prepares it for ingestion.
    """
    logger.info(f"📂 Processing file: {pdf_path.name}")
    
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    chunking_cfg = CONFIG.get("chunking", {})
    chunk_size = int(chunking_cfg.get("size", 1000))
    overlap = int(chunking_cfg.get("overlap", 200))

    chunks = []
    for page_index, page in enumerate(pages):
        text = page.page_content or ""
        page_chunks = _chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        for chunk_index, chunk_text in enumerate(page_chunks):
            # Keep metadata stable for retrieval and citations.
            meta = dict(page.metadata or {})
            meta.update(
                {
                    "source": pdf_path.name,
                    "chunk_id": f"{pdf_path.name}_pg_{page_index}_c_{chunk_index}",
                    "type": "cardiology_guideline",
                    "page": page_index,
                }
            )
            chunks.append(Document(page_content=chunk_text, metadata=meta))

    logger.info(f"✂️ Created {len(chunks)} chunks from {pdf_path.name}")
    return chunks


@trace_task
def sync_to_vector_store(all_chunks):
    """Batches data into ChromaDB and refreshes BM25."""
    manager = RetrieverManager.get_instance()

    # Batch to reduce peak RAM.
    batch_size = 64
    for i in range(0, len(all_chunks), batch_size):
        manager.vectorstore.add_documents(all_chunks[i : i + batch_size])
    manager.refresh_bm25()
    logger.info("🧬 Vector: Successfully indexed all chunks into ChromaDB.")


@trace_task
async def ingest_guidelines():
    """Main Orchestrator for the Ingestion Pipeline."""
    
    raw_dir = Path(CONFIG["paths"]["raw_data"])
    pdf_files = list(raw_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"⚠️ No PDF files found in {raw_dir}")
        return

    all_chunks = []

    print("\n🚀 Starting PDF Ingestion...")

    for pdf_path in tqdm(pdf_files, desc="Overall Progress", unit="file"):
        try:
            # Step A: Load and Chunk
            chunks = load_and_chunk_pdf(pdf_path)

            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"❌ Failed to process {pdf_path.name}: {e}", exc_info=True)

    
    if all_chunks:
        sync_to_vector_store(all_chunks)
    
    logger.info("✅ Ingestion Pipeline Completed Successfully.")


if __name__ == "__main__":
    asyncio.run(ingest_guidelines())