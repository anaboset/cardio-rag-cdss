import nltk
from pathlib import Path
import asyncio
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from src.utils.config_loader import CONFIG
from src.utils.logger import trace_task, logger
from src.graph.manager import GraphitiManager
from src.retrieval.retriever import RetrieverManager


def apply_rolling_overlap(pages):
    """
    Logic to ensure clinical context isn't cut off at page breaks.
    Standardizes page content into chunks with a sliding window."""

    for i in range(1, len(pages)):
        prev_content = pages[i-1].page_content.strip()
        if not prev_content: continue

        sentences = nltk.sent_tokenize(prev_content)
        if sentences:
            last_sentence = sentences[-1]
            pages[i].page_content = f"...{last_sentence}\n\n{pages[i].page_content}"
    
    return pages


@trace_task
def load_and_chunk_pdf(pdf_path: Path):
    """
    Parses a single PDF and prepares it for ingestion.
    """
    logger.info(f"📂 Processing file: {pdf_path.name}")
    
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()
    
    # Enrich metadata for better retrieval
    for i, page in enumerate(pages):
        page.metadata.update({
            "source": pdf_path.name, 
            "chunk_id": f"{pdf_path.name}_pg_{i}",
            "type": "cardiology_guideline"
        })
    
    chunks = apply_rolling_overlap(pages)
    logger.info(f"✂️ Created {len(chunks)} chunks from {pdf_path.name}")
    return chunks


@trace_task
async def sync_to_graphiti(chunks, graph_manager: GraphitiManager):
    """
    Sends chunks to Graphiti. 
    Graphiti uses Ollama to extract triplets (Subject-Predicate-Object).
    """
    for chunk in chunks:
        await graph_manager.ingest_clinical_episode(
            text=chunk.page_content,
            source=chunk.metadata["source"]
        )
    logger.info(f"🕸️ Graphiti: Successfully processed {len(chunks)} episodes via Ollama.")


@trace_task
def sync_to_vector_store(all_chunks):
    """Batches data into ChromaDB and refreshes BM25."""
   
    manager = RetrieverManager.get_instance()
    manager.vectorstore.add_documents(all_chunks)
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

    # Initialize Graphiti Manager
    graph_manager = GraphitiManager()
    all_chunks = []

    print("\n🚀 Starting PDF Ingestion...")

    for pdf_path in tqdm(pdf_files, desc="Overall Progress", unit="file"):
        try:
            # Step A: Load and Chunk
            chunks = load_and_chunk_pdf(pdf_path)
            
            # Step B: Sync to Knowledge Graphmediate 
            await sync_to_graphiti(chunks, graph_manager)
            
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"❌ Failed to process {pdf_path.name}: {e}", exc_info=True)

    
    if all_chunks:
        sync_to_vector_store(all_chunks)
    
    logger.info("✅ Ingestion Pipeline Completed Successfully.")


if __name__ == "__main__":
    asyncio.run(ingest_guidelines())