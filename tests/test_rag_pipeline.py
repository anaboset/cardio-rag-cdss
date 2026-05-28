import pytest
import os
import asyncio
from src.generation.pipeline import rag_response

@pytest.mark.integration
@pytest.mark.timeout(30) # Prevent indefinite hanging
def test_rag_retrieval_quality():
    # Integration tests (LLM + retrieval) skip by default.
    if os.getenv("RUN_LLM_TESTS") != "1":
        pytest.skip("Skipping integration test: set RUN_LLM_TESTS=1 to enable.")

    # rag_response uses "auto" provider:
    # - Groq if GROQ_API_KEY is set
    # - otherwise Ollama (local) if available
    if not os.getenv("GROQ_API_KEY"):
        # No Groq key: still allow Ollama-based demo, but we have no reliable env var to validate.
        # If you want deterministic CI, set RUN_LLM_TESTS=1 with the right local services.
        pass

    query = "What is the recommended treatment for hypertension in elderly patients?"
    patient_summary = "65-year-old with BP 160/95"
    
    try:
        response, docs, variants = asyncio.run(rag_response(query, patient_summary))
        
        # Assertions
        assert len(docs) > 0, "No guidelines were retrieved"
        assert "hypertension" in response.lower()
        # Ensure we are pulling from the authoritative source mentioned in your proposal
        assert any("ESC" in str(d.metadata.get("source", "")) for d in docs), "ESC guidelines not found in context"
        
    except Exception as e:
        pytest.fail(f"RAG Pipeline crashed with error: {e}")