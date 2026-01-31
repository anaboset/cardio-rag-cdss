import pytest
import os
from src.generation.pipeline import rag_response

@pytest.mark.integration
@pytest.mark.timeout(30) # Prevent indefinite hanging
def test_rag_retrieval_quality():
    # Pre-test check: Ensure API keys are present
    if not os.getenv("GROQ_API_KEY") or not os.getenv("neo4j_uri"):
        pytest.skip("Skipping integration test: Missing API keys or DB URI")

    query = "What is the recommended treatment for hypertension in elderly patients?"
    patient_summary = "65-year-old with BP 160/95"
    
    try:
        response, docs, variants = rag_response(query, patient_summary)
        
        # Assertions
        assert len(docs) > 0, "No guidelines were retrieved"
        assert "hypertension" in response.lower()
        # Ensure we are pulling from the authoritative source mentioned in your proposal
        assert any("ESC" in str(d.metadata.get("source", "")) for d in docs), "ESC guidelines not found in context"
        
    except Exception as e:
        pytest.fail(f"RAG Pipeline crashed with error: {e}")