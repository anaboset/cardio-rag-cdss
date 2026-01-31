import pytest
from pathlib import Path
from src.ingestion.loader import load_and_chunk_pdf

def test_load_and_chunk_pdf(tmp_path):
    """Verify that a PDF is correctly split into chunks with metadata."""
    # Create a dummy PDF path (Note: requires a real small pdf or mocking PyPDFLoader)
    # For unit testing, we often mock the PyPDFLoader to avoid needing real files
    test_pdf = tmp_path / "test_guideline.pdf"
    test_pdf.write_text("dummy content") 

    # If you have a sample medical pdf in your data folder, use that path instead
    sample_path = Path("data/guidelines/sample.pdf")
    if not sample_path.exists():
        pytest.skip("Sample PDF not found for loader test.")

    chunks = load_and_chunk_pdf(sample_path)
    
    assert len(chunks) > 0
    assert "source" in chunks[0].metadata
    assert "chunk_id" in chunks[0].metadata
    assert chunks[0].metadata["source"] == "sample.pdf"