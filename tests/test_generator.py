import sys
import os
from langchain_core.documents import Document

# Ensure src is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.generation.generator import get_rag_chain, format_docs

def test_generator_scenarios():
    print("\n🩺 --- STARTING GENERATOR SANITY CHECK --- 🧪")
    
    rag_chain = get_rag_chain()
    patient_summary = "65-year-old male, Smoker, BP 155/95, LDL 190."
    
    # --- SCENARIO 1: Perfect Context ---
    print("\n[Scenario 1]: Valid Guidelines Found")
    mock_docs = [
        Document(page_content="ESC 2024: Patients with BP > 140/90 should start ACE inhibitors.", metadata={"source": "ESC_2024.pdf"}),
        Document(page_content="High intensity statins are recommended for LDL > 180.", metadata={"source": "Statin_Guidelines.pdf"})
    ]
    
    res1 = rag_chain.invoke({
        "query": "What is the management plan for this patient?",
        "patient_summary": patient_summary,
        "context": format_docs(mock_docs)
    })
    print(f"✅ Response with context: {res1[:150]}...")

    # --- SCENARIO 2: Empty Context (The Hallucination Test) ---
    print("\n[Scenario 2]: No Guidelines Found (Hallucination Test)")
    res2 = rag_chain.invoke({
        "query": "What is the specific dosage for the experimental drug X-99?",
        "patient_summary": patient_summary,
        "context": "No relevant medical guidelines were found in the database."
    })
    
    # We want to see if the LLM is honest here
    print(f"⚠️ Response with NO context: {res2}")
    
    print("\n🧪 --- SANITY CHECK COMPLETE --- 🧪")

if __name__ == "__main__":
    test_generator_scenarios()