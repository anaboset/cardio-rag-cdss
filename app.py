import streamlit as st
import asyncio
from pathlib import Path
from src.generation.pipeline import rag_response
from src.ingestion.loader import ingest_guidelines

# App Configuration
st.set_page_config(page_title="CardioCDSS", page_icon="🩺", layout="wide")

st.title("🩺 CardioCDSS")
st.markdown("""
**Bridging the Evidence–Practice Gap in Cardiovascular Care.** Use this system to ingest authoritative guidelines and generate patient-specific management plans.
""")

# Sidebar: Management & Uploads
with st.sidebar:
    st.header("⚙️ Data Management")
    uploaded_files = st.file_uploader(
        "Upload ESC Guidelines (PDF)", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if st.button("🚀 Ingest Guidelines"):
        if uploaded_files:
            # Save files to raw data directory defined in your config
            raw_data_path = Path("data/guidelines")
            raw_data_path.mkdir(parents=True, exist_ok=True)
            
            for uploaded_file in uploaded_files:
                with open(raw_data_path / uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            with st.spinner("Indexing guidelines into Graph and Vector DBs..."):
                try:
                    asyncio.run(ingest_guidelines())
                    st.success("Ingestion Complete!")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
        else:
            st.warning("Please upload PDF files first.")

# Main UI: Patient Case & Query
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("👤 Patient Case")
    age = st.number_input("Age", min_value=1, max_value=120, value=65)
    sex = st.selectbox("Sex", ["Male", "Female", "Other"])
    bp = st.text_input("Blood Pressure (e.g., 155/95)", value="155/95")
    ldl = st.text_input("LDL-C Level (mg/dL)", value="190")
    comorbidities = st.text_area("Comorbidities (e.g., Smoker, CKD)", value="Smoker")
    
    patient_summary = f"{age}yo {sex}, {comorbidities}, BP {bp}, LDL {ldl}"

with col2:
    st.subheader("🔍 Clinical Query")
    query = st.text_input(
        "What is your management question?", 
        placeholder="e.g., Optimal hypertension management"
    )
    
    if st.button("Generate Recommendation"):
        if query:
            with st.spinner("Analyzing guidelines..."):
                # Execute our Hybrid RAG Pipeline
                response, docs, variants = asyncio.run(rag_response(query, patient_summary))
                
                st.markdown("### 📋 Management Recommendation")
                st.write(response)
                
                with st.expander("📚 View Cited Sources"):
                    for doc in docs:
                        source = doc.metadata.get('source', 'Unknown')
                        st.info(f"**Source:** {source}\n\n**Content Snippet:** {doc.page_content[:300]}...")
                
                with st.expander("🔄 Query Expansion Variants"):
                    st.write("The system expanded your query to improve recall:")
                    for v in variants:
                        st.code(v)
        else:
            st.error("Please enter a query.")

st.divider()
st.caption("Note: This is a Decision Support System. Final clinical judgment remains with the physician.")